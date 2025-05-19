import pandas as pd
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from textwrap import dedent
from tqdm import tqdm
import os
import re

GENERATION_KWARGS = {
    "do_sample": True,
    "temperature": 0.6,
    "max_new_tokens": 2048,
    "top_p": 0.9
}
SAVE_INTERVAL = 10
BATCH_SIZE = 16
PROMPT = """"You are a linguistic evaluator. Your task is to assess whether a given paraphrase accurately
          conveys the same meaning as the original sentence.

          Instructions:

          1. The paraphrase must retain the same semantic meaning as the original.
          2. It must include exactly the same amount of information as the original, without adding or 
          omitting knowledge.
          3. It must not imply much additional context or introduce many new interpretations."

          Do not penalize the paraphrase in the following situations:
          Acronyms and Expansions: When the paraphrase substitutes between full names and acronyms.
          Example:
          Original: 'Organisation for Economic Co-operation and Development'
          Paraphrase: 'What does OECD stand for?'

          Synonyms or Equivalent Phrases: When words or phrases are substituted with synonyms that preserve meaning.
          Example:
          Original: 'How to cook pasta quickly?'
          Paraphrase: 'How to make pasta fast?'

          Reordering or Structural Variations: When words are reordered or the sentence is restructured, retaining the same intent.
          Example:
          Original: 'What is the capital of France?'
          Paraphrase: 'France's capital is what?'

          Implicit vs. Explicit Questions: When phrasing switches between implicit and explicit forms without changing intent.
          Example:
          Original: 'What is Einstein known for?'
          Paraphrase: 'What scientific contributions is Einstein famous for?'

          Conversion Between Formats (e.g., Questions and Statements): When a question is transformed into a statement or vice versa.
          Example:
          Original: 'Explain photosynthesis.'
          Paraphrase: 'What is photosynthesis?'

          Variations in Focus or Emphasis: When emphasis shifts between parts of the sentence without altering meaning.
          Example:
          Original: 'Who discovered gravity?'
          Paraphrase: 'Gravity was discovered by whom?'

          Variations in Granularity: When slight changes in specificity occur, but context implies equivalence.
          Example:
          Original: 'How many planets are in the solar system?'
          Paraphrase: 'How many planets orbit the Sun?'

          Simplifications That Retain Meaning: When language is condensed or simplified while keeping the same intent.
          Example:
          Original: 'Steps to create a new email account on Gmail?'
          Paraphrase: 'How to set up Gmail?'

          Alternative Representations of Numerical Information: When numerical formats or ranges are switched.
          Example:
          Original: 'What happened in the 20th century?'
          Paraphrase: 'What events occurred between 1901 and 2000?'

          Contextual Inferences with Unambiguous Terms: When shorter or implicit expressions are used, remaining clear in context.
          Example:
          Original: 'What is the full form of NATO?'
          Paraphrase: 'What does NATO stand for?'

          Differences in Question Type (Why/How): When closely related question types are switched but lead to the same answer.
          Example:
          Original: 'Why is the sky blue?'
          Paraphrase: 'How does the sky appear blue?'
        
          If the paraphrase changed the style of original sentence to the search query, e.g., 'What is the capital of France?' to 'Search for capital of France', then this kind transformation is acceptable.

          For each pair of sentences, return a Python dictionary object with:
          label: 0 if the paraphrase is accurate, 1 otherwise. 
          If the paraphrase is exactly the same or very close to the original sentence, the label should be set to "duplicate".

          Three examples:

          Original sentence: 'The cat is sitting on the mat.'
          Paraphrase: 'The mat has a cat sitting on it.'
          Output: {"label": 0}

          Original sentence: 'The cat is sitting on the mat.'
          Paraphrase: 'The cat is on the mat and it looks hungry. The dog wants to go to the pet store.'
          Output: {"label": 1}
          
          Original sentence: 'The cat is sitting on the mat.'
          Paraphrase: 'The cats are sitting on the mat.'
          Output: {"label": "duplicate"}
          """

def process_batch(
    batch_messages: list[list[dict[str, str]]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
) -> list[str]:

    inputs = [
        tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        for messages in batch_messages
    ]

    tokenized_inputs = tokenizer(inputs, padding="longest", return_tensors="pt")
    tokenized_inputs.to(model.device)

    outputs = model.generate(
        **tokenized_inputs,
        max_new_tokens=GENERATION_KWARGS.get("max_new_tokens", 512),
        temperature=GENERATION_KWARGS.get("temperature", 0.6),
        do_sample=GENERATION_KWARGS.get("do_sample", True),
        top_p=GENERATION_KWARGS.get("top_p", 0.9)
    )

    return [
        tokenizer.decode(output[tokenized_inputs["input_ids"][0].shape[-1] :], skip_special_tokens=True)
        for output in outputs
    ]
def main(args):
        benchmark =  pd.read_json(args.paraphrases_path)
        model_name = "_".join(args.model_path.split("/"))
        checkpoints_dir = f"checkpoints_{model_name}"
        os.makedirs(checkpoints_dir, exist_ok=True)

        input_messages = []

        for query, paraphrase in zip(benchmark["query"].tolist(), benchmark["paraphrase"].tolist()):
            input_messages.append(
                    [
                        {"role": "system", "content": dedent(re.sub("\s\s+" , " ", PROMPT))},
                        {"role": "user", "content": 
                            f"""You must response in 1 sentence, do not cite any of the text I mentioned before 
                                in your response. Assess the quality of a given paraphrase based on the 
                                instructions provided earlier. 
                                Original sentence: {query}. \n Paraphrase: {paraphrase}. \n Response:"""},
                    ]
                )
            
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        responses = []
        batch_counter = 0

        for i in tqdm(range(0, len(input_messages), BATCH_SIZE)):
            batch = input_messages[i: i + BATCH_SIZE]
            generations = process_batch(batch, tokenizer, model)
            responses.extend(generations)
            
            # save checkpoint every n batches to avoid loss of progress
            if batch_counter % SAVE_INTERVAL == 0 and batch_counter != 0:
                interval_start = i - BATCH_SIZE * SAVE_INTERVAL
                temp_df = benchmark.iloc[interval_start: i]
                temp_df['response'] = responses[interval_start: i]
                temp_df.to_json(f'{checkpoints_dir}/checkpoint_{batch_counter // SAVE_INTERVAL}.json', 
                                orient="records", 
                                indent=True)
            batch_counter += 1
    
        # save final results
        benchmark["response"] = responses
        benchmark.to_json(f'{checkpoints_dir}/judged_paraphrases.json', orient="records", indent=True)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--paraphrases_path", type=str, required=True, 
                        help="path to the json file with 'query' and 'paraphrase' fields which will be automatically judged")
    parser.add_argument("-m", "--model_path", type=str, required=True, 
                        help="HuggingFace path to the LLM judge, e.g. Qwen/Qwen2.5-72B-Instruct")
    
    args = parser.parse_args()
    main(args)
    