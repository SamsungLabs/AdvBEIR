import random
from textwrap import dedent
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.constants import RANDOM_STATE
from src.perturbations.text_perturbation import TextPerturbation


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(RANDOM_STATE)


class Paraphraser(TextPerturbation):
    TYPE="sentences_paraphrase"

    def __init__(self, config):
        self.config = config
        self.batch_size = config.generation_params.batch_size
        self.do_sample = config.generation_params.do_sample
        self.max_new_tokens = config.generation_params.max_new_tokens
        self.temperature = config.generation_params.temperature
        self.top_p = config.generation_params.top_p
        self.hf_llm_path = config.hf_llm_path

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_llm_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_llm_path, device_map="auto")
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def process_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
    ) -> list[str]:

        if self.tokenizer.chat_template:
            inputs = [
                self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                for messages in batch_messages
            ]

        else:
            inputs = [
                dedent("<|begin_of_text|>" + (x[0]["content"] + " " + x[1]["content"]))
                for x in batch_messages
            ]

        tokenized_inputs = self.tokenizer(inputs, padding=True, return_tensors="pt")
        tokenized_inputs.to(device=DEVICE)
        generate_kwargs = dict(
            **tokenized_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        outputs = self.model.generate(**generate_kwargs)

        return [
            self.tokenizer.decode(
                output[tokenized_inputs["input_ids"][0].shape[-1] :], skip_special_tokens=True
            )
            for output in outputs
        ]

    def __call__(self, queries: List[str], dataset_names: List[str]):
        dataset_types = [x if ("cqadupstack" not in x) else "cqadupstack" for x in dataset_names]
        prompts = [
            (f"{self.config.prompt.prompt_base} {self.config.prompt.prompt_types[x]}") for x in dataset_types
        ]
        input_messages = []
        for query, prompt in zip(queries, prompts):
            input_messages.append(
                [
                    {"role": "system", "content": dedent(prompt)},
                    {"role": "user", "content": f"paraphrase the given document: {query}"},
                ]
            )

        responses = []
        with tqdm(total=len(input_messages), desc=self.TYPE) as pbar:
            for i in range(0, len(input_messages), self.batch_size):
                batch = input_messages[i : i + self.batch_size]
                generations = self.process_batch(batch)
                responses.extend(generations)
                pbar.update(len(batch)) 

        return responses
