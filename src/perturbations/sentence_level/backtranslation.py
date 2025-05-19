import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from typing import List, Dict, Any
from src.perturbations.text_perturbation import TextPerturbation


GROUP2LANG = {
    1: ["da", "nl", "de", "is", "no", "sv", "af"],
    2: ["ca", "ro", "gl", "it", "pt", "es"],
    3: ["bg", "mk", "sr", "uk", "ru"],
    4: ["id", "ms", "th", "vi", "mg", "fr"],
    5: ["hu", "el", "cs", "pl", "lt", "lv"],
    6: ["ka", "zh", "ja", "ko", "fi", "et"],
    7: ["gu", "hi", "mr", "ne", "ur"],
    8: ["az", "kk", "ky", "tr", "uz", "ar", "he", "fa"],
    }
SHORT2FULL = {
    'de': 'German', 
    'en': 'English',
    'fr': 'French',
    'pl': 'Polish',
    'ko': 'Korean'
}
LANG2GROUP = {lang: str(group) for group, langs in GROUP2LANG.items() for lang in langs}


class Backtranslation(TextPerturbation):
    TYPE = "sentences_backtranslation"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.inter_language = config.inter_language
        self.source_language = config.source_language
        self.max_new_tokens = config.max_new_tokens
        self.num_beams = config.num_beams
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.batch_size = config.batch_size
        self.model_name = self._get_full_model_name(config.model_name, config.inter_language)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
        

    def _get_full_model_name(self, base_model_name: str, inter_language: str) -> str:
        
        last_part = base_model_name.split('-')[-1]
        if last_part == "Group":
            group_id = LANG2GROUP[inter_language]
            full_model_name = f"{base_model_name}{group_id}"
        else:
            full_model_name = base_model_name

        return full_model_name

    def _get_prompt(self, source_language: str, target_language: str, source_sentence: str):
        prompt= (
            f"Translate this from {source_language} to {target_language}:"
            f"\n{source_language}: {source_sentence}\n{target_language}:"
            )

        return prompt

    def _prepare_model_input(self, tokenizer: PreTrainedTokenizer, prompts: List[str]) -> Dict[str, torch.Tensor]:
        prompts = [[{"role": "user", "content": prompt}] for prompt in prompts]
        prompts = tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompts, return_tensors="pt", padding=True)

        return input_ids

    def _generate_model_output(self, prompts: List[str]) -> str:
        
        input = self._prepare_model_input(self.tokenizer, prompts).input_ids.cuda()
        # Translation
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input, 
                num_beams=self.num_beams, 
                max_new_tokens=self.max_new_tokens, 
                do_sample=True, 
                temperature=self.temperature, 
                top_p=self.top_p
            )

            output = self.tokenizer.batch_decode(generated_ids[:, input.shape[1]: ], skip_special_tokens=True)
        
        return output
 
    def __call__(
            self, 
            queries: List[str],
    ) -> List[str]:
        backtranslated_queries = []

        queries = [self._get_prompt(SHORT2FULL[self.source_language], SHORT2FULL[self.inter_language], query) for query in queries]

        for batch_index in tqdm.tqdm(range(0, len(queries), self.batch_size)):
            batch_queries = queries[batch_index: batch_index+self.batch_size]
            translated_batch = self._generate_model_output(batch_queries)
            translated_batch = [self._get_prompt(SHORT2FULL[self.inter_language], SHORT2FULL[self.source_language], query) for query in translated_batch]
            backtranslated_batch = self._generate_model_output(translated_batch)
            backtranslated_queries.extend(backtranslated_batch)
        
        return backtranslated_queries
