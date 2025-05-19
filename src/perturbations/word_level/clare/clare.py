import logging
import math
import random
import re
from typing import Any, List

from sentence_transformers import SentenceTransformer
import spacy
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.constants import RANDOM_STATE
from src.perturbations.text_perturbation import TextPerturbation
from .insert import CLAREInsert
from .merge import CLAREMerge
from .replace import CLAREReplace

# Default values used if sth is not defined in config
DEFAULTS = {
    "spacy_model": "en_core_web_sm",
    "mlm_model": "answerdotai/ModernBERT-large",
    "st_encoder": "all-MiniLM-L6-v2",
    "top_p": 0.005,
    "top_k": 5,
    "min_sim_score": 0.75,
}


class CLARE(TextPerturbation):
    TYPE = "words_clare"
    ACTION_CLASSES = (CLAREReplace, CLAREInsert, CLAREMerge)

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.fill_config_with_defaults()

        self.strength = config["perturbation_strength"]

        self.nlp = spacy.load(config["spacy_model"])

        tokenizer = AutoTokenizer.from_pretrained(config["mlm_model"], use_fast=True)
        model = AutoModelForMaskedLM.from_pretrained(config["mlm_model"], device_map="auto")

        st_encoder = SentenceTransformer(config["st_encoder"])

        self.actions = [ac(config, self.nlp, tokenizer, model, st_encoder) for ac in self.ACTION_CLASSES]

    def __call__(self, texts: List[str]) -> List[str]:
        random.seed(RANDOM_STATE)
        modified_texts = []
        for text_id, text in enumerate(tqdm(texts, self.TYPE)):
            # In order not to complicate spacys tokenization we separate all dash-joined words 
            src_text = re.sub(r"([a-zA-Z])-([a-zA-Z])", r"\1 \2", text)

            doc = self.nlp(src_text)
            num_words = len([i for i, token in enumerate(doc) if token.is_alpha])
            # No modifications for one word texts
            num_modifications = math.ceil(num_words * self.strength) if num_words > 1 else 0 

            modified_text = src_text
            modified_indices = []  # Need to keep track to modify words only once (as stated in the paper)
            for modification_num in range(num_modifications):
                random.shuffle(self.actions)
                for action in self.actions:
                    new_text = action(src_text, modified_text, modified_indices)
                    if new_text is not None:
                        # Text modified successfully
                        break
                    else:
                        # Try another action, because no modifications were possible for the current one
                        continue

                if new_text is None:
                    curr_strength = round(modification_num / num_words, 4)
                    logging.warning(
                        f"Unable to reach perturbation strenght of {self.strength} for text number {text_id}. "
                        f"Reached strength of {curr_strength} after which no more modifications were possible."
                    )
                    break
                else:
                    modified_text = new_text

            modified_texts.append(modified_text)

        return modified_texts

    def fill_config_with_defaults(self) -> None:
        for param_name, default_value in DEFAULTS.items():
            self.config[param_name] = self.config.get(param_name, default_value)
