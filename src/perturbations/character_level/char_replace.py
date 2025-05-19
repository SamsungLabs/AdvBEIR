import random
import math
from typing import Any, Dict
from abc import ABC
from typing import List
import json


from src.perturbations.text_perturbation import TextPerturbation
from src.constants import RANDOM_STATE

class CharacterReplacementWithMapping(TextPerturbation, ABC):
    """
    Perturbs text by replacing characters with randomly choosen candidates.
    
    Config should include:
    - `replacement_map_fpath` is a file path to the mapping that candidate replacements are defined in.
    - `perturbation_strength` defines the fraction of the original query characters that will be replaced.
    - `case_sensitivity` indicates whether the replacements are chosen based on original casing (when `True`) 
    or lowercased (when `False`) queries.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.perturbation_strength = config["perturbation_strength"]
        self.case_sensitive = config["case_sensitive"]

        with open(self.config["replacement_map_fpath"]) as f:
            self._replacement_map = json.load(f)
        
        random.seed(RANDOM_STATE)  # for reproducibility

    def __call__(self, queries: List[str]) -> List[str]:
        perturbed_queries = []
        for query in queries:
            orig_chars = list(query if self.case_sensitive else query.lower())
            replacable_idxs = [i for i, c in enumerate(orig_chars) if c in self._replacement_map]

            # Limiting number of replacements to all possible ones, applying at least one perturbation if possible
            num_replacements = min(math.ceil(len(replacable_idxs) * self.perturbation_strength), len(replacable_idxs))
            idxs_to_replace = sorted(random.sample(replacable_idxs, k=num_replacements))

            new_chars = list(query)
            for idx_to_replace in idxs_to_replace:
                orig_char = orig_chars[idx_to_replace]
                new_chars[idx_to_replace] = self.get_replacement(orig_char)
            perturbed_queries.append(''.join(new_chars))
        return perturbed_queries
       
    def get_replacement(self, orig_char: str) -> str:
        """
        Returns a perturbed character (replacement) for the original character. 
        `None` is returned if there are no candidates defined in the `self.replacement_map`.  
        """
        possible_replacements = self._replacement_map.get(orig_char, [None])
        replacement = random.choice(possible_replacements)
        return replacement


class ShiftKeyMiss(CharacterReplacementWithMapping):
    TYPE = 'characters_shift_key_miss'


class OCRError(CharacterReplacementWithMapping):
    TYPE = 'characters_ocr_error'
