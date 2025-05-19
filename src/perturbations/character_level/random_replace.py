import random
import math
import string
from typing import Any, Dict
from typing import List

from src.perturbations.text_perturbation import TextPerturbation
from src.constants import RANDOM_STATE

class RandomReplace(TextPerturbation):
    """
    Replaces a percentage (perturbation_strength) of selected characters of the query with the random ones. 
    """

    TYPE="characters_random_replace"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.perturbation_strength = config["perturbation_strength"]
        self.sampling_container = set(string.ascii_lowercase + string.ascii_uppercase + string.digits)
        random.seed(RANDOM_STATE)

    def __call__(self, queries: List[str]) -> List[str]:
        perturbed_queries = []

        for query in queries:
            query = query.strip()
            editable_indices = [i for i, char in enumerate(query) if not char.isspace()]
            num_chars_to_replace = min(math.ceil(len(editable_indices) * self.perturbation_strength), len(editable_indices))
            
            query_chars = list(query)
            while num_chars_to_replace != 0:
                replace_idx = random.choice(editable_indices)
                replace_char = random.choice(tuple(self.sampling_container - {query_chars[replace_idx]}))
                query_chars[replace_idx] = replace_char
                editable_indices.remove(replace_idx)
                num_chars_to_replace -= 1
        
            perturbed_queries.append("".join(query_chars))

        return perturbed_queries
