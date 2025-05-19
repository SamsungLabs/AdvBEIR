import random
import math
import string
from typing import Any, Dict
from typing import List

from src.perturbations.text_perturbation import TextPerturbation
from src.constants import RANDOM_STATE

class RandomInsert(TextPerturbation):
    """
    Inserts a percentage (perturbation_strength) of additional random characters in the query. 
    """

    TYPE="characters_random_insert"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.perturbation_strength = config["perturbation_strength"]
        self.sampling_container = string.ascii_lowercase + string.ascii_uppercase + string.digits
        random.seed(RANDOM_STATE)

    def __call__(self, queries: List[str]) -> List[str]:
        perturbed_queries = []

        for query in queries:
            query = query.strip()
            editable_indices = [i for i, char in enumerate(query) if not char.isspace()]
            editable_indices.append(len(query))
            num_chars_to_insert = min(math.ceil(len(editable_indices) * self.perturbation_strength), len(editable_indices))
            
            query_chars = list(query)
            while num_chars_to_insert != 0:
                insert_idx = random.choice(editable_indices)
                query_chars.insert(insert_idx, random.choice(self.sampling_container))
                editable_indices = [index + 1 if index > insert_idx else index for index in editable_indices]
                editable_indices.append(insert_idx + 1)
                num_chars_to_insert -= 1
        
            perturbed_queries.append("".join(query_chars))

        return perturbed_queries
