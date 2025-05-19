import random
import math
from typing import Any, Dict
from typing import List

from src.perturbations.text_perturbation import TextPerturbation
from src.constants import RANDOM_STATE

class RandomDelete(TextPerturbation):
    """
    Deletes a percentage (perturbation_strength) of characters in the query. 
    """

    TYPE="characters_random_delete"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.perturbation_strength = config["perturbation_strength"]
        random.seed(RANDOM_STATE)

    def __call__(self, queries: List[str]) -> List[str]:
        perturbed_queries = []

        for query in queries:
            query = query.strip()
            editable_indices = [i for i, char in enumerate(query) if not char.isspace()]
            num_chars_to_delete = min(math.ceil(len(editable_indices) * self.perturbation_strength), len(editable_indices))
            indices_to_remove = random.sample(editable_indices, k=num_chars_to_delete)
            perturbed_query = "".join([char for i, char in enumerate(query) if i not in indices_to_remove])
            perturbed_queries.append(perturbed_query)

        return perturbed_queries
