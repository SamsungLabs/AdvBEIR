import random

from typing import Any
from src.perturbations.text_perturbation import TextPerturbation
from src.constants import RANDOM_STATE


class Capitalization(TextPerturbation):
    """Capitalizing / lowercasing given percentage (perturbation_strengh) of characters."""
    TYPE = "characters_capitalization"
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.perturbation_strength = config["perturbation_strength"]
        random.seed(RANDOM_STATE)

    def __call__(self, queries: list[str]):
        processed_queries = []
        for query in queries:
            characters_indices = [i for i, char in enumerate(query) if char.isalpha()]
            num_to_modify = max(int(len(characters_indices) * self.perturbation_strength), 1) 
            selected_indices = random.sample(characters_indices, num_to_modify)
            
            modified_chars = [char for char in query]
            for idx in selected_indices:
                selected_char = query[idx] 
                modified_chars[idx] = selected_char.upper() if selected_char.islower() else selected_char.lower()
                
            modified_query = "".join(modified_chars)
            processed_queries.append(modified_query)
        return processed_queries
