import random
from typing import Any

from tqdm.auto import tqdm

from src.perturbations.text_perturbation import TextPerturbation


class WordsJoining(TextPerturbation):
    """Deleting given percentage (perturbation strength) of whitespaces in the sequence"""
    TYPE = "words_joining"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.perturbation_strength = config["perturbation_strength"]

    def __call__(self, queries: list[str]):
        processed_queries = []
        for query in tqdm(queries, self.TYPE):
            if not any(char.isspace() for char in query):
                modified_query = query
            else:
                whitespace_indices = [i for i, char in enumerate(query) if char.isspace()]
                num_to_remove = max(int(len(whitespace_indices) * self.perturbation_strength), 1) 
                selected_indices = random.sample(whitespace_indices, num_to_remove)
                modified_query = ''.join(char for i, char in enumerate(query) if i not in selected_indices)

            processed_queries.append(modified_query)
        return processed_queries
