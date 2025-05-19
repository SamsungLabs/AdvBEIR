import random
from typing import Any

from tqdm.auto import tqdm

from src.perturbations.text_perturbation import TextPerturbation


class WordDuplication(TextPerturbation):
    """Inserting duplicates of words from the query in its random places (but far from words indicating negations),
       the number of duplicates is determined by perturbation_strength, the distance from negation is defined
       by permitted_negation_distance."""
    TYPE = "words_duplication"
       
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.perturbation_strength = config["perturbation_strength"]
        self.permitted_negation_distance = config["permitted_negation_distance"]
        self.negations = {"not", "no", "never", "none", "nothing", "nobody", "neither", "nor"}

    def __call__(self, queries):
        processed_queries = []
        for query in tqdm(queries, self.TYPE):
            words = query.split()
        
            if len(words) < 2:  # Avoid modifying very short queries
                processed_queries.append(query)
                continue
            
            negation_indices = [
                i for i, word in enumerate(words) if word.lower() in self.negations or word.endswith("n't")
            ]
            valid_words = [word for i, word in enumerate(words) if i not in negation_indices]
            num_modifications = int(max(self.perturbation_strength * len(words), 1))
            
            for _ in range(num_modifications):
                valid_indices = [
                    i
                    for i, word in enumerate(words)
                    if word in valid_words
                    and all(
                        abs(i - negation_index) > self.permitted_negation_distance
                        for negation_index in negation_indices
                    ) # avoid inserting near negations to maintain logical consistency of the sentence
       
                ]
                if not valid_indices:
                    break
                
                insertion_index = random.choice(valid_indices)
                word = random.choice(valid_words)
                words.insert(insertion_index, word)
                negation_indices = [i + 1 if i >= insertion_index else i for i in negation_indices]
        
            processed_queries.append(" ".join(words))
        return processed_queries
