from typing import List, Any

from tqdm.auto import tqdm

from src.perturbations.text_perturbation import TextPerturbation


class PerturbationInterface(TextPerturbation):
    """Common interface for some of the perturbation engines."""
    def __init__(self, config):
        self.config = config
        self.perturbation_strength = None
        self.perturbation_args = dict()
        self.engine = None

    def get_metadata(self) -> dict[str, Any]:
        metadata = {
            "perturbation_method": self.TYPE ,
            "perturbation_strength": self.perturbation_strength
        }
        return metadata

    def __call__(self, queries: List[str]) -> List[str]:
        perturbed_queries = []
        for query in tqdm(queries, self.TYPE, disable=(self.level == "character_level")):
            perturbed_query = self.engine.augment(query)[0]
            perturbed_queries.append(perturbed_query)
        return perturbed_queries
