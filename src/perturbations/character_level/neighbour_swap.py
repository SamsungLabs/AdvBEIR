import random
import math
from typing import Any, Dict
from typing import List

from src.perturbations.text_perturbation import TextPerturbation
from src.constants import RANDOM_STATE

class NeighbourSwap(TextPerturbation):
    """
    Swaps a percentage (perturbation_strength) of editable characters (has the neighbouring character on the right) 
    with its right neighbour. 
    """

    TYPE="characters_neighbour_swap"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.perturbation_strength = config["perturbation_strength"]
        random.seed(RANDOM_STATE)

    def __call__(self, queries: List[str]) -> List[str]:
        perturbed_queries = []

        for query in queries:
            query = query.strip()
            valid_indices = [i for i in range(len(query[:-1])) if (not query[i].isspace() and not query[i+1].isspace())]
            num_chars_to_switch = min(math.ceil(len(valid_indices) * self.perturbation_strength), len(valid_indices))

            q_list = list(query)
            random.shuffle(valid_indices)
            for index in valid_indices:
                
                if num_chars_to_switch == 0:
                    break
                
                q_list[index], q_list[index + 1] = q_list[index + 1], q_list[index]
                # remove the token with which our character was swapped, to not manipulate it once again
                if index + 1 in valid_indices:
                    valid_indices.remove(index + 1)
                # the same with the preceding character to not swap it with the already manipulated character   
                if index - 1 in valid_indices:
                    valid_indices.remove(index - 1)
                
                valid_indices.remove(index)
                num_chars_to_switch -= 1
                
            perturbed_queries.append("".join(q_list))
               
        return perturbed_queries
