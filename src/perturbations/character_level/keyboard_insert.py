import random
from src.constants import RANDOM_STATE
from src.perturbations.perturbation_interface import PerturbationInterface
from src.perturbations.character_level.big_thumb.bt_augmenter import BigThumbAug

class KeyboardInsert(PerturbationInterface):
    """For percentage (perturbation_strength) of characters in text,
    insert their keyboard neighbor next to them. Also known as a 'Big Thumb' perturbation method.
    Performed with self-implemented perturbation mechanism."""

    TYPE = "characters_keyboard_insert"

    def __init__(self, config: dict):
        super().__init__(config)
        self.level = "character_level"
        self.perturbation_strength = config["perturbation_strength"]
        self.perturbation_args = {
            "perturbation_fraction": self.perturbation_strength,
        }
        self.engine = BigThumbAug(**self.perturbation_args)
        random.seed(RANDOM_STATE)
