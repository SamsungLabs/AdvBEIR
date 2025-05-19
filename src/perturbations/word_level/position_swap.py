import nlpaug.augmenter.word as naw
from src.perturbations.perturbation_interface import PerturbationInterface


class PositionSwap(PerturbationInterface):
    """Swapping percentage (perturbation_strength) of adjacent words in a sentence.
    Performed with RandomWordAug augmenter from `nlpaug` library."""

    TYPE = "words_position_swap"

    def __init__(self, config: dict):
        super().__init__(config)
        self.level = "word_level"
        self.perturbation_strength = config["perturbation_strength"]
        self.perturbation_args = {"aug_p": self.perturbation_strength, "aug_min": 1, "action": "swap"}
        self.engine = naw.RandomWordAug(**self.perturbation_args)
