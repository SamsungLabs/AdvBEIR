import os
import json
import random

from typing import List, Optional
from src.constants import CONFIGS_DIR


class BigThumbAug:
    def __init__(self, **config):
        self.perturbation_fraction = config['perturbation_fraction']
        self.KEYBOARD = self.load_keyboard(
            os.path.join(CONFIGS_DIR, "keyboard_neighbours_en.json")
        )

    @staticmethod
    def load_keyboard(path):
        with open(path, 'r') as file:
            return json.load(file)

    def get_neighbours(self, target):
        return self.KEYBOARD.get(target)

    @staticmethod
    def pick_random(container) -> str:
        return random.choice(container)

    def draw_place(self, text) -> Optional[int]:
        """return an index of a character that occurs in the KEYBOARD and could be perturbed"""
        valid_chars = [i for i, char in enumerate(text) if char in self.KEYBOARD.keys()]
        if valid_chars:
            return random.choice(valid_chars)
        else:
            return None

    @staticmethod
    def insert_letter(word, letter, idx):
        return word[:idx] + letter + word[idx:]

    @staticmethod
    def replace_element(container, idx, elem):
        container[idx] = elem
        return container

    @staticmethod
    def calc_chars_per_mistake(perturbation_fraction: int):
        return round(1 / perturbation_fraction)

    def split_text(self, text):
        chars_per_mistake = BigThumbAug.calc_chars_per_mistake(self.perturbation_fraction)
        results = []
        for i in range(0, len(text), chars_per_mistake):
            results.append(text[i:i + chars_per_mistake])
        return results

    def augment(self, text: str) -> List[str]:
        results = []
        for text_batch in self.split_text(text):
            char_idx = self.draw_place(text_batch)  # draw a character from text to perturbate
            if not char_idx:
                results.append(text_batch)
                continue
            neighbours = self.get_neighbours(text_batch[char_idx])
            perturbed_batch = self.insert_letter(
                text_batch, self.pick_random(neighbours), char_idx
            )
            results.append(perturbed_batch)
        return ["".join(results)]
