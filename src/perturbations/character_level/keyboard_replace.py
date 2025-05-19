from src.perturbations.character_level.char_replace import CharacterReplacementWithMapping

class KeyboardReplace(CharacterReplacementWithMapping):
    """Replacing specific percentage (perturbation_strength) of characters with their neighbours located
    on the keyboard. It simulates an error when a person clicks a neighbour character instead of the right one."""
    TYPE = "characters_keyboard_replace"