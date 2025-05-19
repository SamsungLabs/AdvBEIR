from .character_level.capitalization import Capitalization
from .character_level.punctuation import Punctuation
from .character_level.char_replace import ShiftKeyMiss, OCRError
from .character_level.random_replace import RandomReplace
from .character_level.random_delete import RandomDelete
from .character_level.random_insert import RandomInsert
from .character_level.keyboard_replace import KeyboardReplace
from .character_level.keyboard_insert import KeyboardInsert
from .character_level.neighbour_swap import NeighbourSwap
from .word_level.words_joining import WordsJoining
from .word_level.word_duplication import WordDuplication
from .word_level.word_simplifications import Lemmatization, Stemming
from .word_level.position_swap import PositionSwap
from .word_level.clare.clare import CLARE
from .sentence_level.backtranslation import Backtranslation
from .sentence_level.paraphrase import Paraphraser


PERTURBATION_CLASSES = (
    Capitalization,
    KeyboardInsert,
    KeyboardReplace,
    NeighbourSwap,
    OCRError,
    Punctuation,
    RandomDelete,
    RandomInsert,
    RandomReplace,
    ShiftKeyMiss,
    CLARE,
    WordDuplication,
    WordsJoining,
    Lemmatization,
    PositionSwap,
    Stemming,
    # Backtranslation,
    Paraphraser
)

AVAILABLE_PERTURBATIONS = tuple(perturb_cls.TYPE for perturb_cls in PERTURBATION_CLASSES)

PERTURBATION_NAME_TO_CLS = {perturb_cls.TYPE: perturb_cls for perturb_cls in PERTURBATION_CLASSES}
PERTURBATION_NAME_TO_ID = {perturb_cls.TYPE: f"P{i+1}" for i, perturb_cls in enumerate(PERTURBATION_CLASSES)}
