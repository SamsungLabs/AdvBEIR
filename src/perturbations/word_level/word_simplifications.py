import random
from abc import ABC, abstractmethod
from typing import Any

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tqdm.auto import tqdm

from src.perturbations.text_perturbation import TextPerturbation


class WordSimplification(TextPerturbation, ABC):
    """Base class for word simplification perturbations."""
    
    def __init__(self, config: dict[str, Any]):
        nltk.download("stopwords")
        self.config = config
        self.perturbation_strength = config["perturbation_strength"]
        self.stop_words = set(stopwords.words(config["language"]))
        self.nlp = spacy.load("en_core_web_sm")

    def __call__(self, queries: list[str]) -> list[str]:
        processed_queries = []
        for query in tqdm(queries, self.TYPE):
            doc = self.nlp(query)
            words = [token.text for token in doc]
            valid_indices = [i for i, token in enumerate(doc) if not token.is_stop and not token.is_punct]
            
            if len(valid_indices) == 0:
                processed_queries.append(query)
                continue
            
            num_modifications = max(int(self.perturbation_strength * len(valid_indices)), 1)
            indices_to_simplify = random.sample(valid_indices, num_modifications)
            
            for i in indices_to_simplify:
                simplified_word = self.simplify_word(doc[i].text)

                if doc[i].text.istitle():
                    simplified_word = simplified_word.capitalize()
                elif doc[i].text.isupper():
                    simplified_word = simplified_word.upper()
                    
                words[i] = simplified_word
            
            simplified_query = "".join([word + doc[i].whitespace_ for i, word in enumerate(words)])
            processed_queries.append(simplified_query)
            
        return processed_queries

    @abstractmethod
    def simplify_word(self, word: str) -> str:
        pass


class Stemming(WordSimplification):
    """Stems percentage (perturbation_strength) of query words which are not stop words."""
    TYPE = "words_stemming"
    
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.stemmer = PorterStemmer()

    def simplify_word(self, word: str) -> str:
        return self.stemmer.stem(word)

class Lemmatization(WordSimplification):
    """Lemmatizes percentage (perturbation_strength) of query words which are not stop words."""
    TYPE = "words_lemmatization"

    def simplify_word(self, word: str) -> str:
        return self.nlp(word)[0].lemma_