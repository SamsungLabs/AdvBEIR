from abc import ABC, abstractmethod
from typing import List, Any


# Define an abstract class
class TextPerturbation(ABC):
    TYPE = ""
    config = {}

    @abstractmethod
    def __call__(self, queries: List[str]) -> List[str]:
        pass  # This is an abstract method, no implementation here.

    def get_metadata(self) -> dict[str, Any]:
        return self.config
