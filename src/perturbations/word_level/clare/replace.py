from typing import List, Tuple

import spacy

from .action_base import CLAREBaseMaskReplacer


class CLAREReplace(CLAREBaseMaskReplacer):
    @staticmethod
    def get_maskable_indices(doc: spacy.tokens.Doc, modified_indices: List[int]) -> List[int]:
        return [i for i, token in enumerate(doc) if (token.is_alpha and i not in modified_indices)]

    def get_masked_text(self, doc: spacy.tokens.Doc, masked_idx: int) -> Tuple[str, str]:
        masked_text = self.get_text_with_replaced_word(doc, masked_idx, self.mask_token)
        return masked_text, doc[masked_idx].text

    @staticmethod
    def get_text_with_replaced_word(doc: spacy.tokens.Doc, index: int, new_word: str) -> str:
        subtexts = [token.text_with_ws for token in doc]
        subtexts[index] = new_word + doc[index].whitespace_
        return "".join(subtexts)

    @staticmethod
    def update_modified_indices(modified_indices: List[int], modified_idx: int) -> None:
        modified_indices.append(modified_idx)  # Simply adding the index of replaced word
