from typing import List

import spacy

from .action_base import CLAREBaseMaskReplacer


class CLAREMerge(CLAREBaseMaskReplacer):
    @classmethod
    def get_maskable_indices(cls, doc: spacy.tokens.Doc, modified_indices: List[int]) -> List[int]:
        """In this case maskable indices indicate that word at index (and the following one) can be merged"""
        maskable_indices = []
        for idx in range(len(doc) - 1):
            idx_is_mergable = (
                doc[idx].is_alpha
                and idx not in modified_indices
                and doc[idx + 1].is_alpha
                and idx + 1 not in modified_indices
            )
            if idx_is_mergable:
                maskable_indices.append(idx)

        return maskable_indices

    @staticmethod
    def get_text_with_merged_words(doc: spacy.tokens.Doc, index: int, new_word: str) -> str:
        preceeding = [token.text_with_ws for token in doc[:index]]

        # Skipping tokens at index and index+1 which will be replaced with new_word
        following = [token.text_with_ws for token in doc[index + 2 :]]

        # Whitespace of the second token is inherited by the new wrod
        subtexts = preceeding + [new_word + doc[index + 1].whitespace_] + following
        return "".join(subtexts)

    def get_masked_text(self, doc: spacy.tokens.Doc, masked_idx: int) -> str:
        masked_text = self.get_text_with_merged_words(doc, masked_idx, self.mask_token)
        replaced_part = doc[masked_idx].text_with_ws + doc[masked_idx + 1].text
        return masked_text, replaced_part

    @staticmethod
    def update_modified_indices(modified_indices: List[int], modified_idx: int) -> None:
        modified_indices.append(modified_idx)  # Including the index of word resulting from merge

        # Two words were replaced with one so we need to shift all following indices to the left
        for i in range(len(modified_indices)):
            if modified_indices[i] > modified_idx:
                modified_indices[i] -= 1
