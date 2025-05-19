from typing import List, Tuple

import spacy

from .action_base import CLAREBaseMaskReplacer


class CLAREInsert(CLAREBaseMaskReplacer):
    @staticmethod
    def get_maskable_indices(doc: spacy.tokens.Doc, modified_indices: List[int]) -> List[int]:
        # Modified indices do not affect potential insertion places
        return [i for i in range(len(doc) + 1)]  # +1 to enable insertion after last word

    @staticmethod
    def get_text_with_inserted_word(doc: spacy.tokens.Doc, index: int, new_word: str):
        """The word gets inserted before given idx (similarily to `list.insert()`)."""

        # The closest token preceeding inserted word needs special treatment so it is excluded here
        preceeding_end = max(index - 1, 0)
        preceeding = [token.text_with_ws for token in doc[:preceeding_end]]

        if index > 0:
            # New word has a predecessor - trailing whitespaces must be modified to preserve logical spacing
            predecessor = doc[index - 1]

            pred_is_open_bracket = predecessor.is_bracket and predecessor.is_left_punct
            # `is_left_punct` has the same value for opening and closing quotes so manual check is necessary
            # Ouote is open if it is either the first token or is preceeded with " "
            pred_is_open_qoute = predecessor.is_quote and (index == 1 or doc[index - 2].whitespace_ == " ")

            if pred_is_open_bracket or pred_is_open_qoute:
                # Whitespace rule is different from default for open quotes and brackets
                inherited_ws = " "  #
                new_predecessor_ws = ""  # no whitespace - new world will come right after the quote/bracket
            else:
                # By default " " preceeds new word and its trailing whitespace is inherited from the predecessor
                inherited_ws = predecessor.whitespace_
                new_predecessor_ws = " "

            preceeding.append(predecessor.text + new_predecessor_ws)
        else:
            # Word inserted at the start should be followed by " "
            inherited_ws = " "

        following = [token.text_with_ws for token in doc[index:]]

        subtexts = preceeding + [new_word + inherited_ws] + following
        return "".join(subtexts)

    def get_masked_text(self, doc: spacy.tokens.Doc, masked_idx: int) -> Tuple[str, str]:
        masked_text = self.get_text_with_inserted_word(doc, masked_idx, self.mask_token)
        return masked_text, ""  # empty str returned in tuple since no part of the text is replaced

    @staticmethod
    def update_modified_indices(modified_indices: List[int], modified_idx: int) -> None:
        # A new word was inserted so we need to shift all following indices to the right
        for i in range(len(modified_indices)):
            if modified_indices[i] >= modified_idx:
                modified_indices[i] += 1

        modified_indices.append(modified_idx)  # Including the index of newly insterted word
