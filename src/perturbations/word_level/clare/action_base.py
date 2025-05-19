import random
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from src.constants import MAX_SEQ_LENGTH


class CLAREBaseMaskReplacer(ABC):
    """Base class for CLARE Actions that mask text in various ways and replace masked place using MLM"""

    # Single replacement token must start with either uppercased letter or WORDSTART_PREFIX (which is a
    # special tokenization artifact) so that we dont replace with a meaningless subword chunk.
    # Still might get only the first part of word that spans across multiple tokens
    WORDSTART_PREFIX = "Ġ"
    WORD_PREFIX = f"^(?:{WORDSTART_PREFIX}|[A-Z])"  # using "?:" cause we dont care about extracting group

    def __init__(
        self,
        config: dict[str, Any],
        nlp: spacy.language.Language,
        mlm_tokenizer: PreTrainedTokenizerFast,
        mlm_model: PreTrainedModel,  # should be a model with MLM head (`ForMaskedLM` suffix)
        st_encoder: SentenceTransformer,  # used to calculate similarity scores
    ):
        self.config = config

        self.top_p = config["top_p"]
        self.top_k = config["top_k"]
        self.min_sim_score = config["min_sim_score"]

        self.nlp = nlp

        self.mlm_tokenizer = mlm_tokenizer
        self.mask_token = self.mlm_tokenizer.mask_token
        self.mask_token_id = self.mlm_tokenizer.mask_token_id

        self.mlm_model = mlm_model
        self.mlm_model.eval()
        self.device = self.mlm_model.device

        self.st_encoder = st_encoder

    def __call__(self, orig_text: str, src_text: str, modified_indices: List[int]) -> str:
        doc = self.nlp(src_text)
        maskable_word_indices = self.get_maskable_indices(doc, modified_indices)
        random.shuffle(maskable_word_indices)
        for masked_word_idx in maskable_word_indices:
            # Masking the source text
            masked_text, replaced_part = self.get_masked_text(doc, masked_word_idx)

            # Generating MLM predictions for masked token
            input_ = self.mlm_tokenizer(
                masked_text,
                max_length=MAX_SEQ_LENGTH,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                mlm_model_out = self.mlm_model(**input_)

            # Converting logits for masked token to replacements probs
            masked_token_idx = input_["input_ids"][0].tolist().index(self.mask_token_id)
            mask_token_logits = mlm_model_out["logits"][0][masked_token_idx]
            replacements_probs = torch.softmax(mask_token_logits, dim=0).tolist()

            # Creating DataFrame with mask replacements
            replacements_df = (
                # df's indices represent token ids
                pd.DataFrame(replacements_probs, columns=["prob"])
                .loc[lambda df: df["prob"] > self.top_p]
                .assign(tok_text=lambda df: self.mlm_tokenizer.convert_ids_to_tokens(df.index))
                .astype({"tok_text": str})  # this is necessary when no top_p replacements exist
                .loc[lambda df: df["tok_text"].str.contains(self.WORD_PREFIX)]
                .assign(
                    word=lambda df: df["tok_text"].apply(
                        lambda t: self.mlm_tokenizer.convert_tokens_to_string([t]).strip()
                    )
                )
                .loc[lambda df: df["word"].str.isalpha()]  # only words - no numbers, punctuation, etc.
                # potential replacement must differ from the replaced part
                .loc[lambda df: df["word"].str.lower() != replaced_part.lower()]
                .assign(
                    new_text=lambda df: df["word"].apply(lambda w: masked_text.replace(self.mask_token, w))
                )
            )
            

            if replacements_df.empty:
                # No possible replacements at this stage, move to the next maskable word idx
                continue

            # Calculating similarity scores between the original text and new texts with replacements
            embeddings = self.st_encoder.encode([orig_text] + replacements_df.new_text.to_list())
            sim_scores = self.st_encoder.similarity(embeddings[:1], embeddings[1:])[0]

            # Final filtering of possible replacements
            final_replacements_df = (
                replacements_df.assign(sim_score=sim_scores)
                .loc[lambda df: df.sim_score >= self.min_sim_score]
                .sort_values(by="prob", ascending=False)
                .iloc[: self.top_k]
            )

            if final_replacements_df.empty:
                # no possible replacements after final filtering, move to the next maskable word idx
                continue

            # Randomly choosing new text (with already replaced mask token)
            # TODO: Perhaps a greedy approach would be better here? (instead of random choice)
            new_text = final_replacements_df.sample(1, weights="prob").new_text.iloc[0]

            # Since modification took place we need to update modified indices
            self.update_modified_indices(modified_indices, masked_word_idx)

            return new_text

        # If no replacements were possible return None which is a signal to perform other CLARE operation
        return None

    @staticmethod
    @abstractmethod
    def get_maskable_indices(doc: spacy.tokens.Doc) -> List[int]:
        pass

    @abstractmethod
    def get_masked_text(self, doc: spacy.tokens.Doc, masked_idx: int) -> Tuple[str, str]:
        pass

    @staticmethod
    @abstractmethod
    def update_modified_indices(modified_indices: List[int], modified_idx: int) -> None:
        pass
