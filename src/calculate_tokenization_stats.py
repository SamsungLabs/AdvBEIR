import argparse
import os
from typing import List, Tuple

import Levenshtein
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.constants import PERTURBED_BEIR_FPATH, RESULTS_DIR


TOKENIZER_NAMES = (
    "WhereIsAI/UAE-Large-V1",
    "BAAI/bge-large-en-v1.5",
    "Alibaba-NLP/gte-large-en-v1.5",
    "answerdotai/ModernBERT-large",
    "intfloat/multilingual-e5-large-instruct",
    "Alibaba-NLP/gte-Qwen2-7B-instruct",
)

STAT_COLNAMES = (
    "tokenizer",
    "perturbation",
    "mean_num_toks",  # Mean number of tokens per query
    "mean_tok_len",  # Mean number of characters per token
    "mean_char_dist",  # Mean characterwise edit distance (levenshtein) between original and perturbed query
    "mean_tok_dist",  # Mean tokenwise edit distance
    "mean_tok_jaccard",  # Mean Jaccard Index (IoU) between set of token ids from original and perturbed 
)


def tokenize(texts: List[str], tokenizer: PreTrainedTokenizer) -> List[List[int]]:
    return tokenizer(texts, add_special_tokens=False)["input_ids"]


def jaccard(set1: set, set2: set) -> float:
    return len(set1 & set2) / len(set1 | set2)


def calc_stats_for_perturbation(perturbation_df: pd.DataFrame, tokenizer: PreTrainedTokenizer) -> Tuple:
    queries = perturbation_df["query"].to_list()
    all_query_ids = tokenize(queries, tokenizer)

    perturbed = perturbation_df["perturbed_query"].to_list()
    all_perturbed_ids = tokenize(perturbed, tokenizer)
    all_perturbed_toks = [[tokenizer.decode(id_) for id_ in ids] for ids in all_perturbed_ids]
    all_tokenwise_distances = [
        Levenshtein.distance(q_ids, p_ids) for q_ids, p_ids in zip(all_query_ids, all_perturbed_ids)
    ]
    all_tokenwise_jaccards = [
        jaccard(set(q_ids), set(p_ids)) for q_ids, p_ids in zip(all_query_ids, all_perturbed_ids)
    ]

    mean_num_toks = np.mean([len(ids) for ids in all_perturbed_ids])
    mean_tok_len = np.mean([len(tok) for toks in all_perturbed_toks for tok in toks])
    mean_tokenwise_dist = np.mean(all_tokenwise_distances)
    mean_charwise_dist = np.mean([Levenshtein.distance(q, p) for q, p in zip(queries, perturbed)])
    mean_tokenwise_jaccard = np.mean(all_tokenwise_jaccards)

    return (mean_num_toks, mean_tok_len, mean_tokenwise_dist, mean_charwise_dist, mean_tokenwise_jaccard)


def main(perturbed_beir_fpath: str):

    perturbed_beir_df = pd.read_json(perturbed_beir_fpath)

    unmodified_query_df = (
        perturbed_beir_df[perturbed_beir_df.method == "characters_capitalization"]
        .copy(deep=True)
        .assign(
            perturbed_query=lambda df: df["query"],
            method="_none",
            metadata=str({"perturbation_strength": 0.0}),
        )
    )

    input_df = pd.concat([unmodified_query_df, perturbed_beir_df])

    print("Calculating stats...")
    raw_stats_data = []
    for toknizer_name in tqdm(TOKENIZER_NAMES, "Tokenizers", leave=True):
        tokenizer = AutoTokenizer.from_pretrained(toknizer_name)
        for pert_name, pert_df in tqdm(input_df.groupby("method"), "Perturbations", leave=False):
            curr_stats = calc_stats_for_perturbation(pert_df, tokenizer)
            raw_stats_data.append((toknizer_name, pert_name, *curr_stats))
    print("Done.")

    raw_stats_df = pd.DataFrame(raw_stats_data, columns=STAT_COLNAMES)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_stats_fpath = os.path.join(RESULTS_DIR, "raw_tokenization_stats.csv")
    raw_stats_df.to_csv(raw_stats_fpath, index=False)
    print(f"Raw tokenization statistics saved to: {raw_stats_fpath}")

    # Aggregating stats per perturbation level
    aggr_stats_data = []
    for level in ("_none", "character", "word"):
        curr_aggr_stats = raw_stats_df[raw_stats_df.perturbation.str.contains(level)].iloc[:, 2:].mean()
        aggr_stats_data.append(curr_aggr_stats)

    aggr_stats_df = pd.DataFrame(aggr_stats_data)
    aggr_stats_df.insert(0, "level", ["unmodified", "character_level", "word_level"])

    aggr_stats_fpath = os.path.join(RESULTS_DIR, "aggr_tokenization_stats.csv")
    aggr_stats_df.to_csv(aggr_stats_fpath, index=False)
    print(f"Aggregated tokenization statistics saved to: {aggr_stats_fpath}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-p", "--perturbed_beir_fpath", default=PERTURBED_BEIR_FPATH, help="Path to perturbed BEIR dataset"
    )
    args = argparser.parse_args()
    main(**args.__dict__)
