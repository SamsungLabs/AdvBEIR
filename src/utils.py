import os
from pathlib import Path
from typing import Union
import csv
import json
import numpy as np
from Levenshtein import distance
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from typing import List, Dict
from torchmetrics.text import BERTScore
from tqdm import tqdm
import hashlib

import pandas as pd


def get_repo_root() -> Path:
    """
    :return Path: Root project path
    """
    return Path(os.path.abspath(__file__)).parent.parent


def load_qrels(qrels_file_path):
    qrels = {}
    reader = csv.reader(open(qrels_file_path, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    next(reader)

    for _, row in enumerate(reader):
        query_id, corpus_id, score = row[0], row[1], int(row[2])

        if query_id not in qrels:
            qrels[query_id] = {corpus_id: score}
        else:
            qrels[query_id][corpus_id] = score
    return qrels


def load_queries(queries_file_path):
    queries = {}
    with open(queries_file_path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            id = line.get("_id")
            text = line.get("text")
            queries[id] = text
    return queries


def save_df_to_json_or_jsonl(df, file_path: Union[str, Path], ext: str) -> None:
    os.makedirs(os.path.split(file_path)[0], exist_ok=True)
    if ext == '.jsonl':
        df.to_json(file_path, orient='records', lines=True)
    else:  # .json
        df.to_json(file_path, orient='records')


def read_json_or_jsonl(file_path: Union[str, Path], ext: str) -> pd.DataFrame:
    if ext == '.jsonl':
        return pd.read_json(file_path, orient='records', lines=True)
    # .json
    return pd.read_json(file_path, orient='records')


def calculate_metrics(original_queries: List[str], perturbed_queries: List[str], bertscore_batch_size) -> Dict[str, List[float]]:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    bertscorer = BERTScore(model_name_or_path="roberta-base", truncation=True)
    levenshtein = [distance(original, perturbed) for original, perturbed in tqdm(zip(original_queries, perturbed_queries), desc="Levenshtein")]
    rougeL = [scorer.score(original, perturbed)['rougeL'].fmeasure for original, perturbed in tqdm(zip(original_queries, perturbed_queries), desc="Rouge")]
    
    splitted_original_queries = [query.split() for query in original_queries]
    splitted_perturbed_queries = [query.split() for query in perturbed_queries]
    bleu = [sentence_bleu([original], perturbed, weights=(1,)) for original, perturbed in tqdm(zip(splitted_original_queries, splitted_perturbed_queries), desc="Bleu")]

    bertscore = [
        bertscorer(preds=perturbed_queries[batch_index:batch_index+bertscore_batch_size],
                   target=original_queries[batch_index:batch_index+bertscore_batch_size])['f1'] 
                   for batch_index
                   in tqdm(range(0, len(perturbed_queries), bertscore_batch_size), desc="Bert score")
    ]
    bertscore = np.concatenate(bertscore).tolist()
    metrics = {
        "levenshtein": levenshtein,
        "rouge": rougeL,
        "bleu": bleu,
        "bertscore": bertscore
    }
    return metrics


def make_hash(item: str) -> str:
    return hashlib.sha256(item.encode("utf8")).hexdigest()
