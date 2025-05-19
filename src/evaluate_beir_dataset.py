from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader

import logging
import os
import numpy as np
import pandas as pd
import mlflow
from faiss import read_index
from argparse import ArgumentParser
from omegaconf import OmegaConf
from typing import Any

from src.models import DefaultModel, ModelWrapper
from src.retrievers import DRESV2
from src.evaluators import EvaluateRetrievalV2
from src.constants import RAW_BEIR_DIR, BENCHMARK_DATASETS_NAMES, INDICES_DIR, EVALUATION_K_VALUES
from src.utils import make_hash

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

def load_model_and_retriever(config):
    model = DRESV2(
        DefaultModel(
            model_path=config["model_path"],
            query_prefix=config["query_prefix"],
            passage_prefix=config["passage_prefix"],
            instruct_prefix=config["instruct_prefix"],
            use_prompt=config["use_prompt"]
        ),
        batch_size=config["batch_size"],
        use_cache=config["use_cache"],
        convert_to_tensor = False,
        normalize_embeddings=True
    )

    model = ModelWrapper(model)
    retriever = EvaluateRetrievalV2(model)
    return model, retriever

def _init_index(config: dict[str, Any], corpus: dict[str, dict[str, str]], model: ModelWrapper):
    hash = hash_index(config["model_path"], pd.DataFrame(corpus).T)
    
    if config["use_cache"] == False:
        logging.info("Cache usage is turned off, set use_cache argument in config file to True in order to utilize it.")
        model.create_index(corpus, hash)
    elif validate_index(config, hash):
        load_index(model, hash, corpus)
    else:
        model.create_index(corpus, hash)

def validate_index(config: dict[str, Any], hash: str):
    indices = os.listdir(INDICES_DIR)
    if any(index.startswith(hash) for index in indices):
        logging.info(
            f"Index calculated for {config['dataset_name']} dataset and {config['model_path']} model is already in cache, loading..."
        )
        return True
    else:
        logging.info(
            f"Didn't find index for {config['dataset_name']} dataset and {config['model_path']} model in cache, creating a new one..."
        )
        return False


def hash_index(model: str, data: pd.DataFrame):
    os.makedirs(INDICES_DIR, exist_ok=True)
    config_hash = make_hash(model)
    row_hashes = pd.util.hash_pandas_object(data)
    data_hash = hash(tuple(row_hashes))
    retriever_tuple = (config_hash, data_hash)
    retriever_hash = make_hash(str(retriever_tuple))
    return retriever_hash


def load_index(model: ModelWrapper, hash: str, corpus: dict[str, dict[str, str]]) -> None:
    model.retriever.index = read_index(os.path.join(INDICES_DIR, hash + ".index"))
    corpus_ids = sorted(
        corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True
    )
    texts = [corpus[cid]["text"] for cid in corpus_ids]
    titles = [corpus[cid]["title"] for cid in corpus_ids]
    model.retriever.index_df = pd.DataFrame({"id": corpus_ids, "text": texts, "title": titles})

def evaluate(
    retriever: EvaluateRetrievalV2,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    k_values: list[int] = [1, 3, 5, 10, 50, 100, 1000],
) -> dict[str, float]:
    results = retriever.retrieve(queries)
    ndcg, map, recall, precision = retriever.evaluate(qrels, results, k_values)
    metrics = {"ndcg": ndcg, "map": map, "recall": recall, "precision": precision}
    return metrics


def _evaluate_dupstack(model: DRESV2, retriever: EvaluateRetrievalV2, config: dict[str, Any]):
    """Performs evaluation on each domain of cqadupstack and calculates average metrics"""

    metrics = {"ndcg": {}, "map": {}, "recall": {}, "precision": {}}
    dir_content = os.listdir(config["data_path"])
    categories_paths = [
        os.path.join(config["data_path"], entry)
        for entry in dir_content
        if os.path.isdir(os.path.join(config["data_path"], entry))
    ]
    for path in categories_paths:
        logging.info(f"Evaluating cqadupstack category: {path.split('/')[-1]}")
        corpus, queries, qrels = GenericDataLoader(data_folder=path).load(split=config["split"])
        _init_index(config, corpus, model)

        category_metrics = evaluate(retriever, queries, qrels, EVALUATION_K_VALUES)
        for general_metric, general_metric_values in category_metrics.items():
            for specific_metric in general_metric_values.keys():
                if specific_metric in metrics[general_metric]:
                    metrics[general_metric][specific_metric].append(general_metric_values[specific_metric])
                else:
                    metrics[general_metric][specific_metric] = [general_metric_values[specific_metric]]

    aggregated_metrics = {
        general_metric: {
            specific_metric: np.round(np.mean(values), 5)
            for specific_metric, values in specific_metrics.items()
        }
        for general_metric, specific_metrics in metrics.items()
    }

    return aggregated_metrics


def main(model: ModelWrapper, retriever: EvaluateRetrievalV2, config: dict[str, Any]):
    mlflow.set_experiment("beir_dataset_evaluation")
    model.set_dataset_specific_params(config["dataset_name"])
    
    if config["use_prompt"] == True:
        logging.info(f"Prompt: {model.retriever.model.prompt}")
    if config["dataset_name"] == "cqadupstack":
        metrics = _evaluate_dupstack(model, retriever, config)
    else:
        corpus, queries, qrels = GenericDataLoader(data_folder=config["data_path"]).load(
            split=config["split"]
        )
        _init_index(config, corpus, model)
        metrics = evaluate(retriever, queries, qrels, EVALUATION_K_VALUES)

    return metrics


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        default="configs/beir_evaluation_config.yaml",
        help="Path to the yaml file with evaluation parameters",
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    for key in ("passage_prefix", "instruct_prefix", "query_prefix"):
        config[key] = config.get(key, None)

    model, retriever = load_model_and_retriever(config)

    assert (
        config["dataset_name"] in BENCHMARK_DATASETS_NAMES
    ), f"The dataset name must be one of the following: {BENCHMARK_DATASETS_NAMES}"

    config["data_path"] = os.path.join(RAW_BEIR_DIR, config["dataset_name"])
    config["split"] = "dev" if config["dataset_name"] == "msmarco" else "test"
    results = main(model, retriever, config)
    # mlflow does not accept '@' in metric names
    results = {
        metric_key.replace("@", "_at_"): value
        for key in results.keys()
        for metric_key, value in results[key].items()
    }

    experiment_params = {
        "model_path": config["model_path"],
        "data_path": config["data_path"],
        "split": config["split"],
    }

    mlflow.log_params(experiment_params)
    mlflow.set_tags({"dataset": config["dataset_name"], "model": config["model_path"].split("/")[-1]})
    mlflow.log_metrics(results)
