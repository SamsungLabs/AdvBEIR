from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader

import os
import logging
import pandas as pd
import numpy as np
import mlflow
from argparse import ArgumentParser
from omegaconf import OmegaConf
from typing import Any

from src.constants import RAW_BEIR_DIR, INDICES_DIR, EVALUATION_K_VALUES
from src.perturbations import AVAILABLE_PERTURBATIONS
from src.models import ModelWrapper
from src.evaluators import EvaluateRetrievalV2
from src.evaluate_beir_dataset import evaluate, _init_index, load_model_and_retriever


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


def _aggregate_dupstack_metrics(metrics_data):
    """Calculates average metrics for the cqadupstack dataset"""
    aggregated_metrics = {}
    for dataset_name in metrics_data:
        if "dupstack" in dataset_name:
            metrics = metrics_data[dataset_name]
            for metric_name in metrics:
                aggregated_metrics[metric_name] = aggregated_metrics.get(metric_name, []) + [
                    metrics[metric_name]
                ]
        else:
            continue

    aggregated_metrics = {
        metric_name: np.mean(aggregated_metrics[metric_name]) for metric_name in aggregated_metrics.keys()
    }
    return aggregated_metrics


def _prepare_metrics_dfs(metrics_dict):
    """Prepares a specific table with metrics for mlflow"""
    metric_dfs = {}

    for metric_name in next(iter(metrics_dict.values())).keys():
        metric_data = {dataset: metrics[metric_name] for dataset, metrics in metrics_dict.items()}
        if any("dupstack" in key for key in metric_data.keys()):
            metric_data["cqadupstack_aggregated"] = _aggregate_dupstack_metrics(metric_data)
        metric_df = pd.DataFrame(metric_data).T
        metric_df.reset_index(names="dataset", inplace=True)
        metric_dfs[metric_name] = metric_df

    return metric_dfs


def _create_mlflow_tables(metrics_dict):
    """Creates tables with metrics for mlflow"""
    original_query_dict = {
        dataset_name: metrics_dict[dataset_name]["query"] for dataset_name in metrics_dict.keys()
    }
    perturbed_query_dict = {
        dataset_name: metrics_dict[dataset_name]["perturbed_query"] for dataset_name in metrics_dict.keys()
    }
    original_query_metrics = _prepare_metrics_dfs(original_query_dict)
    perturbed_query_metrics = _prepare_metrics_dfs(perturbed_query_dict)

    original_query_mlflow_tables = {
        f"{metric_name}_original_query": table for metric_name, table in original_query_metrics.items()
    }
    perturbed_query_mlflow_tables = {
        f"{metric_name}_perturbed_query": table for metric_name, table in perturbed_query_metrics.items()
    }
    mlflow_tables = {**original_query_mlflow_tables, **perturbed_query_mlflow_tables}

    return mlflow_tables


def main(benchmark: pd.DataFrame, model: ModelWrapper, retriever: EvaluateRetrievalV2, config: dict[str, Any]):
    mlflow_dataset = mlflow.data.from_pandas(benchmark)
    mlflow.log_input(mlflow_dataset, context="Benchmark dataset")
    config["perturbation_method"] = benchmark.iloc[0]["method"]
    config["perturbation_strength"] = benchmark.iloc[0]["metadata"]["perturbation_strength"]
    datasets = benchmark["dataset"].unique()

    datasets_paths = [
        (
            os.path.join(RAW_BEIR_DIR, dataset)
            if "cqadupstack" not in dataset
            else os.path.join(RAW_BEIR_DIR, "/".join(dataset.split("_")))
        )
        for dataset in datasets
    ]

    metrics_dict = {}
    for dataset_name, path in zip(datasets, datasets_paths):
        config["dataset_name"] = dataset_name
        metrics_dict[dataset_name] = {}
        split = "dev" if "msmarco" in path else "test"
        model.set_dataset_specific_params(dataset_name)
    
        if config["use_prompt"] == True:
            logging.info(f"Prompt: {model.retriever.model.prompt}")
        if config["use_cache"] == True:
            logging.info(f"Cache directory with indices: {INDICES_DIR}")

        corpus, _, qrels = GenericDataLoader(data_folder=path).load(split=split)
        # create index only once per dataset and reuse it for both queries and perturbed queries
        _init_index(config, corpus, model)
        eval_data = benchmark[benchmark["dataset"] == dataset_name]
        logging.info(f"Evaluating {dataset_name} dataset")

        for column in ["query", "perturbed_query"]:
            eval_queries = {
                query_id: query for query_id, query in zip(eval_data["query_id"], eval_data[column])
            }
            
            # if we are evaluating multible methods, original queries are always the same and we can keep the results in cache
            if column == "query":
                if dataset_name in model.original_query_results_cache.keys():
                    logging.info(f"Found results for original queries of {dataset_name} in cache, moving to the evaluation of perturbed queries.")
                    metrics_dict[dataset_name][column] = model.original_query_results_cache[dataset_name]           
                else:
                    metrics = evaluate(retriever, eval_queries, qrels, EVALUATION_K_VALUES)
                    logging.info(f"Saving metrics for original queries for {dataset_name} in cache, it will speedup the evaluation process if you are testing multiple methods.")
                    model.original_query_results_cache[dataset_name] = metrics
                    metrics_dict[dataset_name][column] = metrics
            else:
                metrics = evaluate(retriever, eval_queries, qrels, EVALUATION_K_VALUES)
                metrics_dict[dataset_name][column] = metrics

    return metrics_dict


if __name__ == "__main__":
    mlflow.set_experiment("benchmark_evaluation")

    parser = ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        default="configs/benchmark_evaluation_config.yaml",
        help="Path to the yaml file with evaluation parameters",
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    benchmark_df = pd.read_json(config["benchmark_path"])
    benchmark_df["query_id"] = benchmark_df["query_id"].astype(str)
    assert (
        benchmark_df.iloc[0]["method"] in AVAILABLE_PERTURBATIONS
    ), f"The perturbation method must be one of the following: {AVAILABLE_PERTURBATIONS}"
   
    for key in ("passage_prefix", "instruct_prefix", "query_prefix"):
                config[key] = config.get(key, None)
                
    model, retriever = load_model_and_retriever(config)
    for method in benchmark_df["method"].unique():
        method_df = benchmark_df[benchmark_df["method"] == method]
        with mlflow.start_run():
            results = main(method_df, model, retriever, config)
            mlflow_tables = _create_mlflow_tables(results)

            mlflow.log_params(config)
            for table_name, table in mlflow_tables.items():
                mlflow.log_table(table, f"{table_name}.json")
