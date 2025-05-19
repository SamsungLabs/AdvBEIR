# NOTE: before running this script, make sure that you have Elastic Search installed on your computer

import logging
import os
import mlflow
import pandas as pd

from argparse import ArgumentParser
from typing import Any
from beir import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25

from src.constants import RAW_BEIR_DIR, EVALUATION_K_VALUES
from src.evaluate_benchmark import _create_mlflow_tables

HOSTNAME = "localhost" 
NUMBER_OF_SHARDS = 1

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

def evaluate(
    retriever: EvaluateRetrieval,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    corpus: dict[str, dict[str, str]],
    k_values: list[int] = [1, 3, 5, 10, 50, 100, 1000],
) -> dict[str, float]:
    results = retriever.retrieve(corpus, queries)
    ndcg, map, recall, precision = retriever.evaluate(qrels, results, k_values)
    metrics = {"ndcg": ndcg, "map": map, "recall": recall, "precision": precision}
    return metrics

def main(benchmark: pd.DataFrame, config: dict[str, Any]): 
    metrics_dict = {}
    original_query_results_cache = {}
    
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

    # we are going to initialize index for each dataset separately in the loop
    model = BM25(
            index_name="blank",
            hostname=HOSTNAME,
            initialize=False,
            number_of_shards=NUMBER_OF_SHARDS
            )
    evaluator = EvaluateRetrieval(retriever=model)
    
    for dataset_name, path in zip(datasets, datasets_paths):
        print(dataset_name, path)
        # checking if index for specific dataset exists, if yes - there is no need to initialize it
        index_exists = model.es.es.indices.exists(dataset_name)
        if index_exists:
            logging.info(f"Index with {dataset_name} dataset has been already created in the past, loading from cache...")
            model.es.index_name = dataset_name
            model.index_name = dataset_name
            model.initialize = False
        else:
            model.es.index_name = dataset_name
            model.index_name = dataset_name
            model.initialize = True
            # creating index object, but observations are added later during the evaluation
            model.initialise()

        config["dataset_name"] = dataset_name
        metrics_dict[dataset_name] = {}
        split = "dev" if "msmarco" in path else "test"

        corpus, _, qrels = GenericDataLoader(data_folder=path).load(split=split)
        eval_data = benchmark[benchmark["dataset"] == dataset_name]
        logging.info(f"Evaluating {dataset_name} dataset")

        for column in ["query", "perturbed_query"]:
            eval_queries = {
                query_id: query for query_id, query in zip(eval_data["query_id"].tolist(), eval_data[column].tolist())
            }
            
            # if we are evaluating multible methods, original queries are always the same and we can keep the results in cache
            if column == "query":
                if dataset_name in original_query_results_cache.keys():
                    logging.info(f"Found results for original queries of {dataset_name} in cache, moving to the evaluation of perturbed queries.")
                    metrics_dict[dataset_name][column] = original_query_results_cache[dataset_name]           
                else:
                    metrics = evaluate(evaluator, eval_queries, qrels, corpus, EVALUATION_K_VALUES)
                    logging.info(f"Saving metrics for original queries for {dataset_name} in cache, it will speedup the evaluation process if you are testing multiple methods.")
                    original_query_results_cache[dataset_name] = metrics
                    metrics_dict[dataset_name][column] = metrics
            else:
                metrics = evaluate(evaluator, eval_queries, qrels, corpus, EVALUATION_K_VALUES)
                metrics_dict[dataset_name][column] = metrics
            
            # Setting initialize to False to not add corpus to index for the second time while retrieving for perturbed queries
            model.initialize = False

    return metrics_dict

if __name__ == "__main__":
    mlflow.set_experiment("bm25_eval")

    parser = ArgumentParser()
    parser.add_argument('-b', '--benchmark', type=str, help="Path to the jsonl file with benchmark") 
    args = parser.parse_args()

    benchmark_df = pd.read_json(args.benchmark, lines=True)
    config = {"benchmark_path": args.benchmark}
    print(f"Number of perturbation methods in the benchmark file: {benchmark_df['method'].nunique()}")    

    for method in benchmark_df["method"].unique():    
        method_df = benchmark_df[benchmark_df["method"] == method]
        with mlflow.start_run():
            results = main(method_df, config)
            mlflow_tables = _create_mlflow_tables(results)

            mlflow.log_params(config)
            for table_name, table in mlflow_tables.items():
                mlflow.log_table(table, f"{table_name}.json")