# This script loads multiple models and evaluates them on all perturbation files from specific directory
import os
import mlflow
import argparse
from omegaconf import OmegaConf
import pandas as pd

from src.evaluate_benchmark import main, _create_mlflow_tables
from src.evaluate_beir_dataset import load_model_and_retriever

USE_CACHE = True

if __name__ == "__main__":
    mlflow.set_experiment("benchmark_model_evaluation")
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_dir", required=True, type=str, help="Directory where yaml files with model configs are stored")
    parser.add_argument("--benchmark_path", required=True, default="metrics",  type=str, help="Path to the json file with all the perturbations (created using create_perturbed_data.py)")
    args = parser.parse_args()
    
    benchmark = pd.read_json(args.benchmark_path)
    
    for config_name in os.listdir(args.configs_dir):
        config_path = os.path.join(args.configs_dir, config_name)
        config = OmegaConf.load(config_path)
        config["use_cache"] = USE_CACHE
        config["benchmark_path"] = args.benchmark_path
        for key in ("passage_prefix", "instruct_prefix", "query_prefix"):
            config[key] = config.get(key, None)
            
        model, retriever = load_model_and_retriever(config)
        for method in benchmark["method"].unique():
            method_df = benchmark[benchmark["method"] == method]
            with mlflow.start_run():
                results = main(method_df, model, retriever, config)
                mlflow_tables = _create_mlflow_tables(results)
                mlflow.log_params(config)
                for table_name, table in mlflow_tables.items():
                    mlflow.log_table(table, f"{table_name}.json")