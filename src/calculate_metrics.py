import os
from pathlib import Path
import argparse
import pandas as pd
from src.utils import calculate_metrics


def main(args):
    benchmark = pd.read_json(args.perturbed_benchmark)
    queries = benchmark["query"].tolist()
    perturbed_queries = benchmark["perturbed_query"].tolist()

    metrics = calculate_metrics(queries, perturbed_queries, args.bertscore_batch_size)

    metrics_df = pd.DataFrame(metrics)
    benchmark_name = os.path.splitext(os.path.split(args.perturbed_benchmark)[-1])[0]
    save_dir = Path(args.save_dir) / benchmark_name
    os.makedirs(save_dir, exist_ok=True)
    metrics_df.to_csv(save_dir / "metrics.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--perturbed_benchmark", required=True, type=str)
    parser.add_argument("--save_dir", required=False, default="metrics",  type=str)
    parser.add_argument("--bertscore_batch_size", required=False, default=128, type=int)
    args = parser.parse_args()
    main(args)