import os
import random
import numpy as np
import pandas as pd
from termcolor import colored

from src.constants import (
    BENCHMARK_DATASETS,
    BENCHMARK_DATASETS_NAMES,
    RANDOM_STATE,
    RAW_BEIR_DIR,
    SAMPLED_BEIR_FPATH
)
from src.utils import load_queries, load_qrels

DESIRED_TOTAL = 6000
INITIAL_SAMPLES = 300

random.seed(RANDOM_STATE)


def create_benchmark_distribution(datasets: dict[str, int]) -> dict[str, int]:
    initial_allocation = {key: min(value, INITIAL_SAMPLES) for key, value in datasets.items()}
    remaining_total = DESIRED_TOTAL - sum(initial_allocation.values())
    remaining_datasets = {
        key: value - INITIAL_SAMPLES for key, value in datasets.items() if value > INITIAL_SAMPLES
    }

    scaling_factor = remaining_total / sum(remaining_datasets.values())
    scaled_datasets = {key: value * scaling_factor for key, value in remaining_datasets.items()}
    scaled_datasets_rounded = {key: int(value) for key, value in scaled_datasets.items()}
    final_allocation = {
        key: initial_allocation.get(key, 0) + scaled_datasets_rounded.get(key, 0)
        for key, _ in datasets.items()
    }

    # Sometimes the final allocation might not be equal to DESIRED_TOTAL because of the rounding errors,
    # this is why it is fixed by remainders, I am calculating which datasets were affected the most while rounding
    # and then adjusting them accordingly.

    if sum(final_allocation.values()) != DESIRED_TOTAL:
        remainders = {key: value - scaled_datasets_rounded[key] for key, value in scaled_datasets.items()}
        difference = DESIRED_TOTAL - sum(final_allocation.values())
        sorted_remainders = sorted(remainders.items(), key=lambda item: -item[1])

        for i in range(difference):
            key = sorted_remainders[i][0]
            final_allocation[key] += 1

    original_distribution = {
        key: np.round(100 * value / sum(datasets.values()), 3) for key, value in datasets.items()
    }
    sampled_distribution = {
        key: np.round(100 * value / sum(final_allocation.values()), 3)
        for key, value in final_allocation.items()
    }

    print(f"{'Dataset':<20}{'Original %':<15}{'Sampled %':<15}{'Indicator':<10} ")
    for dataset, original_percentage in original_distribution.items():
        sampled_percentage = sampled_distribution.get(dataset, None)

        if sampled_percentage > original_percentage:
            indicator = colored("🡑 Higher", "green")
        elif sampled_percentage < original_percentage:
            indicator = colored("🡓 Lower", "red")
        else:
            indicator = "= Same"
        print(f"{dataset:<20}{original_percentage:<15}{sampled_percentage:<15}{indicator:<10}")

    print(
        f"{'SUM':<20}{sum(original_distribution.values()):<15.3f}{sum(sampled_distribution.values()):<15.3f}"
    )
    return final_allocation


def load_split_queries(
    queries_file_path: str | os.PathLike, qrels_file_path: str | os.PathLike
) -> dict[str, str]:
    all_queries = load_queries(queries_file_path)
    qrels = load_qrels(qrels_file_path)
    split_queries = {qid: all_queries[qid] for qid in qrels}
    return split_queries


def _prepare_dataset_df(dataset_path: str | os.PathLike, dataset_name: str) -> pd.DataFrame:
    split = "dev" if dataset_name == "msmarco" else "test"
    queries_file_path = os.path.join(dataset_path, "queries.jsonl")
    qrels_file_path = os.path.join(dataset_path, "qrels", f"{split}.tsv")
    split_queries = load_split_queries(queries_file_path, qrels_file_path)
    dataset_df = pd.DataFrame(
        {
            "query_id": list(split_queries.keys()),
            "query": list(split_queries.values()),
            "dataset": [f"{dataset_name}"] * len(split_queries),
        }
    )
    return dataset_df


def create_benchmark(
    datasets_queries: dict[str, pd.DataFrame], benchmark_distribution: dict[str, int]
) -> pd.DataFrame:
    sampled_dfs = []
    for dataset in benchmark_distribution.keys():
        population_df = datasets_queries[dataset]
        n_to_sample = benchmark_distribution[dataset]
        sample_df = population_df.sample(n_to_sample, random_state=RANDOM_STATE)
        sampled_dfs.append(sample_df)

    benchmark = pd.concat(sampled_dfs)
    return benchmark


if __name__ == "__main__":
    benchmark_distribution = create_benchmark_distribution(BENCHMARK_DATASETS)
    datasets_queries = {}

    for dataset_name in BENCHMARK_DATASETS_NAMES:
        dataset_path = os.path.join(RAW_BEIR_DIR, dataset_name)

        # Unfortunately, CQADupstack dataset is divided into domains and does not have the same structure as other
        # collections, authors of BEIR evaluate it by calculating the mean performance across all domains
        # (see https://github.com/beir-cellar/beir/issues/9). This is why we have to handle CQADupstack in a slightly different way.
        if dataset_name == "cqadupstack":
            dir_content = os.listdir(dataset_path)
            categories = [entry for entry in dir_content if os.path.isdir(os.path.join(dataset_path, entry))]

            categories_dfs = []
            for c in categories:
                category_path = os.path.join(dataset_path, c)
                category_df = _prepare_dataset_df(category_path, f"{dataset_name}_{c}")
                categories_dfs.append(category_df)

            dataset_df = pd.concat(categories_dfs)

        else:
            dataset_df = _prepare_dataset_df(dataset_path, dataset_name)

        datasets_queries[dataset_name] = dataset_df

    benchmark = create_benchmark(datasets_queries, benchmark_distribution)
    benchmark.to_json(SAMPLED_BEIR_FPATH, orient="records", indent=4)
