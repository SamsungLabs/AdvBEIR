import argparse
import os

import pandas as pd
from omegaconf import OmegaConf
from src.constants import (BENCHMARK_DATASETS_NAMES, PERTURBED_BEIR_DIR, PERTURBED_BEIR_FPATH,
                           SAMPLED_BEIR_FPATH)
from src.perturbations import PERTURBATION_NAME_TO_CLS
from src.perturbations.sentence_level.paraphrase import Paraphraser


def main(args):
    config = OmegaConf.load(args.config_file)

    sampled_beir_df = pd.read_json(SAMPLED_BEIR_FPATH)

    processed_datasets = config.get("datasets_subset")
    if processed_datasets is not None:
        sampled_beir_df = sampled_beir_df[sampled_beir_df["dataset"].isin(processed_datasets)]
    else:
        processed_datasets = BENCHMARK_DATASETS_NAMES  # process all datasets

    queries = sampled_beir_df["query"].tolist()
    ds_names = sampled_beir_df["dataset"].tolist()
    methods = config.perturbation_methods
    perturbed_dfs = []

    print(f"Applying {len(methods)} perturbations on {len(processed_datasets)} datasets:")

    for i, method in enumerate(methods):
        print(f"[{i+1}/{len(methods)}] {method}...")

        perturbation_cls = PERTURBATION_NAME_TO_CLS[method]
        perturbation = perturbation_cls(config[method])

        # Workaround due to one additional argument in Paraphraser.__call__ function
        kwargs = {"dataset_names": ds_names} if isinstance(perturbation, Paraphraser) else {}

        perturbed_queries = perturbation(queries, **kwargs)

        perturbed_df = sampled_beir_df.copy(deep=True)
        metadata = str(perturbation.get_metadata())
        perturbed_df["perturbed_query"] = perturbed_queries
        perturbed_df["method"] = method
        perturbed_df["metadata"] = metadata

        perturbed_dfs.append(perturbed_df)

    all_perturbed_df = pd.concat(perturbed_dfs).reset_index(drop=True)

    # Saving perturbed queries
    for ds_name in processed_datasets:
        ds_df = all_perturbed_df[all_perturbed_df["dataset"] == ds_name]

        out_dir = os.path.join(PERTURBED_BEIR_DIR, ds_name)
        os.makedirs(out_dir, exist_ok=True)

        out_fpath = os.path.join(out_dir, f"perturbed_{ds_name}.json")
        ds_df.to_json(out_fpath, force_ascii=False, orient="records", index=False)
    print(f"Pertubed queries for each dataset saved to: {PERTURBED_BEIR_DIR}")

    # Aggregating perturbed queries from all datasets
    all_ds_dfs = []
    for ds_name in BENCHMARK_DATASETS_NAMES:
        ds_fpath = os.path.join(PERTURBED_BEIR_DIR, ds_name, f"perturbed_{ds_name}.json")
        if os.path.isfile(ds_fpath):
            ds_df = pd.read_json(ds_fpath)
            all_ds_dfs.append(ds_df)
        else:
            print(f"No perturbed queries from {ds_name} - missing file: {ds_fpath}")

    # Merging all perturbed datasets to single file
    final_benchmark_df = pd.concat(all_ds_dfs).reset_index(drop=True)
    print(f"Merged data from all {len(all_ds_dfs)} datasets")

    # Saving final benchmark file
    final_benchmark_fpath = config.get("out_fpath")
    if final_benchmark_fpath is None:
        final_benchmark_fpath = PERTURBED_BEIR_FPATH
    final_benchmark_df.to_json(final_benchmark_fpath, force_ascii=False, orient="records", index=False)
    print(f"Saved benchmark to a single file: {final_benchmark_fpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
