#!/usr/bin/env python
import os
from beir import util

from src.constants import BENCHMARK_DATASETS_NAMES, RAW_BEIR_DIR

def download(dataset_name):
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset_name)
    print(url)
    _ = util.download_and_unzip(url, RAW_BEIR_DIR)


if __name__ == "__main__":
    os.makedirs(RAW_BEIR_DIR, exist_ok=True)
    for dataset in BENCHMARK_DATASETS_NAMES:
        ds_dir = os.path.join(RAW_BEIR_DIR, dataset)
        if os.path.exists(ds_dir):
            print(f"Download is skipped for {dataset} (data already exist in {ds_dir})")
            continue
        try:
            download(dataset)
        except Exception as e:
            print(f"Following error occured while downloading {dataset}: {e}")
