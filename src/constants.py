import os
import json 

from src.utils import get_repo_root


# for reproducibility
RANDOM_STATE = 42

# directories
ROOT_DIR = get_repo_root()
CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
RAW_BEIR_DIR = os.path.join(DATA_DIR, "raw_beir_datasets")
PERTURBED_BEIR_DIR = os.path.join(DATA_DIR, "perturbed_beir_datasets")
INDICES_DIR = os.environ.get("INDICES_CACHE_DIR", os.path.join(ROOT_DIR, ".cache", "indices"))

# file paths
SAMPLED_BEIR_FPATH = os.path.join(DATA_DIR, "sampled_beir.json")
PERTURBED_BEIR_FPATH = os.path.join(DATA_DIR, "perturbed_beir.json")

# BEIR datasets used by us (public ones)
BENCHMARK_DATASETS = {
    "msmarco": 6980,
    "trec-covid": 50,
    "nfcorpus": 323,
    "nq": 3452,
    "hotpotqa": 7405,
    "fiqa": 648,
    "arguana": 1406,
    "webis-touche2020": 49,
    "cqadupstack": 13145,
    "quora": 10000,
    "dbpedia-entity": 400,
    "scidocs": 1000,
    "fever": 6666,
    "climate-fever": 1535,
    "scifact": 300,
}
BENCHMARK_DATASETS_NAMES = list(BENCHMARK_DATASETS.keys())

# evaluation parameters
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 1024
EVALUATION_K_VALUES = [1, 3, 5, 10, 50, 100, 1000]

# taken from https://huggingface.co/nvidia/NV-Embed-v2/blob/main/instructions.json
INSTRUCTIONS_PATH = os.path.join(ROOT_DIR, "data", "instructions.json")
PROMPTS = json.load(open(INSTRUCTIONS_PATH))
