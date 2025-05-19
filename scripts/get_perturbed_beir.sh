#!/bin/bash

cd ..
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# export CUDA_VISIBLE_DEVICES="3,4"

num_steps=4
curr_step=1

# Create venv
echo "[$((curr_step++))/$num_steps] Creating a virtual environment...";
python -m venv .venv
source .venv/bin/activate
pip install -r ./requirements.txt

# Download raw beir
echo "[$((curr_step++))/$num_steps] Downloading all beir datasets...";
cd ./src
python ./download_all_beir_datasets.py

# Sample from beir
echo "[$((curr_step++))/$num_steps] Sampling queries from beir...";
python ./sample_benchmark_data.py

# Perturb queries
echo "[$((curr_step++))/$num_steps] Recreating perturbed queries that are missing from the repo..."
python ./create_perturbed_data.py --config ../configs/final_data_perturbations.yaml
