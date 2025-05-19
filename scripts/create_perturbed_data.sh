#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(cd ..; pwd)"
export CUDA_VISIBLE_DEVICES="3,4"

cd ../src
python ./create_perturbed_data.py --config ../configs/final_data_perturbations.yaml
