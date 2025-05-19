### Adding new perturbation method
1. The new method class should inherit from the `TextPerturbation` class (from `transformations/text_perturbation.py`). 
2. Then new class should be added to `PERTURBATION_CLASSES` in `transformations/__init__.py`. 
3. Finally parameters for the new method shoud be added to the config that you use with `src/create_perturbed_data.py` (e.g. `configs/final_data_perturbations.yaml`).
