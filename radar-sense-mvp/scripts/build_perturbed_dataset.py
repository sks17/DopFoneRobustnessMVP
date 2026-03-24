"""Build the perturbed synthetic dataset for RADAR-Sense-MVP."""

from __future__ import annotations

from benchmark.dataset_builder import (
    build_clean_dataset,
    build_default_perturbation_parameters,
    build_perturbed_dataset,
)
from scripts.build_clean_dataset import serialize_dataset
from utils.io import read_yaml_file, write_json_file


# Spec:
# - General description: Build a default perturbed dataset and write it to disk.
# - Params: None.
# - Pre: `configs/generation.yaml` exists and is valid.
# - Post: Writes `data/manifests/perturbed_manifest.json`.
# - Mathematical definition: Not applicable; this is an orchestration entry point.
def main() -> None:
    """Run the perturbed dataset build entry point."""
    generation_config = read_yaml_file("configs/generation.yaml")
    clean_dataset = build_clean_dataset(generation_config)
    perturbation_parameters = build_default_perturbation_parameters(
        perturbation_name="gaussian_noise",
        seed=int(generation_config["seed"]),
    )
    perturbed_dataset = build_perturbed_dataset(
        clean_dataset=clean_dataset,
        perturbation_name="gaussian_noise",
        parameters=perturbation_parameters,
    )
    write_json_file(
        "data/manifests/perturbed_manifest.json",
        serialize_dataset(perturbed_dataset),
    )


if __name__ == "__main__":
    main()
