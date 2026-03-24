"""Build the clean synthetic dataset for RADAR-Sense-MVP."""

from __future__ import annotations

from benchmark.dataset_builder import build_clean_dataset
from utils.io import read_yaml_file, write_json_file


# Spec:
# - General description: Convert a signal-example dataset into JSON-serializable dictionaries.
# - Params: `dataset`, list of signal examples.
# - Pre: Each example in `dataset` is valid.
# - Post: Returns a list of dictionaries ready for JSON serialization.
# - Mathematical definition: Applies a field-preserving map from datatype values to JSON-compatible records.
def serialize_dataset(dataset: list[object]) -> list[dict[str, object]]:
    """Return a JSON-serializable dataset representation."""
    serialized_dataset: list[dict[str, object]] = []
    for example in dataset:
        serialized_dataset.append(
            {
                "example_id": example.example_id,
                "signal": example.signal.tolist(),
                "sample_rate_hz": example.sample_rate_hz,
                "heart_rate_bpm": example.heart_rate_bpm,
                "is_perturbed": example.is_perturbed,
                "perturbation_name": example.perturbation_name,
            }
        )
    return serialized_dataset


# Spec:
# - General description: Build the clean dataset from default config paths and write it to disk.
# - Params: None.
# - Pre: `configs/generation.yaml` exists and is valid.
# - Post: Writes `data/manifests/clean_manifest.json`.
# - Mathematical definition: Not applicable; this is an orchestration entry point.
def main() -> None:
    """Run the clean dataset build entry point."""
    generation_config = read_yaml_file("configs/generation.yaml")
    clean_dataset = build_clean_dataset(generation_config)
    write_json_file("data/manifests/clean_manifest.json", serialize_dataset(clean_dataset))


if __name__ == "__main__":
    main()
