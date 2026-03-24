"""Run the RADAR-Sense-MVP benchmark."""

from __future__ import annotations

from benchmark.dataset_builder import (
    build_clean_dataset,
    build_default_perturbation_parameters,
    build_perturbed_dataset,
)
from benchmark.runner import run_example_level_benchmark, summarize_benchmark_results
from utils.io import read_yaml_file, write_json_file


# Spec:
# - General description: Convert benchmark results to JSON-serializable dictionaries.
# - Params: `results`, list of benchmark result values.
# - Pre: Each element in `results` is valid.
# - Post: Returns a list of dictionaries ready for JSON serialization.
# - Mathematical definition: Applies a field-preserving map from datatype values to JSON-compatible records.
def serialize_results(results: list[object]) -> list[dict[str, object]]:
    """Return a JSON-serializable benchmark-result representation."""
    serialized_results: list[dict[str, object]] = []
    for result in results:
        serialized_results.append(
            {
                "example_id": result.example_id,
                "true_heart_rate_bpm": result.true_heart_rate_bpm,
                "estimated_heart_rate_bpm": result.estimated_heart_rate_bpm,
                "absolute_error_bpm": result.absolute_error_bpm,
                "within_tolerance": result.within_tolerance,
            }
        )
    return serialized_results


# Spec:
# - General description: Run the default benchmark pipeline and write result artifacts.
# - Params: None.
# - Pre: Config files exist and are valid.
# - Post: Writes `data/manifests/benchmark_results.json`.
# - Mathematical definition: Not applicable; this is an orchestration entry point.
def main() -> None:
    """Run the benchmark entry point."""
    generation_config = read_yaml_file("configs/generation.yaml")
    benchmark_config = read_yaml_file("configs/benchmark.yaml")
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
    results = run_example_level_benchmark(perturbed_dataset, tolerance_bpm=10.0)
    summary = summarize_benchmark_results(results)
    write_json_file(
        benchmark_config["results_path"],
        {"results": serialize_results(results), "summary": summary},
    )


if __name__ == "__main__":
    main()
