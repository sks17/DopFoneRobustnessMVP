"""Benchmark execution helpers with clean separation from file I/O."""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt

from benchmark.metrics import (
    compute_grouped_mae_by_artifact_type,
    compute_grouped_mae_by_severity,
    compute_mean_absolute_error_bpm,
    compute_success_rate,
    extract_artifact_type_from_example_id,
    extract_severity_from_example_id,
)
from datatypes.benchmark_result import BenchmarkResult, create_benchmark_result
from datatypes.signal_example import SignalExample
from estimation.peak_estimator import estimate_heart_rate_bpm


FloatArray = npt.NDArray[np.float64]
WaveformLoader = Callable[[str], FloatArray]


# Spec:
# - General description: Evaluate one in-memory signal example with the MVP estimator.
# - Params: `example`, valid signal example; `tolerance_bpm`, positive evaluation tolerance.
# - Pre: `example` is valid and `tolerance_bpm > 0`.
# - Post: Returns one validated `BenchmarkResult`.
# - Mathematical definition: result = create_benchmark_result(example_id, y, f(x), tolerance).
def evaluate_signal_example(
    example: SignalExample,
    tolerance_bpm: float,
) -> BenchmarkResult:
    """Return the benchmark result for one in-memory signal example."""
    if tolerance_bpm <= 0.0:
        raise ValueError("tolerance_bpm must be positive.")
    estimated_heart_rate_bpm = estimate_heart_rate_bpm(
        signal=example.signal,
        sample_rate_hz=example.sample_rate_hz,
    )
    return create_benchmark_result(
        example_id=example.example_id,
        true_heart_rate_bpm=example.heart_rate_bpm,
        estimated_heart_rate_bpm=estimated_heart_rate_bpm,
        tolerance_bpm=tolerance_bpm,
    )


# Spec:
# - General description: Evaluate a non-empty list of in-memory signal examples.
# - Params: `dataset`, non-empty list of signal examples; `tolerance_bpm`, positive evaluation tolerance.
# - Pre: `dataset` is non-empty and `tolerance_bpm > 0`.
# - Post: Returns one benchmark result per example.
# - Mathematical definition: results_i = evaluate_signal_example(dataset_i, tolerance_bpm).
def run_example_level_benchmark(
    dataset: list[SignalExample],
    tolerance_bpm: float,
) -> list[BenchmarkResult]:
    """Return one benchmark result per signal example."""
    if len(dataset) == 0:
        raise ValueError("dataset must be non-empty.")
    return [evaluate_signal_example(example, tolerance_bpm) for example in dataset]


# Spec:
# - General description: Evaluate one manifest record using an injected waveform loader.
# - Params: `manifest_record`, record with waveform path and label fields; `waveform_loader`, callable that loads a waveform from a path string; `tolerance_bpm`, positive evaluation tolerance.
# - Pre: `manifest_record` contains keys `example_id`, `waveform_path`, `sample_rate_hz`, and `heart_rate_bpm`; `tolerance_bpm > 0`.
# - Post: Returns one validated `BenchmarkResult`.
# - Mathematical definition: waveform = waveform_loader(path), estimate = f(waveform), result = create_benchmark_result(id, y, estimate, tolerance).
def evaluate_manifest_record(
    manifest_record: dict[str, object],
    waveform_loader: WaveformLoader,
    tolerance_bpm: float,
) -> BenchmarkResult:
    """Return the benchmark result for one manifest record."""
    if tolerance_bpm <= 0.0:
        raise ValueError("tolerance_bpm must be positive.")
    waveform_path = str(manifest_record["waveform_path"])
    example_id = str(manifest_record["example_id"])
    sample_rate_hz = float(manifest_record["sample_rate_hz"])
    true_heart_rate_bpm = float(manifest_record["heart_rate_bpm"])
    waveform = waveform_loader(waveform_path)
    estimated_heart_rate_bpm = estimate_heart_rate_bpm(
        signal=waveform,
        sample_rate_hz=sample_rate_hz,
    )
    return create_benchmark_result(
        example_id=example_id,
        true_heart_rate_bpm=true_heart_rate_bpm,
        estimated_heart_rate_bpm=estimated_heart_rate_bpm,
        tolerance_bpm=tolerance_bpm,
    )


# Spec:
# - General description: Evaluate a non-empty list of manifest records using an injected waveform loader.
# - Params: `manifest_records`, non-empty list of manifest records; `waveform_loader`, callable waveform loader; `tolerance_bpm`, positive evaluation tolerance.
# - Pre: `manifest_records` is non-empty and each record satisfies `evaluate_manifest_record` preconditions.
# - Post: Returns one benchmark result per manifest record.
# - Mathematical definition: results_i = evaluate_manifest_record(record_i, waveform_loader, tolerance_bpm).
def evaluate_manifest_records(
    manifest_records: list[dict[str, object]],
    waveform_loader: WaveformLoader,
    tolerance_bpm: float,
) -> list[BenchmarkResult]:
    """Return benchmark results for a non-empty manifest-record list."""
    if len(manifest_records) == 0:
        raise ValueError("manifest_records must be non-empty.")
    return [
        evaluate_manifest_record(record, waveform_loader, tolerance_bpm)
        for record in manifest_records
    ]


# Spec:
# - General description: Convert one benchmark result into a JSON/CSV-ready record with dataset-split metadata.
# - Params: `result`, benchmark result; `dataset_split`, either `clean` or `perturbed`.
# - Pre: `dataset_split` is `clean` or `perturbed`.
# - Post: Returns a dictionary containing benchmark fields and split metadata; perturbed rows also include artifact type and severity.
# - Mathematical definition: record is a field-preserving projection of `result`, augmented by split-specific provenance fields.
def benchmark_result_to_record(
    result: BenchmarkResult,
    dataset_split: str,
) -> dict[str, object]:
    """Return a serializable result row."""
    if dataset_split not in {"clean", "perturbed"}:
        raise ValueError("dataset_split must be either 'clean' or 'perturbed'.")
    record: dict[str, object] = {
        "dataset_split": dataset_split,
        "example_id": result.example_id,
        "true_heart_rate_bpm": result.true_heart_rate_bpm,
        "estimated_heart_rate_bpm": result.estimated_heart_rate_bpm,
        "absolute_error_bpm": result.absolute_error_bpm,
        "within_tolerance": result.within_tolerance,
        "artifact_type": None,
        "severity": None,
    }
    if dataset_split == "perturbed":
        record["artifact_type"] = extract_artifact_type_from_example_id(result.example_id)
        record["severity"] = extract_severity_from_example_id(result.example_id)
    return record


# Spec:
# - General description: Convert a list of benchmark results into serializable rows with split metadata.
# - Params: `results`, list of benchmark results; `dataset_split`, either `clean` or `perturbed`.
# - Pre: `dataset_split` is `clean` or `perturbed`.
# - Post: Returns one serializable row per result.
# - Mathematical definition: rows_i = benchmark_result_to_record(results_i, dataset_split).
def benchmark_results_to_records(
    results: list[BenchmarkResult],
    dataset_split: str,
) -> list[dict[str, object]]:
    """Return serializable result rows."""
    return [benchmark_result_to_record(result, dataset_split) for result in results]


# Spec:
# - General description: Summarize one non-empty set of benchmark results by overall MAE and success rate.
# - Params: `results`, non-empty list of benchmark results.
# - Pre: `results` is non-empty.
# - Post: Returns a dictionary with overall MAE and success rate.
# - Mathematical definition: summary = {MAE(results), success_rate(results)}.
def summarize_benchmark_results(results: list[BenchmarkResult]) -> dict[str, float]:
    """Return overall benchmark summary metrics for one result set."""
    return {
        "mean_absolute_error_bpm": compute_mean_absolute_error_bpm(results),
        "success_rate": compute_success_rate(results),
    }


# Spec:
# - General description: Summarize a pair of clean and perturbed benchmark result lists.
# - Params: `clean_results`, non-empty list of clean benchmark results; `perturbed_results`, non-empty list of perturbed benchmark results.
# - Pre: Both lists are non-empty and perturbed result identifiers follow the perturbed benchmark identifier format.
# - Post: Returns a nested dictionary with combined, clean-only, and perturbed-only summaries, including grouped perturbed MAE by artifact type and severity.
# - Mathematical definition: clean_summary = S(clean_results), perturbed_summary = S(perturbed_results) union grouped MAE metrics, combined_summary = S(clean_results union perturbed_results).
def summarize_clean_and_perturbed_results(
    clean_results: list[BenchmarkResult],
    perturbed_results: list[BenchmarkResult],
) -> dict[str, object]:
    """Return benchmark summaries for clean, perturbed, and combined result sets."""
    if len(clean_results) == 0:
        raise ValueError("clean_results must be non-empty.")
    if len(perturbed_results) == 0:
        raise ValueError("perturbed_results must be non-empty.")
    combined_results = clean_results + perturbed_results
    return {
        "clean": summarize_benchmark_results(clean_results),
        "perturbed": {
            **summarize_benchmark_results(perturbed_results),
            "mae_by_artifact_type": compute_grouped_mae_by_artifact_type(perturbed_results),
            "mae_by_severity": compute_grouped_mae_by_severity(perturbed_results),
        },
        "combined": summarize_benchmark_results(combined_results),
    }
