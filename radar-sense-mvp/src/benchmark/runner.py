"""Benchmark execution pipeline."""

from __future__ import annotations

from benchmark.metrics import compute_mean_absolute_error_bpm, compute_success_rate
from datatypes.benchmark_result import BenchmarkResult, create_benchmark_result
from datatypes.signal_example import SignalExample
from estimation.peak_estimator import estimate_heart_rate_bpm


# Spec:
# - General description: Evaluate one dataset of signal examples with a fixed estimator tolerance.
# - Params: `dataset`, list of signal examples; `tolerance_bpm`, positive tolerance.
# - Pre: `dataset` is non-empty and `tolerance_bpm > 0`.
# - Post: Returns one `BenchmarkResult` per example.
# - Mathematical definition: result_i = create_benchmark_result(id_i, y_i, f(x_i), tolerance).
def run_example_level_benchmark(
    dataset: list[SignalExample],
    tolerance_bpm: float,
) -> list[BenchmarkResult]:
    """Return per-example benchmark results."""
    if len(dataset) == 0:
        raise ValueError("dataset must be non-empty.")
    return [
        create_benchmark_result(
            example_id=example.example_id,
            true_heart_rate_bpm=example.heart_rate_bpm,
            estimated_heart_rate_bpm=estimate_heart_rate_bpm(
                signal=example.signal,
                sample_rate_hz=example.sample_rate_hz,
            ),
            tolerance_bpm=tolerance_bpm,
        )
        for example in dataset
    ]


# Spec:
# - General description: Summarize benchmark results into a small metric dictionary.
# - Params: `results`, non-empty list of benchmark results.
# - Pre: `results` is non-empty.
# - Post: Returns a dictionary containing `mean_absolute_error_bpm` and `success_rate`.
# - Mathematical definition: summary = {MAE(results), success_rate(results)}.
def summarize_benchmark_results(results: list[BenchmarkResult]) -> dict[str, float]:
    """Return aggregate metrics for a set of benchmark results."""
    return {
        "mean_absolute_error_bpm": compute_mean_absolute_error_bpm(results),
        "success_rate": compute_success_rate(results),
    }
