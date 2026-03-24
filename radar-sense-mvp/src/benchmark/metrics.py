"""Aggregate benchmark metrics."""

from __future__ import annotations

from datatypes.benchmark_result import BenchmarkResult


# Spec:
# - General description: Compute mean absolute error over benchmark results.
# - Params: `results`, non-empty list of benchmark results.
# - Pre: `results` is non-empty.
# - Post: Returns the arithmetic mean of `absolute_error_bpm`.
# - Mathematical definition: MAE = (1 / n) * sum_{i=1}^{n} e_i.
def compute_mean_absolute_error_bpm(results: list[BenchmarkResult]) -> float:
    """Return the mean absolute error in beats per minute."""
    if len(results) == 0:
        raise ValueError("results must be non-empty.")
    total_error = sum(result.absolute_error_bpm for result in results)
    return total_error / float(len(results))


# Spec:
# - General description: Compute the fraction of results within tolerance.
# - Params: `results`, non-empty list of benchmark results.
# - Pre: `results` is non-empty.
# - Post: Returns a value in [0, 1].
# - Mathematical definition: success_rate = (1 / n) * sum_{i=1}^{n} 1[result_i.within_tolerance].
def compute_success_rate(results: list[BenchmarkResult]) -> float:
    """Return the fraction of results that fall within tolerance."""
    if len(results) == 0:
        raise ValueError("results must be non-empty.")
    success_count = sum(1 for result in results if result.within_tolerance)
    return success_count / float(len(results))
