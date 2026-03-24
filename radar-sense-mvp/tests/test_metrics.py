"""Tests for aggregate benchmark metrics."""

from __future__ import annotations

import pytest

from benchmark.metrics import compute_mean_absolute_error_bpm, compute_success_rate
from datatypes.benchmark_result import create_benchmark_result


def test_compute_mean_absolute_error_one_case_matches_expected_value() -> None:
    """One-case test for mean absolute error."""
    results = [create_benchmark_result("m1", 120.0, 123.0, tolerance_bpm=5.0)]
    assert compute_mean_absolute_error_bpm(results) == 3.0


def test_compute_success_rate_two_case_matches_expected_value() -> None:
    """Two-case test for success rate."""
    results = [
        create_benchmark_result("m2a", 120.0, 121.0, tolerance_bpm=2.0),
        create_benchmark_result("m2b", 120.0, 125.0, tolerance_bpm=2.0),
    ]
    assert compute_success_rate(results) == 0.5


def test_metrics_many_case_handle_multiple_results() -> None:
    """Many-case test over several benchmark results."""
    results = [
        create_benchmark_result("m3a", 120.0, 121.0, tolerance_bpm=5.0),
        create_benchmark_result("m3b", 130.0, 127.0, tolerance_bpm=5.0),
        create_benchmark_result("m3c", 140.0, 144.0, tolerance_bpm=5.0),
    ]

    assert compute_mean_absolute_error_bpm(results) == (1.0 + 3.0 + 4.0) / 3.0
    assert compute_success_rate(results) == 1.0


def test_metrics_branch_case_reject_empty_results() -> None:
    """Branch-coverage test for empty metric inputs."""
    with pytest.raises(ValueError):
        compute_mean_absolute_error_bpm([])
    with pytest.raises(ValueError):
        compute_success_rate([])
