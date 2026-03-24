"""Tests for the `BenchmarkResult` datatype."""

from __future__ import annotations

import pytest

from datatypes.benchmark_result import (
    BenchmarkResult,
    create_benchmark_result,
    validate_benchmark_result,
)


def test_benchmark_result_one_case_has_signed_error() -> None:
    """One-case test for a single benchmark result."""
    result = BenchmarkResult(
        example_id="result-1",
        true_heart_rate_bpm=120.0,
        estimated_heart_rate_bpm=123.0,
        absolute_error_bpm=3.0,
        within_tolerance=True,
    )

    assert result.signed_error_bpm() == 3.0


def test_benchmark_result_two_case_tolerance_branch_behaves_as_expected() -> None:
    """Two-case test for in-tolerance and out-of-tolerance construction."""
    in_tolerance = create_benchmark_result("r2a", 130.0, 132.0, tolerance_bpm=3.0)
    out_of_tolerance = create_benchmark_result("r2b", 130.0, 136.0, tolerance_bpm=3.0)

    assert in_tolerance.within_tolerance is True
    assert out_of_tolerance.within_tolerance is False


def test_benchmark_result_many_case_validation_passes() -> None:
    """Many-case test over several valid result instances."""
    results = [
        create_benchmark_result("r3a", 110.0, 111.0, tolerance_bpm=5.0),
        create_benchmark_result("r3b", 120.0, 118.0, tolerance_bpm=5.0),
        create_benchmark_result("r3c", 140.0, 135.0, tolerance_bpm=5.0),
    ]

    for result in results:
        validate_benchmark_result(result)


def test_benchmark_result_statement_case_allows_zero_estimate() -> None:
    """Statement-coverage test for a failed detection represented as zero BPM."""
    result = create_benchmark_result("r-zero", 120.0, 0.0, tolerance_bpm=5.0)
    assert result.estimated_heart_rate_bpm == 0.0
    assert result.within_tolerance is False


@pytest.mark.parametrize(
    ("example_id", "true_rate", "estimated_rate", "absolute_error"),
    [
        ("", 120.0, 121.0, 1.0),
        ("bad-true", 0.0, 121.0, 1.0),
        ("bad-estimated", 120.0, -1.0, 1.0),
        ("bad-abs", 120.0, 121.0, -1.0),
    ],
)
def test_benchmark_result_branch_cases_raise_value_error(
    example_id: str,
    true_rate: float,
    estimated_rate: float,
    absolute_error: float,
) -> None:
    """Branch-coverage test for benchmark result validation."""
    with pytest.raises(ValueError):
        BenchmarkResult(
            example_id=example_id,
            true_heart_rate_bpm=true_rate,
            estimated_heart_rate_bpm=estimated_rate,
            absolute_error_bpm=absolute_error,
            within_tolerance=False,
        )


def test_create_benchmark_result_rejects_non_positive_tolerance() -> None:
    """Statement-coverage test for tolerance precondition failure."""
    with pytest.raises(ValueError):
        create_benchmark_result("r4", 120.0, 121.0, tolerance_bpm=0.0)
