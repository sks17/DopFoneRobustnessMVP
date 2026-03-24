"""Tests for benchmark metric helpers."""

from __future__ import annotations

import pytest

from benchmark.metrics import (
    compute_absolute_error_bpm,
    compute_grouped_mae_by_artifact_type,
    compute_grouped_mae_by_severity,
    compute_mean_absolute_error_bpm,
    compute_success_rate,
    extract_artifact_type_from_example_id,
    extract_severity_from_example_id,
    group_results_by_artifact_type,
    group_results_by_severity,
    parse_perturbed_example_id,
)
from datatypes.benchmark_result import create_benchmark_result


def build_result(example_id: str, true_bpm: float, estimated_bpm: float) -> object:
    """Return a benchmark result for metrics tests."""
    return create_benchmark_result(
        example_id=example_id,
        true_heart_rate_bpm=true_bpm,
        estimated_heart_rate_bpm=estimated_bpm,
        tolerance_bpm=10.0,
    )


def test_compute_absolute_error_bpm_one_case_matches_expected_value() -> None:
    """One-case test for absolute error."""
    assert compute_absolute_error_bpm(120.0, 126.0) == 6.0


def test_compute_mean_absolute_error_bpm_two_case_matches_expected_value() -> None:
    """Two-case test for MAE over two results."""
    results = [
        build_result("clean-120-0-gaussian_noise-sev0.0-s7", 120.0, 122.0),
        build_result("clean-120-1-gaussian_noise-sev0.0-s7", 120.0, 126.0),
    ]
    assert compute_mean_absolute_error_bpm(results) == 4.0


def test_compute_grouped_mae_many_case_by_artifact_and_severity() -> None:
    """Many-case test for grouped MAE computations."""
    results = [
        build_result("clean-120-0-gaussian_noise-sev0.25-s7", 120.0, 124.0),
        build_result("clean-135-0-gaussian_noise-sev0.25-s7", 135.0, 141.0),
        build_result("clean-120-0-dropout-sev0.50-s7", 120.0, 130.0),
        build_result("clean-135-0-dropout-sev0.50-s7", 135.0, 147.0),
        build_result("clean-150-0-attenuation-sev0.25-s7", 150.0, 156.0),
    ]

    grouped_by_artifact = compute_grouped_mae_by_artifact_type(results)
    grouped_by_severity = compute_grouped_mae_by_severity(results)

    assert grouped_by_artifact["gaussian_noise"] == pytest.approx(5.0)
    assert grouped_by_artifact["dropout"] == pytest.approx(11.0)
    assert grouped_by_artifact["attenuation"] == pytest.approx(6.0)
    assert grouped_by_severity[0.25] == pytest.approx((4.0 + 6.0 + 6.0) / 3.0)
    assert grouped_by_severity[0.5] == pytest.approx((10.0 + 12.0) / 2.0)


def test_group_helpers_statement_case_preserve_membership_counts() -> None:
    """Statement-coverage test for grouping helper membership."""
    results = [
        build_result("clean-120-0-gaussian_noise-sev0.25-s7", 120.0, 124.0),
        build_result("clean-120-1-gaussian_noise-sev0.25-s8", 120.0, 126.0),
        build_result("clean-120-2-dropout-sev0.50-s7", 120.0, 129.0),
    ]

    grouped_by_artifact = group_results_by_artifact_type(results)
    grouped_by_severity = group_results_by_severity(results)

    assert len(grouped_by_artifact["gaussian_noise"]) == 2
    assert len(grouped_by_artifact["dropout"]) == 1
    assert len(grouped_by_severity[0.25]) == 2
    assert len(grouped_by_severity[0.5]) == 1


def test_extract_helpers_two_case_parse_expected_tokens() -> None:
    """Two-case test for identifier parsing helpers."""
    example_id_a = "clean-120-0-gaussian_noise-sev0.25-s7"
    example_id_b = "clean-135-2-dropout-sev1.0-s9"

    assert extract_artifact_type_from_example_id(example_id_a) == "gaussian_noise"
    assert extract_severity_from_example_id(example_id_a) == 0.25
    assert extract_artifact_type_from_example_id(example_id_b) == "dropout"
    assert extract_severity_from_example_id(example_id_b) == 1.0


def test_parse_perturbed_example_id_statement_case_returns_pair() -> None:
    """Statement-coverage test for the shared identifier parser."""
    artifact_type, severity = parse_perturbed_example_id(
        "clean-120-0-attenuation-sev0.75-s7"
    )
    assert artifact_type == "attenuation"
    assert severity == 0.75


@pytest.mark.parametrize(
    ("true_bpm", "estimated_bpm"),
    [
        (0.0, 120.0),
        (120.0, -1.0),
    ],
)
def test_compute_absolute_error_bpm_branch_cases_raise_value_error(
    true_bpm: float,
    estimated_bpm: float,
) -> None:
    """Branch-coverage test for absolute-error input validation."""
    with pytest.raises(ValueError):
        compute_absolute_error_bpm(true_bpm, estimated_bpm)


def test_metric_aggregators_branch_case_reject_empty_results() -> None:
    """Branch-coverage test for empty metric inputs."""
    with pytest.raises(ValueError):
        compute_mean_absolute_error_bpm([])
    with pytest.raises(ValueError):
        compute_success_rate([])
    with pytest.raises(ValueError):
        compute_grouped_mae_by_artifact_type([])
    with pytest.raises(ValueError):
        compute_grouped_mae_by_severity([])


@pytest.mark.parametrize(
    "example_id",
    [
        "",
        "clean-120-0",
        "clean-120-0-gaussian_noise-severity0.25-s7",
        "clean-120-0-gaussian_noise-sev1.5-s7",
    ],
)
def test_identifier_parsing_branch_cases_raise_value_error(example_id: str) -> None:
    """Branch-coverage test for invalid perturbed identifier parsing."""
    with pytest.raises(ValueError):
        extract_artifact_type_from_example_id(example_id)


@pytest.mark.parametrize(
    "example_id",
    [
        "",
        "clean-120-0",
        "clean-120-0-gaussian_noise-severity0.25-s7",
        "clean-120-0-gaussian_noise-sev1.5-s7",
    ],
)
def test_severity_parsing_branch_cases_raise_value_error(example_id: str) -> None:
    """Branch-coverage test for invalid severity parsing."""
    with pytest.raises(ValueError):
        extract_severity_from_example_id(example_id)
