"""Tests for dataset construction and benchmark orchestration."""

from __future__ import annotations

import pytest

from benchmark.dataset_builder import (
    build_clean_dataset,
    build_default_perturbation_parameters,
    build_perturbed_dataset,
)
from benchmark.runner import run_example_level_benchmark, summarize_benchmark_results


def test_build_clean_dataset_one_case_creates_expected_count() -> None:
    """One-case test for clean dataset construction."""
    config = {
        "sample_rate_hz": 200.0,
        "duration_seconds": 2.0,
        "carrier_frequency_hz": 18.0,
        "heart_rate_bpm_values": [120.0],
        "examples_per_rate": 1,
    }

    dataset = build_clean_dataset(config)

    assert len(dataset) == 1
    assert dataset[0].is_perturbed is False


def test_build_perturbed_dataset_two_case_preserves_length() -> None:
    """Two-case test for perturbing a two-example dataset."""
    config = {
        "sample_rate_hz": 200.0,
        "duration_seconds": 2.0,
        "carrier_frequency_hz": 18.0,
        "heart_rate_bpm_values": [120.0],
        "examples_per_rate": 2,
    }
    clean_dataset = build_clean_dataset(config)

    perturbed_dataset = build_perturbed_dataset(
        clean_dataset=clean_dataset,
        perturbation_name="attenuation",
        parameters={"attenuation_factor": 0.5},
    )

    assert len(perturbed_dataset) == 2
    assert all(example.is_perturbed for example in perturbed_dataset)


def test_dataset_builder_many_case_and_runner_summary_work() -> None:
    """Many-case test covering dataset generation, perturbation defaults, and summary metrics."""
    config = {
        "sample_rate_hz": 200.0,
        "duration_seconds": 4.0,
        "carrier_frequency_hz": 18.0,
        "heart_rate_bpm_values": [120.0, 135.0, 150.0],
        "examples_per_rate": 1,
    }
    clean_dataset = build_clean_dataset(config)
    noise_parameters = build_default_perturbation_parameters("gaussian_noise", seed=7)
    perturbed_dataset = build_perturbed_dataset(
        clean_dataset=clean_dataset,
        perturbation_name="gaussian_noise",
        parameters=noise_parameters,
    )
    results = run_example_level_benchmark(perturbed_dataset, tolerance_bpm=20.0)
    summary = summarize_benchmark_results(results)

    assert len(results) == 3
    assert set(summary.keys()) == {"mean_absolute_error_bpm", "success_rate"}
    assert 0.0 <= summary["success_rate"] <= 1.0


def test_dataset_builder_branch_cases_raise_value_error() -> None:
    """Branch-coverage test for invalid dataset-builder inputs."""
    with pytest.raises(ValueError):
        build_perturbed_dataset([], perturbation_name="attenuation", parameters={"attenuation_factor": 0.5})
    with pytest.raises(ValueError):
        build_default_perturbation_parameters("unknown", seed=7)
