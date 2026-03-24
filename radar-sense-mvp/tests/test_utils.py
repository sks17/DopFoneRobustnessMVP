"""Tests for utility helpers and script serializers."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.dataset_builder import build_clean_dataset
from benchmark.runner import run_example_level_benchmark
from scripts.build_clean_dataset import serialize_dataset
from scripts.run_benchmark import serialize_results
from scripts.summarize_results import read_json_file
from utils.io import read_yaml_file, write_json_file
from utils.logging_utils import configure_logger
from utils.seed import build_rng


def test_build_rng_one_case_is_deterministic() -> None:
    """One-case test for deterministic RNG creation."""
    rng_a = build_rng(7)
    rng_b = build_rng(7)
    assert rng_a.normal() == rng_b.normal()


def test_configure_logger_two_case_reuses_logger_name() -> None:
    """Two-case test for logger configuration reuse."""
    logger_a = configure_logger("radar_sense_test")
    logger_b = configure_logger("radar_sense_test")
    assert logger_a.name == logger_b.name
    assert len(logger_a.handlers) >= 1


def test_io_and_script_serializers_many_case_round_trip(tmp_path: Path) -> None:
    """Many-case test for YAML reading, JSON writing, and serializer helpers."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("sample_rate_hz: 200.0\nexamples_per_rate: 1\n", encoding="utf-8")
    yaml_data = read_yaml_file(yaml_path)

    config = {
        "sample_rate_hz": 200.0,
        "duration_seconds": 2.0,
        "carrier_frequency_hz": 18.0,
        "heart_rate_bpm_values": [120.0, 135.0, 150.0],
        "examples_per_rate": 1,
    }
    dataset = build_clean_dataset(config)
    results = run_example_level_benchmark(dataset, tolerance_bpm=20.0)

    json_path = tmp_path / "results.json"
    write_json_file(
        json_path,
        {"dataset": serialize_dataset(dataset), "results": serialize_results(results)},
    )
    json_data = read_json_file(str(json_path))

    assert yaml_data["sample_rate_hz"] == 200.0
    assert len(json_data["dataset"]) == 3
    assert len(json_data["results"]) == 3


def test_utils_branch_cases_raise_value_error(tmp_path: Path) -> None:
    """Branch-coverage test for utility precondition failures."""
    bad_yaml_path = tmp_path / "bad.yaml"
    bad_yaml_path.write_text("- item\n- item2\n", encoding="utf-8")

    with pytest.raises(ValueError):
        build_rng(-1)
    with pytest.raises(ValueError):
        configure_logger("")
    with pytest.raises(ValueError):
        read_yaml_file(bad_yaml_path)
