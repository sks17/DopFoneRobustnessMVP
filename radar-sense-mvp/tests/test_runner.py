"""Tests for benchmark runner and manifest-driven benchmark scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from benchmark.dataset_builder import (
    build_clean_dataset,
    expand_to_perturbed_dataset,
    perturbed_example_to_record,
)
from benchmark.runner import (
    benchmark_result_to_record,
    evaluate_manifest_record,
    evaluate_manifest_records,
    summarize_clean_and_perturbed_results,
)
from datatypes.benchmark_result import create_benchmark_result
from scripts.run_benchmark import main as run_benchmark_main
from scripts.summarize_results import main as summarize_results_main, read_json_file
from simulation.generator import signal_example_to_record
from utils.io import write_manifest_jsonl, write_waveform_npy


def build_generation_config() -> dict[str, object]:
    """Return a minimal generation config for runner tests."""
    return {
        "sample_rate_hz": 500.0,
        "duration_seconds": 8.0,
        "carrier_frequency_hz": 18.0,
        "heart_rate_bpm_values": [120.0],
        "examples_per_rate": 1,
        "seed": 7,
    }


def test_evaluate_manifest_record_one_case_returns_result(tmp_path: Path) -> None:
    """One-case test for manifest-record evaluation."""
    clean_example = build_clean_dataset(build_generation_config())[0]
    waveform_path = tmp_path / "clean.npy"
    write_waveform_npy(waveform_path, clean_example.signal)
    manifest_record = signal_example_to_record(clean_example)
    manifest_record["waveform_path"] = str(waveform_path)

    result = evaluate_manifest_record(
        manifest_record=manifest_record,
        waveform_loader=lambda path: np.load(path),
        tolerance_bpm=20.0,
    )

    assert result.example_id == clean_example.example_id
    assert result.absolute_error_bpm >= 0.0


def test_evaluate_manifest_records_two_case_returns_two_results(tmp_path: Path) -> None:
    """Two-case test for evaluating two manifest records."""
    config = {**build_generation_config(), "heart_rate_bpm_values": [120.0, 150.0]}
    clean_examples = build_clean_dataset(config)
    manifest_records: list[dict[str, object]] = []
    for index, example in enumerate(clean_examples):
        waveform_path = tmp_path / f"waveform-{index}.npy"
        write_waveform_npy(waveform_path, example.signal)
        record = signal_example_to_record(example)
        record["waveform_path"] = str(waveform_path)
        manifest_records.append(record)

    results = evaluate_manifest_records(
        manifest_records=manifest_records,
        waveform_loader=lambda path: np.load(path),
        tolerance_bpm=20.0,
    )

    assert len(results) == 2


def test_summarize_clean_and_perturbed_results_many_case_has_grouped_metrics() -> None:
    """Many-case test for benchmark summary structure."""
    clean_results = [create_benchmark_result("clean-120-0", 120.0, 121.0, 10.0)]
    perturbed_results = [
        create_benchmark_result("clean-120-0-gaussian_noise-sev0.25-s7", 120.0, 124.0, 10.0),
        create_benchmark_result("clean-120-0-dropout-sev0.5-s7", 120.0, 132.0, 10.0),
        create_benchmark_result("clean-120-0-gaussian_noise-sev0.5-s8", 120.0, 126.0, 10.0),
    ]

    summary = summarize_clean_and_perturbed_results(clean_results, perturbed_results)

    assert "clean" in summary
    assert "perturbed" in summary
    assert "combined" in summary
    assert "mae_by_artifact_type" in summary["perturbed"]
    assert "mae_by_severity" in summary["perturbed"]


def test_benchmark_result_to_record_branch_case_rejects_invalid_split() -> None:
    """Branch-coverage test for invalid dataset split."""
    result = create_benchmark_result("clean-120-0", 120.0, 120.0, 10.0)
    with pytest.raises(ValueError):
        benchmark_result_to_record(result, dataset_split="other")


def test_run_benchmark_and_summarize_statement_case_small_end_to_end(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Statement-coverage test for a small end-to-end manifest-driven benchmark run."""
    generation_config = build_generation_config()
    clean_examples = build_clean_dataset(generation_config)
    perturbed_examples = expand_to_perturbed_dataset(
        clean_examples=clean_examples,
        perturbation_names=["gaussian_noise"],
        severities=[0.25],
        seeds=[7],
    )

    clean_manifest_records: list[dict[str, object]] = []
    for example in clean_examples:
        waveform_path = tmp_path / "clean_waveforms" / f"{example.example_id}.npy"
        write_waveform_npy(waveform_path, example.signal)
        record = signal_example_to_record(example)
        record["waveform_path"] = str(waveform_path)
        clean_manifest_records.append(record)

    perturbed_manifest_records: list[dict[str, object]] = []
    for example in perturbed_examples:
        waveform_path = tmp_path / "perturbed_waveforms" / f"{example.example_id}.npy"
        write_waveform_npy(waveform_path, example.signal)
        record = perturbed_example_to_record(
            example,
            source_example_id=clean_examples[0].example_id,
            severity=0.25,
            seed=7,
        )
        record["waveform_path"] = str(waveform_path)
        perturbed_manifest_records.append(record)

    clean_manifest_path = tmp_path / "clean_manifest.jsonl"
    perturbed_manifest_path = tmp_path / "perturbed_manifest.jsonl"
    write_manifest_jsonl(clean_manifest_path, clean_manifest_records)
    write_manifest_jsonl(perturbed_manifest_path, perturbed_manifest_records)

    benchmark_config_path = tmp_path / "benchmark.yaml"
    with benchmark_config_path.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(
            {
                "clean_manifest_path": str(clean_manifest_path),
                "perturbed_manifest_path": str(perturbed_manifest_path),
                "result_rows_path": str(tmp_path / "benchmark_results.jsonl"),
                "summary_path": str(tmp_path / "benchmark_summary.json"),
                "result_rows_format": "jsonl",
                "tolerance_bpm": 20.0,
            },
            output_file,
        )

    run_benchmark_main(config_path=benchmark_config_path)
    summarize_results_main(config_path=benchmark_config_path)
    captured = capsys.readouterr()
    summary = read_json_file(tmp_path / "benchmark_summary.json")

    assert "combined" in summary
    assert "perturbed" in summary
    assert "mae_by_artifact_type" in summary["perturbed"]
    assert "gaussian_noise" in captured.out
