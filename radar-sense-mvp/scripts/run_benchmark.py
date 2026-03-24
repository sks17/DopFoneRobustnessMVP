"""Run the manifest-driven RADAR-Sense-MVP benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.runner import (
    benchmark_results_to_records,
    evaluate_manifest_records,
    summarize_clean_and_perturbed_results,
)
from estimation.preprocess import load_waveform_npy
from utils.io import read_manifest_jsonl, read_yaml_file, write_json_file


RESULT_FIELD_NAMES: list[str] = [
    "dataset_split",
    "example_id",
    "true_heart_rate_bpm",
    "estimated_heart_rate_bpm",
    "absolute_error_bpm",
    "within_tolerance",
    "artifact_type",
    "severity",
]


# Spec:
# - General description: Convert benchmark results to serializable row dictionaries for one dataset split.
# - Params: `results`, list of benchmark results; `dataset_split`, either `clean` or `perturbed`.
# - Pre: `dataset_split` is `clean` or `perturbed`.
# - Post: Returns one row dictionary per result.
# - Mathematical definition: rows_i = benchmark_results_to_records(results_i, dataset_split).
def serialize_results(
    results: list[object],
    dataset_split: str,
) -> list[dict[str, object]]:
    """Return serializable benchmark result rows."""
    return benchmark_results_to_records(results, dataset_split)


# Spec:
# - General description: Write a list of result rows to a JSONL file.
# - Params: `path`, output file path; `rows`, list of serializable row dictionaries.
# - Pre: `rows` is non-empty.
# - Post: Writes one JSON object per line.
# - Mathematical definition: line_i = json.dumps(rows_i).
def write_result_rows_jsonl(path: str | Path, rows: list[dict[str, object]]) -> None:
    """Write benchmark result rows to a JSONL file."""
    if len(rows) == 0:
        raise ValueError("rows must be non-empty.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row) + "\n")


# Spec:
# - General description: Write a list of result rows to a CSV file.
# - Params: `path`, output file path; `rows`, list of serializable row dictionaries.
# - Pre: `rows` is non-empty.
# - Post: Writes one CSV header row followed by one data row per result.
# - Mathematical definition: Not applicable; this is tabular file serialization.
def write_result_rows_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    """Write benchmark result rows to a CSV file."""
    if len(rows) == 0:
        raise ValueError("rows must be non-empty.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RESULT_FIELD_NAMES)
        writer.writeheader()
        writer.writerows(rows)


# Spec:
# - General description: Write benchmark result rows using either JSONL or CSV format.
# - Params: `path`, output file path; `rows`, non-empty row list; `output_format`, either `jsonl` or `csv`.
# - Pre: `rows` is non-empty and `output_format` is supported.
# - Post: Writes the result rows to disk.
# - Mathematical definition: Dispatches to either JSONL or CSV serialization.
def write_result_rows(
    path: str | Path,
    rows: list[dict[str, object]],
    output_format: str,
) -> None:
    """Write benchmark result rows to the requested format."""
    if output_format == "jsonl":
        write_result_rows_jsonl(path, rows)
        return
    if output_format == "csv":
        write_result_rows_csv(path, rows)
        return
    raise ValueError("output_format must be either 'jsonl' or 'csv'.")


# Spec:
# - General description: Resolve a project-relative path from configuration.
# - Params: `path_text`, path string from configuration.
# - Pre: `path_text` is non-empty.
# - Post: Returns an absolute `Path`.
# - Mathematical definition: resolved_path = PROJECT_ROOT / path_text when relative, else Path(path_text).
def resolve_project_path(path_text: str) -> Path:
    """Return an absolute path resolved from the project root."""
    if path_text.strip() == "":
        raise ValueError("path_text must be non-empty.")
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


# Spec:
# - General description: Run the benchmark over clean and perturbed manifests and write result rows plus summary artifacts.
# - Params: `config_path`, path to the benchmark YAML config.
# - Pre: `config_path` exists and the referenced manifests exist.
# - Post: Writes benchmark result rows and a summary JSON file.
# - Mathematical definition: clean_results = E(clean_manifest), perturbed_results = E(perturbed_manifest), summary = summarize_clean_and_perturbed_results(clean_results, perturbed_results).
def main(config_path: str | Path = PROJECT_ROOT / "configs" / "benchmark.yaml") -> None:
    """Run the benchmark over clean and perturbed manifests."""
    benchmark_config: dict[str, Any] = read_yaml_file(config_path)
    clean_manifest_path = resolve_project_path(str(benchmark_config["clean_manifest_path"]))
    perturbed_manifest_path = resolve_project_path(str(benchmark_config["perturbed_manifest_path"]))
    result_rows_path = resolve_project_path(str(benchmark_config["result_rows_path"]))
    summary_path = resolve_project_path(str(benchmark_config["summary_path"]))
    output_format = str(benchmark_config.get("result_rows_format", "jsonl"))
    tolerance_bpm = float(benchmark_config.get("tolerance_bpm", 10.0))

    clean_manifest_records = read_manifest_jsonl(clean_manifest_path)
    perturbed_manifest_records = read_manifest_jsonl(perturbed_manifest_path)

    clean_results = evaluate_manifest_records(
        clean_manifest_records,
        waveform_loader=load_waveform_npy,
        tolerance_bpm=tolerance_bpm,
    )
    perturbed_results = evaluate_manifest_records(
        perturbed_manifest_records,
        waveform_loader=load_waveform_npy,
        tolerance_bpm=tolerance_bpm,
    )

    result_rows = (
        serialize_results(clean_results, dataset_split="clean")
        + serialize_results(perturbed_results, dataset_split="perturbed")
    )
    summary = summarize_clean_and_perturbed_results(clean_results, perturbed_results)

    write_result_rows(result_rows_path, result_rows, output_format=output_format)
    write_json_file(summary_path, summary)


if __name__ == "__main__":
    main()
