"""Summarize benchmark result rows from JSONL or CSV."""

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

from benchmark.runner import summarize_clean_and_perturbed_results
from datatypes.benchmark_result import BenchmarkResult
from utils.io import read_yaml_file, write_json_file


# Spec:
# - General description: Read a JSON file from disk.
# - Params: `path`, path to a JSON file.
# - Pre: `path` exists and contains a JSON dictionary.
# - Post: Returns the parsed dictionary.
# - Mathematical definition: Parses serialized JSON text into an in-memory mapping.
def read_json_file(path: str | Path) -> dict[str, object]:
    """Return the parsed JSON contents for a result file."""
    with Path(path).open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, dict):
        raise ValueError("JSON result file must contain a dictionary.")
    return data


# Spec:
# - General description: Read benchmark result rows from a JSONL file.
# - Params: `path`, path to a JSONL file.
# - Pre: `path` exists and each non-empty line is a JSON object.
# - Post: Returns a list of row dictionaries.
# - Mathematical definition: rows_i = json.loads(line_i).
def read_result_rows_jsonl(path: str | Path) -> list[dict[str, object]]:
    """Return benchmark result rows loaded from a JSONL file."""
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped_line = line.strip()
            if stripped_line:
                rows.append(json.loads(stripped_line))
    return rows


# Spec:
# - General description: Read benchmark result rows from a CSV file.
# - Params: `path`, path to a CSV file.
# - Pre: `path` exists and contains a header row matching the result-row schema.
# - Post: Returns a list of row dictionaries.
# - Mathematical definition: Not applicable; this is tabular file parsing.
def read_result_rows_csv(path: str | Path) -> list[dict[str, object]]:
    """Return benchmark result rows loaded from a CSV file."""
    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


# Spec:
# - General description: Read benchmark result rows from either JSONL or CSV.
# - Params: `path`, path to the result-row file; `input_format`, either `jsonl` or `csv`.
# - Pre: `input_format` is supported.
# - Post: Returns a list of row dictionaries.
# - Mathematical definition: Dispatches to either JSONL or CSV parsing.
def read_result_rows(path: str | Path, input_format: str) -> list[dict[str, object]]:
    """Return benchmark result rows loaded from the requested format."""
    if input_format == "jsonl":
        return read_result_rows_jsonl(path)
    if input_format == "csv":
        return read_result_rows_csv(path)
    raise ValueError("input_format must be either 'jsonl' or 'csv'.")


# Spec:
# - General description: Convert a result-row dictionary back into a `BenchmarkResult`.
# - Params: `row`, one result-row dictionary.
# - Pre: `row` contains the benchmark-result schema fields.
# - Post: Returns a validated `BenchmarkResult`.
# - Mathematical definition: Reconstructs a datatype from its serialized scalar fields.
def row_to_benchmark_result(row: dict[str, object]) -> BenchmarkResult:
    """Return a benchmark result reconstructed from one row dictionary."""
    return BenchmarkResult(
        example_id=str(row["example_id"]),
        true_heart_rate_bpm=float(row["true_heart_rate_bpm"]),
        estimated_heart_rate_bpm=float(row["estimated_heart_rate_bpm"]),
        absolute_error_bpm=float(row["absolute_error_bpm"]),
        within_tolerance=str(row["within_tolerance"]).lower() == "true",
    )


# Spec:
# - General description: Split result rows into clean and perturbed benchmark result lists.
# - Params: `rows`, non-empty list of result-row dictionaries.
# - Pre: Each row contains a `dataset_split` field equal to `clean` or `perturbed`.
# - Post: Returns a pair `(clean_results, perturbed_results)` where both lists are non-empty.
# - Mathematical definition: clean_results = {row_i : split_i = clean}, perturbed_results = {row_i : split_i = perturbed}.
def split_result_rows_by_dataset_split(
    rows: list[dict[str, object]],
) -> tuple[list[BenchmarkResult], list[BenchmarkResult]]:
    """Return clean and perturbed result lists reconstructed from row dictionaries."""
    if len(rows) == 0:
        raise ValueError("rows must be non-empty.")
    clean_results: list[BenchmarkResult] = []
    perturbed_results: list[BenchmarkResult] = []
    for row in rows:
        dataset_split = str(row["dataset_split"])
        benchmark_result = row_to_benchmark_result(row)
        if dataset_split == "clean":
            clean_results.append(benchmark_result)
        elif dataset_split == "perturbed":
            perturbed_results.append(benchmark_result)
        else:
            raise ValueError("dataset_split must be either 'clean' or 'perturbed'.")
    if len(clean_results) == 0:
        raise ValueError("clean result rows must be non-empty.")
    if len(perturbed_results) == 0:
        raise ValueError("perturbed result rows must be non-empty.")
    return clean_results, perturbed_results


# Spec:
# - General description: Summarize result rows by reconstructing clean and perturbed benchmark result lists.
# - Params: `rows`, non-empty list of result-row dictionaries.
# - Pre: `rows` contains both clean and perturbed rows.
# - Post: Returns a nested benchmark summary dictionary.
# - Mathematical definition: summary = summarize_clean_and_perturbed_results(clean_results, perturbed_results).
def summarize_result_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    """Return benchmark summary statistics reconstructed from result rows."""
    clean_results, perturbed_results = split_result_rows_by_dataset_split(rows)
    return summarize_clean_and_perturbed_results(clean_results, perturbed_results)


# Spec:
# - General description: Read benchmark result rows, recompute the summary, write the summary JSON, and print it.
# - Params: `config_path`, path to the benchmark config.
# - Pre: `config_path` exists and the configured result-row file exists.
# - Post: Writes the summary JSON and prints the summary dictionary.
# - Mathematical definition: rows = read_result_rows(path, format), summary = summarize_result_rows(rows).
def main(config_path: str | Path = PROJECT_ROOT / "configs" / "benchmark.yaml") -> None:
    """Recompute and print the benchmark summary from persisted result rows."""
    benchmark_config: dict[str, Any] = read_yaml_file(config_path)
    result_rows_path = Path(benchmark_config["result_rows_path"])
    if not result_rows_path.is_absolute():
        result_rows_path = PROJECT_ROOT / result_rows_path
    summary_path = Path(benchmark_config["summary_path"])
    if not summary_path.is_absolute():
        summary_path = PROJECT_ROOT / summary_path
    input_format = str(benchmark_config.get("result_rows_format", "jsonl"))

    result_rows = read_result_rows(result_rows_path, input_format=input_format)
    summary = summarize_result_rows(result_rows)
    write_json_file(summary_path, summary)
    print(summary)


if __name__ == "__main__":
    main()
