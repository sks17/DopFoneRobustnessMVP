"""Summarize RADAR-Sense-MVP benchmark results."""

from __future__ import annotations

import json
from pathlib import Path


# Spec:
# - General description: Read a benchmark-result JSON file from disk.
# - Params: `path`, path to the JSON file.
# - Pre: `path` exists and contains JSON.
# - Post: Returns the parsed JSON object.
# - Mathematical definition: Parses serialized JSON text into an in-memory object.
def read_json_file(path: str) -> dict[str, object]:
    """Return the parsed JSON contents for a result file."""
    with Path(path).open("r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    if not isinstance(data, dict):
        raise ValueError("JSON result file must contain a dictionary.")
    return data


# Spec:
# - General description: Load and print the summary block from the benchmark result file.
# - Params: None.
# - Pre: `data/manifests/benchmark_results.json` exists and contains a `summary` key.
# - Post: Prints the summary dictionary.
# - Mathematical definition: Not applicable; this is an I/O entry point.
def main() -> None:
    """Run the result summarization entry point."""
    result_data = read_json_file("data/manifests/benchmark_results.json")
    print(result_data["summary"])


if __name__ == "__main__":
    main()
