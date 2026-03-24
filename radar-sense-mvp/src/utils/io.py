"""Small file I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


# Spec:
# - General description: Read a YAML file into a Python dictionary.
# - Params: `path`, file path to a YAML document.
# - Pre: `path` exists and contains a dictionary-shaped YAML document.
# - Post: Returns the parsed dictionary.
# - Mathematical definition: Parses the serialized YAML mapping into an in-memory mapping.
def read_yaml_file(path: str | Path) -> dict[str, Any]:
    """Return the parsed contents of a YAML file."""
    with Path(path).open("r", encoding="utf-8") as input_file:
        data = yaml.safe_load(input_file)
    if not isinstance(data, dict):
        raise ValueError("YAML file must contain a dictionary.")
    return data


# Spec:
# - General description: Write a JSON-serializable object to disk with indentation.
# - Params: `path`, output path; `data`, JSON-serializable object.
# - Pre: Parent directory for `path` exists or can be created.
# - Post: Writes `data` to `path` in UTF-8 JSON format.
# - Mathematical definition: Serializes the in-memory object into JSON text.
def write_json_file(path: str | Path, data: Any) -> None:
    """Write a JSON object to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)
