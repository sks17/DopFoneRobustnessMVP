"""Small file I/O helpers.

Provides YAML reading, JSON writing, JSONL manifest writing, and NumPy
waveform persistence.  All writers create parent directories automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
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


# Spec:
# - General description: Write a list of dictionaries to a JSONL file (one JSON object
#   per line).  This is the canonical format for dataset manifests.
# - Params: `path`, output path; `records`, non-empty list of JSON-serializable dicts.
# - Pre: `records` is a non-empty list of dicts.  Parent directory exists or can be created.
# - Post: Writes one compact JSON line per record, terminated by ``\n``.
# - Mathematical definition: For each r_i in records, the i-th line of the file is
#   ``json.dumps(r_i)``.
def write_manifest_jsonl(path: str | Path, records: List[Dict[str, Any]]) -> None:
    """Write a list of record dicts as a JSONL manifest file."""
    if not records:
        raise ValueError("records must be non-empty.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")


# Spec:
# - General description: Read a JSONL manifest file back into a list of dicts.
# - Params: `path`, file path to a JSONL document.
# - Pre: `path` exists and each line is a valid JSON object.
# - Post: Returns a list of dicts with the same length as the number of non-empty lines.
# - Mathematical definition: Inverse of ``write_manifest_jsonl``.
def read_manifest_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Read a JSONL manifest file and return a list of record dicts."""
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


# Spec:
# - General description: Save a one-dimensional float64 waveform array as a ``.npy`` file.
# - Params: `path`, output ``.npy`` path; `waveform`, one-dimensional float64 array.
# - Pre: `waveform.ndim == 1` and `waveform.size >= 2`.  Parent directory exists or can
#   be created.
# - Post: Writes a NumPy ``.npy`` file that can be loaded with ``np.load``.
# - Mathematical definition: Persists the array x in R^n to disk in NumPy binary format.
def write_waveform_npy(
    path: str | Path,
    waveform: npt.NDArray[np.float64],
) -> None:
    """Save a waveform array as a .npy file."""
    if waveform.ndim != 1:
        raise ValueError("waveform must be one-dimensional.")
    if waveform.size < 2:
        raise ValueError("waveform must contain at least two samples.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, waveform)
