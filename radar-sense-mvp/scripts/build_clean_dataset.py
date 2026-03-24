"""Build the clean synthetic dataset for RADAR-Sense-MVP.

Reads ``configs/generation.yaml``, generates clean signal examples via
``generate_clean_dataset``, writes each waveform as a ``.npy`` file, and
writes a JSONL manifest summarising every example.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from datatypes.signal_example import SignalExample
from simulation.generator import (
    generate_clean_dataset,
    signal_example_to_record,
)
from utils.io import read_yaml_file, write_manifest_jsonl, write_waveform_npy


# Spec:
# - General description: Build the clean dataset from a config file, write waveforms to
#   disk as individual ``.npy`` files, and write a JSONL manifest.
# - Params:
#     ``config_path`` — path to the generation YAML config.
#     ``output_dir`` — root directory for waveform files.
#     ``manifest_path`` — output path for the JSONL manifest.
# - Pre: ``config_path`` exists and contains valid generation parameters.
# - Post: ``output_dir`` contains one ``.npy`` per example; ``manifest_path`` is a JSONL
#   file with one record per example, each record including a ``waveform_path`` key.
# - Mathematical definition: Not applicable; this is an orchestration entry point.
def main(
    config_path: str = "configs/generation.yaml",
    output_dir: str = "data/waveforms/clean",
    manifest_path: str = "data/manifests/clean_manifest.jsonl",
) -> None:
    """Run the clean dataset build pipeline."""
    config: Dict[str, Any] = read_yaml_file(config_path)

    dataset: List[SignalExample] = generate_clean_dataset(
        heart_rate_bpm_values=[float(v) for v in config["heart_rate_bpm_values"]],
        sample_rate_hz=float(config["sample_rate_hz"]),
        duration_seconds=float(config["duration_seconds"]),
        carrier_frequency_hz=float(config["carrier_frequency_hz"]),
        base_seed=int(config.get("seed", 0)),
    )

    out = Path(output_dir)
    records: List[Dict[str, Any]] = []
    for example in dataset:
        waveform_path = out / f"{example.example_id}.npy"
        write_waveform_npy(waveform_path, example.signal)
        record = signal_example_to_record(example)
        record["waveform_path"] = str(waveform_path)
        records.append(record)

    write_manifest_jsonl(manifest_path, records)


if __name__ == "__main__":
    main()
