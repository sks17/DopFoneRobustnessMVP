"""Build the RADAR-inspired perturbed benchmark dataset.

Reads the generation config, builds clean examples, expands each across the
full artifact-type x severity x seed grid, and writes per-example waveform
files plus a JSONL manifest with full provenance metadata.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark.dataset_builder import (
    build_clean_dataset,
    expand_to_perturbed_dataset,
    perturbed_example_to_record,
)
from perturbations.registry import list_perturbation_names
from utils.io import read_yaml_file, write_manifest_jsonl, write_waveform_npy


# Default severity grid for the RADAR benchmark expansion.
DEFAULT_SEVERITIES: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]


# Spec:
# - General description: Build the full perturbed dataset and write it to disk.
# - Params: `config_path`, path to generation YAML; `output_dir`, directory for
#   waveform files; `manifest_path`, path for the output JSONL manifest.
# - Pre: Config file exists and is valid.
# - Post: Writes per-example .npy waveforms and a JSONL manifest with provenance.
# - Mathematical definition: Not applicable; this is an I/O orchestration entry
#   point.
def main(
    config_path: str = str(PROJECT_ROOT / "configs" / "generation.yaml"),
    output_dir: str = str(PROJECT_ROOT / "data" / "perturbed"),
    manifest_path: str = str(PROJECT_ROOT / "data" / "manifests" / "perturbed_manifest.jsonl"),
) -> None:
    """Run the perturbed dataset build entry point."""
    generation_config = read_yaml_file(config_path)
    base_seed = int(generation_config["seed"])

    # 1. Build clean source examples.
    clean_dataset = build_clean_dataset(generation_config)

    # 2. Expand across the RADAR benchmark grid:
    #    artifact type x severity x seed.
    perturbation_names = list_perturbation_names()
    seeds = [base_seed + i for i in range(int(generation_config.get("examples_per_rate", 3)))]

    perturbed_dataset = expand_to_perturbed_dataset(
        clean_examples=clean_dataset,
        perturbation_names=perturbation_names,
        severities=DEFAULT_SEVERITIES,
        seeds=seeds,
    )

    # 3. Write waveforms and manifest.
    output_path = Path(output_dir)
    records = []
    for perturbed_example in perturbed_dataset:
        # Recover provenance from the structured example_id.
        # ID format: "{clean_id}-{perturbation}-sev{severity}-s{seed}"
        example_id_suffix = f"-{perturbed_example.perturbation_name}-sev"
        source_id, suffix = perturbed_example.example_id.split(example_id_suffix, maxsplit=1)
        severity_text, seed_text = suffix.rsplit("-s", maxsplit=1)
        severity = float(severity_text)
        seed = int(seed_text)

        waveform_path = output_path / f"{perturbed_example.example_id}.npy"
        write_waveform_npy(str(waveform_path), perturbed_example.signal)

        record = perturbed_example_to_record(
            perturbed_example,
            source_example_id=source_id,
            severity=severity,
            seed=seed,
        )
        record["waveform_path"] = str(waveform_path)
        records.append(record)

    write_manifest_jsonl(manifest_path, records)


if __name__ == "__main__":
    main()
