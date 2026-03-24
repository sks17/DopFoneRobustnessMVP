"""RADAR-inspired clean and perturbed dataset construction.

The RADAR benchmark model works by:
1. Generating clean source examples with known objective labels (true BPM).
2. Expanding each clean example across a grid of artifact type x severity x seed.
3. Preserving the objective label through all perturbations so that estimator
   accuracy can be measured against ground truth.

This module provides the dataset-building layer for that pipeline.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from datatypes.signal_example import SignalExample
from perturbations.registry import (
    apply_perturbation_with_severity,
    list_perturbation_names,
)
from simulation.generator import generate_clean_signal_example, signal_example_to_record


# ---------------------------------------------------------------------------
# Perturbations that require a ``seed`` keyword argument for reproducibility.
# Deterministic perturbations (dropout, attenuation) do not need a seed.
# ---------------------------------------------------------------------------
_SEED_PERTURBATION_NAMES: frozenset[str] = frozenset({"gaussian_noise"})


# Spec:
# - General description: Build a list of clean synthetic signal examples from a
#   small config dictionary.
# - Params: `generation_config`, dictionary containing sample rate, duration,
#   carrier frequency, heart rates, and examples per rate.
# - Pre: Required config keys exist and carry positive values.
# - Post: Returns a non-empty list of clean `SignalExample` values.
# - Mathematical definition: Produces one example for each pair in
#   heart_rate_bpm_values x {0, ..., examples_per_rate - 1}.
def build_clean_dataset(generation_config: dict[str, object]) -> list[SignalExample]:
    """Return a list of clean synthetic signal examples."""
    sample_rate_hz = float(generation_config["sample_rate_hz"])
    duration_seconds = float(generation_config["duration_seconds"])
    carrier_frequency_hz = float(generation_config["carrier_frequency_hz"])
    heart_rate_bpm_values = list(generation_config["heart_rate_bpm_values"])
    examples_per_rate = int(generation_config["examples_per_rate"])

    dataset: list[SignalExample] = []
    for heart_rate_bpm in heart_rate_bpm_values:
        for example_index in range(examples_per_rate):
            example_id = f"clean-{int(float(heart_rate_bpm))}-{example_index}"
            dataset.append(
                generate_clean_signal_example(
                    example_id=example_id,
                    sample_rate_hz=sample_rate_hz,
                    duration_seconds=duration_seconds,
                    heart_rate_bpm=float(heart_rate_bpm),
                    carrier_frequency_hz=carrier_frequency_hz,
                )
            )
    return dataset


# Spec:
# - General description: Expand clean examples across the full cross-product of
#   perturbation name x severity x seed.  This is the core RADAR-inspired
#   expansion: each clean source example is perturbed by every combination of
#   artifact type, severity level, and random seed to produce a comprehensive
#   robustness-testing grid.
# - Params: `clean_examples`, list of clean SignalExample items;
#   `perturbation_names`, list of registered perturbation names;
#   `severities`, list of floats in [0, 1];
#   `seeds`, list of non-negative integers.
# - Pre: All four lists are non-empty, severities lie in [0, 1], seeds are
#   non-negative, and every perturbation name is registered.
# - Post: Returns a list of
#   len(clean_examples) * len(perturbation_names) * len(severities) * len(seeds)
#   perturbed SignalExample items.  Each output preserves the heart_rate_bpm of
#   its source (the RADAR invariant: the objective label is unchanged).
# - Mathematical definition: output = { P_name(x_i, severity_j, seed_k)
#   for x_i in clean_examples, for name in perturbation_names,
#   for severity_j in severities, for seed_k in seeds }.
def expand_to_perturbed_dataset(
    clean_examples: List[SignalExample],
    perturbation_names: List[str],
    severities: List[float],
    seeds: List[int],
) -> List[SignalExample]:
    """Expand clean examples into a perturbed benchmark grid.

    Each clean example is perturbed by every (perturbation, severity, seed)
    combination, preserving the objective heart-rate label throughout.
    """
    if len(clean_examples) == 0:
        raise ValueError("clean_examples must be non-empty.")
    if len(perturbation_names) == 0:
        raise ValueError("perturbation_names must be non-empty.")
    if len(severities) == 0:
        raise ValueError("severities must be non-empty.")
    if len(seeds) == 0:
        raise ValueError("seeds must be non-empty.")

    registered_names = set(list_perturbation_names())
    for name in perturbation_names:
        if name not in registered_names:
            raise ValueError(f"Unknown perturbation_name: {name}")

    for severity in severities:
        if not 0.0 <= severity <= 1.0:
            raise ValueError(f"severity must lie in [0, 1], got {severity}.")

    for seed in seeds:
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}.")

    perturbed: List[SignalExample] = []

    for clean_example in clean_examples:
        for name in perturbation_names:
            for severity in severities:
                for seed in seeds:
                    # Build extra keyword arguments for the severity function.
                    # Only stochastic perturbations receive a seed.
                    extra_kwargs: dict[str, object] = {}
                    if name in _SEED_PERTURBATION_NAMES:
                        extra_kwargs["seed"] = seed

                    perturbed_example = apply_perturbation_with_severity(
                        clean_example,
                        perturbation_name=name,
                        severity=severity,
                        **extra_kwargs,
                    )

                    # Overwrite the auto-generated example_id with a
                    # fully-qualified RADAR benchmark identifier that
                    # encodes the source, perturbation, severity, and seed.
                    perturbed_example = SignalExample(
                        example_id=(
                            f"{clean_example.example_id}"
                            f"-{name}"
                            f"-sev{severity}"
                            f"-s{seed}"
                        ),
                        signal=perturbed_example.signal,
                        sample_rate_hz=perturbed_example.sample_rate_hz,
                        heart_rate_bpm=perturbed_example.heart_rate_bpm,
                        is_perturbed=perturbed_example.is_perturbed,
                        perturbation_name=perturbed_example.perturbation_name,
                    )

                    perturbed.append(perturbed_example)

    return perturbed


# Spec:
# - General description: Serialize a perturbed SignalExample to a JSONL-ready
#   dictionary with full RADAR benchmark provenance metadata.
# - Params: `example`, a perturbed SignalExample; `source_example_id`, the
#   example_id of the clean source; `severity`, the severity level used;
#   `seed`, the random seed used.
# - Pre: `example` is a valid perturbed SignalExample.
# - Post: Returns a JSON-serializable dictionary extending the base record with
#   `source_example_id`, `severity`, and `seed`.
# - Mathematical definition: record = signal_example_to_record(example) ∪
#   {source_example_id, severity, seed}.
def perturbed_example_to_record(
    example: SignalExample,
    source_example_id: str,
    severity: float,
    seed: int,
) -> Dict[str, object]:
    """Serialize a perturbed example with RADAR benchmark provenance."""
    record = signal_example_to_record(example)
    record["source_example_id"] = source_example_id
    record["severity"] = severity
    record["seed"] = seed
    return record
