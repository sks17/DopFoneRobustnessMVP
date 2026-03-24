"""High-level clean signal generation.

Provides:
- single-example generation (``generate_clean_signal_example``)
- batch generation with deterministic per-example seeds (``generate_clean_dataset``)
- pure serialization of a ``SignalExample`` to a plain dict (``signal_example_to_record``)
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from datatypes.signal_example import SignalExample
from simulation.doppler_like import (
    generate_carrier_wave,
    generate_heartbeat_envelope,
    modulate_carrier_with_envelope,
)
from simulation.heartbeat import FloatArray, build_time_axis
from utils.seed import build_rng


# Spec:
# - General description: Create one clean synthetic Doppler-like benchmark example.
# - Params: `example_id`, non-empty identifier; `sample_rate_hz`, positive sampling rate;
#   `duration_seconds`, positive duration; `heart_rate_bpm`, positive label;
#   `carrier_frequency_hz`, positive carrier frequency.
# - Pre: Scalar parameters satisfy positivity and `example_id` is non-empty.
# - Post: Returns a valid clean `SignalExample` with `is_perturbed == False`.
# - Mathematical definition: x = c .* ((1 - m) + m * e), where c is the carrier and e
#   is the heartbeat envelope.
def generate_clean_signal_example(
    example_id: str,
    sample_rate_hz: float,
    duration_seconds: float,
    heart_rate_bpm: float,
    carrier_frequency_hz: float,
) -> SignalExample:
    """Create one clean synthetic signal example."""
    time_axis: FloatArray = build_time_axis(
        sample_rate_hz=sample_rate_hz, duration_seconds=duration_seconds
    )
    heartbeat_envelope: FloatArray = generate_heartbeat_envelope(
        time_axis=time_axis,
        heart_rate_bpm=heart_rate_bpm,
    )
    carrier_wave: FloatArray = generate_carrier_wave(
        time_axis=time_axis,
        carrier_frequency_hz=carrier_frequency_hz,
    )
    signal: FloatArray = modulate_carrier_with_envelope(
        carrier_wave=carrier_wave,
        heartbeat_envelope=heartbeat_envelope,
    )
    return SignalExample(
        example_id=example_id,
        signal=np.asarray(signal, dtype=np.float64),
        sample_rate_hz=sample_rate_hz,
        heart_rate_bpm=heart_rate_bpm,
        is_perturbed=False,
        perturbation_name=None,
    )


# ---------------------------------------------------------------------------
# Serialization (pure logic — no I/O)
# ---------------------------------------------------------------------------


# Spec:
# - General description: Convert a ``SignalExample`` into a JSON-serializable dictionary.
#   The signal array is stored as a plain Python list of floats.
# - Params: `example`, a valid ``SignalExample``.
# - Pre: ``example`` satisfies all ``SignalExample`` representation invariants.
# - Post: Returns a ``dict`` whose values are all JSON-serializable (str, float, bool,
#   None, or list[float]).  The dict contains keys ``example_id``, ``sample_rate_hz``,
#   ``heart_rate_bpm``, ``is_perturbed``, ``perturbation_name``, and ``sample_count``.
#   The raw signal samples are **not** included (they belong in a separate ``.npy`` file).
# - Mathematical definition: A field-preserving projection from the datatype to a
#   JSON-compatible mapping, omitting the high-volume signal array.
def signal_example_to_record(example: SignalExample) -> Dict[str, Any]:
    """Return a metadata dict for one signal example (no raw samples)."""
    return {
        "example_id": example.example_id,
        "sample_rate_hz": example.sample_rate_hz,
        "heart_rate_bpm": example.heart_rate_bpm,
        "is_perturbed": example.is_perturbed,
        "perturbation_name": example.perturbation_name,
        "sample_count": example.sample_count(),
    }


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


# Spec:
# - General description: Generate N clean ``SignalExample`` items, one per heart-rate
#   value, with deterministic per-example seeds derived from a base seed.
# - Params:
#     ``heart_rate_bpm_values`` — non-empty list of positive heart-rate labels.
#     ``sample_rate_hz`` — positive sampling rate.
#     ``duration_seconds`` — positive signal duration.
#     ``carrier_frequency_hz`` — positive carrier frequency.
#     ``base_seed`` — non-negative integer seed; per-example seed is ``base_seed + i``.
# - Pre: ``heart_rate_bpm_values`` is non-empty; all scalar params satisfy positivity /
#   non-negativity constraints.
# - Post: Returns a list of ``len(heart_rate_bpm_values)`` clean ``SignalExample`` values.
#   Each example has ``is_perturbed == False`` and a unique ``example_id`` of the form
#   ``"clean-{bpm_int}-s{seed}"``.
# - Mathematical definition:
#     For i in {0, ..., N-1}: seed_i = base_seed + i,
#       example_id_i = "clean-{int(bpm_i)}-s{seed_i}",
#       x_i = generate_clean_signal_example(..., bpm_i, ...).
def generate_clean_dataset(
    heart_rate_bpm_values: List[float],
    sample_rate_hz: float,
    duration_seconds: float,
    carrier_frequency_hz: float,
    base_seed: int = 0,
) -> List[SignalExample]:
    """Generate a list of clean signal examples with deterministic seeds."""
    if not heart_rate_bpm_values:
        raise ValueError("heart_rate_bpm_values must be non-empty.")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive.")
    if carrier_frequency_hz <= 0.0:
        raise ValueError("carrier_frequency_hz must be positive.")
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative.")

    dataset: List[SignalExample] = []
    for i, bpm in enumerate(heart_rate_bpm_values):
        seed_i: int = base_seed + i
        example_id: str = f"clean-{int(bpm)}-s{seed_i}"
        example: SignalExample = generate_clean_signal_example(
            example_id=example_id,
            sample_rate_hz=sample_rate_hz,
            duration_seconds=duration_seconds,
            heart_rate_bpm=bpm,
            carrier_frequency_hz=carrier_frequency_hz,
        )
        dataset.append(example)
    return dataset
