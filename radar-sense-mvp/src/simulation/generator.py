"""High-level clean signal generation."""

from __future__ import annotations

import numpy as np

from datatypes.signal_example import SignalExample
from simulation.doppler_like import (
    generate_carrier_wave,
    generate_heartbeat_envelope,
    modulate_carrier_with_envelope,
)
from simulation.heartbeat import build_time_axis


# Spec:
# - General description: Create one clean synthetic Doppler-like benchmark example.
# - Params: `example_id`, non-empty identifier; `sample_rate_hz`, positive sampling rate; `duration_seconds`, positive duration; `heart_rate_bpm`, positive label; `carrier_frequency_hz`, positive carrier frequency.
# - Pre: Scalar parameters satisfy positivity and `example_id` is non-empty.
# - Post: Returns a valid clean `SignalExample` with `is_perturbed == False`.
# - Mathematical definition: x = c .* ((1 - m) + m * e), where c is the carrier and e is the heartbeat envelope.
def generate_clean_signal_example(
    example_id: str,
    sample_rate_hz: float,
    duration_seconds: float,
    heart_rate_bpm: float,
    carrier_frequency_hz: float,
) -> SignalExample:
    """Create one clean synthetic signal example."""
    time_axis = build_time_axis(sample_rate_hz=sample_rate_hz, duration_seconds=duration_seconds)
    heartbeat_envelope = generate_heartbeat_envelope(
        time_axis=time_axis,
        heart_rate_bpm=heart_rate_bpm,
    )
    carrier_wave = generate_carrier_wave(
        time_axis=time_axis,
        carrier_frequency_hz=carrier_frequency_hz,
    )
    signal = modulate_carrier_with_envelope(
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
