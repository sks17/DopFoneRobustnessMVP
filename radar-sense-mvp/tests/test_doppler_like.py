"""Tests for Doppler-like signal synthesis."""

from __future__ import annotations

import numpy as np
import pytest

from simulation.doppler_like import (
    generate_carrier_wave,
    modulate_carrier_with_envelope,
)
from simulation.generator import generate_clean_signal_example
from simulation.heartbeat import build_time_axis


def test_generate_carrier_wave_one_case_is_sinusoidal() -> None:
    """One-case test for carrier wave generation."""
    time_axis = build_time_axis(sample_rate_hz=4.0, duration_seconds=1.0)
    carrier = generate_carrier_wave(time_axis=time_axis, carrier_frequency_hz=1.0)

    assert np.allclose(carrier, np.array([0.0, 1.0, 0.0, -1.0]), atol=1e-8)


def test_modulate_carrier_two_case_changes_amplitude_by_depth() -> None:
    """Two-case test for zero and non-zero modulation depth."""
    carrier = np.array([1.0, -1.0])
    envelope = np.array([0.0, 1.0])

    unmodulated = modulate_carrier_with_envelope(carrier, envelope, modulation_depth=0.0)
    modulated = modulate_carrier_with_envelope(carrier, envelope, modulation_depth=0.5)

    assert np.allclose(unmodulated, np.array([1.0, -1.0]))
    assert np.allclose(modulated, np.array([0.5, -1.0]))


def test_generate_clean_signal_example_many_case_returns_valid_examples() -> None:
    """Many-case test across several clean examples."""
    examples = [
        generate_clean_signal_example("g1", 200.0, 2.0, 120.0, 18.0),
        generate_clean_signal_example("g2", 200.0, 2.0, 135.0, 18.0),
        generate_clean_signal_example("g3", 200.0, 2.0, 150.0, 18.0),
    ]

    for example in examples:
        assert example.is_perturbed is False
        assert example.perturbation_name is None
        assert example.signal.shape[0] == 400


@pytest.mark.parametrize(
    ("carrier_frequency_hz", "modulation_depth"),
    [
        (0.0, 0.5),
        (1.0, -0.1),
        (1.0, 1.1),
    ],
)
def test_doppler_like_branch_cases_raise_value_error(
    carrier_frequency_hz: float,
    modulation_depth: float,
) -> None:
    """Branch-coverage test for Doppler-like preconditions."""
    time_axis = build_time_axis(sample_rate_hz=4.0, duration_seconds=1.0)
    carrier = np.array([1.0, -1.0, 1.0, -1.0])
    envelope = np.array([0.0, 0.5, 1.0, 0.5])

    if carrier_frequency_hz <= 0.0:
        with pytest.raises(ValueError):
            generate_carrier_wave(time_axis=time_axis, carrier_frequency_hz=carrier_frequency_hz)
    else:
        with pytest.raises(ValueError):
            modulate_carrier_with_envelope(
                carrier_wave=carrier,
                heartbeat_envelope=envelope,
                modulation_depth=modulation_depth,
            )
