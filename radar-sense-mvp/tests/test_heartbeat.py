"""Tests for heartbeat-envelope generation."""

from __future__ import annotations

import numpy as np
import pytest

from simulation.heartbeat import (
    build_time_axis,
    generate_heartbeat_envelope,
    heart_rate_to_hz,
)


def test_heart_rate_to_hz_one_case_matches_expected_value() -> None:
    """One-case test for BPM to Hz conversion."""
    assert heart_rate_to_hz(120.0) == 2.0


def test_build_time_axis_two_case_has_expected_shape_and_spacing() -> None:
    """Two-case test for two different sampling configurations."""
    time_axis_a = build_time_axis(sample_rate_hz=4.0, duration_seconds=1.0)
    time_axis_b = build_time_axis(sample_rate_hz=8.0, duration_seconds=1.0)

    assert np.allclose(time_axis_a, np.array([0.0, 0.25, 0.5, 0.75]))
    assert np.allclose(time_axis_b[:3], np.array([0.0, 0.125, 0.25]))


def test_generate_heartbeat_envelope_many_case_is_bounded() -> None:
    """Many-case test over multiple heart rates."""
    time_axis = build_time_axis(sample_rate_hz=100.0, duration_seconds=2.0)

    for heart_rate_bpm in [110.0, 130.0, 150.0]:
        envelope = generate_heartbeat_envelope(time_axis, heart_rate_bpm)
        assert envelope.shape == time_axis.shape
        assert np.min(envelope) >= 0.0
        assert np.max(envelope) <= 1.0


@pytest.mark.parametrize(
    ("sample_rate_hz", "duration_seconds"),
    [
        (0.0, 1.0),
        (2.0, 0.0),
        (1.0, 1.0),
    ],
)
def test_build_time_axis_branch_cases_raise_value_error(
    sample_rate_hz: float,
    duration_seconds: float,
) -> None:
    """Branch-coverage test for invalid time-axis inputs."""
    with pytest.raises(ValueError):
        build_time_axis(sample_rate_hz=sample_rate_hz, duration_seconds=duration_seconds)


@pytest.mark.parametrize(
    ("heart_rate_bpm", "pulse_width_seconds"),
    [
        (0.0, 0.12),
        (120.0, 0.0),
    ],
)
def test_generate_heartbeat_envelope_branch_cases_raise_value_error(
    heart_rate_bpm: float,
    pulse_width_seconds: float,
) -> None:
    """Statement and branch coverage for heartbeat envelope preconditions."""
    time_axis = build_time_axis(sample_rate_hz=100.0, duration_seconds=1.0)
    with pytest.raises(ValueError):
        generate_heartbeat_envelope(
            time_axis=time_axis,
            heart_rate_bpm=heart_rate_bpm,
            pulse_width_seconds=pulse_width_seconds,
        )
