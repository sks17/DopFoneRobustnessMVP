"""Tests for heartbeat timing and pulse-train helpers."""

from __future__ import annotations

import numpy as np
import pytest

from simulation.heartbeat import (
    add_physiological_jitter,
    bpm_to_seconds_per_beat,
    build_time_axis,
    generate_beat_timestamps,
    generate_pulse_train,
)


def test_bpm_to_seconds_per_beat_one_case_matches_expected_value() -> None:
    """One-case test for BPM to seconds-per-beat conversion."""
    assert bpm_to_seconds_per_beat(120.0) == 0.5


def test_generate_beat_timestamps_two_case_has_expected_values() -> None:
    """Two-case test for two different heart-rate settings."""
    timestamps_a = generate_beat_timestamps(target_bpm=60.0, duration_seconds=3.0)
    timestamps_b = generate_beat_timestamps(target_bpm=120.0, duration_seconds=2.0)

    assert np.allclose(timestamps_a, np.array([0.0, 1.0, 2.0]))
    assert np.allclose(timestamps_b, np.array([0.0, 0.5, 1.0, 1.5]))


def test_generate_pulse_train_many_case_has_binary_output() -> None:
    """Many-case test across several pulse-width choices."""
    time_axis = build_time_axis(sample_rate_hz=10.0, duration_seconds=1.0)
    beat_timestamps = np.array([0.0, 0.5], dtype=np.float64)

    for pulse_width_seconds in [0.05, 0.1, 0.2]:
        pulse_train = generate_pulse_train(
            time_axis=time_axis,
            beat_timestamps=beat_timestamps,
            pulse_width_seconds=pulse_width_seconds,
        )
        assert pulse_train.shape == time_axis.shape
        assert set(np.unique(pulse_train)).issubset({0.0, 1.0})


def test_add_physiological_jitter_statement_case_zero_jitter_returns_copy() -> None:
    """Statement-coverage test for the zero-jitter early return."""
    beat_timestamps = np.array([0.0, 0.5, 1.0], dtype=np.float64)

    jittered_timestamps = add_physiological_jitter(
        beat_timestamps=beat_timestamps,
        jitter_std_seconds=0.0,
        duration_seconds=2.0,
        rng=np.random.default_rng(7),
    )

    assert np.allclose(jittered_timestamps, beat_timestamps)
    assert jittered_timestamps is not beat_timestamps


def test_add_physiological_jitter_many_case_keeps_sorted_in_range() -> None:
    """Many-case test across several random seeds."""
    beat_timestamps = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)

    for seed in [1, 2, 3]:
        jittered_timestamps = add_physiological_jitter(
            beat_timestamps=beat_timestamps,
            jitter_std_seconds=0.02,
            duration_seconds=2.0,
            rng=np.random.default_rng(seed),
        )
        assert jittered_timestamps.shape == beat_timestamps.shape
        assert np.all(np.diff(jittered_timestamps) >= 0.0)
        assert np.all(jittered_timestamps >= 0.0)
        assert np.all(jittered_timestamps < 2.0)


@pytest.mark.parametrize(
    ("target_bpm", "duration_seconds"),
    [
        (0.0, 2.0),
        (120.0, 0.0),
    ],
)
def test_generate_beat_timestamps_branch_cases_raise_value_error(
    target_bpm: float,
    duration_seconds: float,
) -> None:
    """Branch-coverage test for invalid beat-timestamp inputs."""
    with pytest.raises(ValueError):
        generate_beat_timestamps(
            target_bpm=target_bpm,
            duration_seconds=duration_seconds,
        )


@pytest.mark.parametrize(
    ("jitter_std_seconds", "duration_seconds"),
    [
        (-0.01, 2.0),
        (0.01, 0.0),
    ],
)
def test_add_physiological_jitter_branch_cases_raise_value_error(
    jitter_std_seconds: float,
    duration_seconds: float,
) -> None:
    """Branch-coverage test for invalid jitter inputs."""
    with pytest.raises(ValueError):
        add_physiological_jitter(
            beat_timestamps=np.array([0.0, 0.5], dtype=np.float64),
            jitter_std_seconds=jitter_std_seconds,
            duration_seconds=duration_seconds,
            rng=np.random.default_rng(7),
        )


def test_generate_pulse_train_branch_case_rejects_non_positive_pulse_width() -> None:
    """Branch-coverage test for invalid pulse width."""
    with pytest.raises(ValueError):
        generate_pulse_train(
            time_axis=build_time_axis(sample_rate_hz=10.0, duration_seconds=1.0),
            beat_timestamps=np.array([0.0], dtype=np.float64),
            pulse_width_seconds=0.0,
        )
