"""Tests for Doppler-like signal synthesis.

Covers:
- generate_carrier_wave          (existing)
- modulate_carrier_with_envelope (existing)
- generate_heartbeat_envelope    (new)
- apply_amplitude_scaling        (new)
- add_baseline_noise             (new)
- DopplerSignalResult            (new datatype)
- generate_doppler_signal        (new high-level entry point)

Each group provides 1-case, 2-case, many-case, branch-coverage, and
statement-coverage tests per the project hard requirements.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from simulation.doppler_like import (
    DEFAULT_CARRIER_FREQUENCY_HZ,
    DEFAULT_SAMPLE_RATE_HZ,
    DopplerSignalResult,
    add_baseline_noise,
    apply_amplitude_scaling,
    generate_carrier_wave,
    generate_doppler_signal,
    generate_heartbeat_envelope,
    modulate_carrier_with_envelope,
)
from simulation.generator import generate_clean_signal_example
from simulation.heartbeat import build_time_axis
from utils.seed import build_rng


# ===================================================================
# generate_carrier_wave
# ===================================================================


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


# ===================================================================
# generate_heartbeat_envelope
# ===================================================================


def test_generate_heartbeat_envelope_one_case_shape_and_range() -> None:
    """One-case: output has same shape as time_axis and values in [0, 1]."""
    time_axis = build_time_axis(sample_rate_hz=1000.0, duration_seconds=2.0)
    envelope = generate_heartbeat_envelope(time_axis, heart_rate_bpm=120.0)

    assert envelope.shape == time_axis.shape
    assert envelope.min() >= 0.0
    assert envelope.max() <= 1.0 + 1e-12


def test_generate_heartbeat_envelope_two_case_different_rates() -> None:
    """Two-case: different heart rates produce different envelopes."""
    time_axis = build_time_axis(sample_rate_hz=1000.0, duration_seconds=2.0)
    env_slow = generate_heartbeat_envelope(time_axis, heart_rate_bpm=60.0)
    env_fast = generate_heartbeat_envelope(time_axis, heart_rate_bpm=180.0)

    assert not np.array_equal(env_slow, env_fast)


def test_generate_heartbeat_envelope_many_case_peak_normalised() -> None:
    """Many-case: envelope peak is always 1.0 for several heart rates."""
    time_axis = build_time_axis(sample_rate_hz=1000.0, duration_seconds=2.0)
    for bpm in [60.0, 120.0, 180.0, 220.0]:
        env = generate_heartbeat_envelope(time_axis, heart_rate_bpm=bpm)
        assert abs(env.max() - 1.0) < 1e-6


@pytest.mark.parametrize(
    "bad_kwarg",
    [
        dict(heart_rate_bpm=-1.0),
        dict(heart_rate_bpm=0.0),
        dict(pulse_width_seconds=-0.01),
        dict(pulse_width_seconds=0.0),
        dict(smooth_std_seconds=-0.01),
        dict(smooth_std_seconds=0.0),
    ],
)
def test_generate_heartbeat_envelope_branch_invalid_params_raise(
    bad_kwarg: dict,
) -> None:
    """Branch: invalid scalar params raise ValueError."""
    time_axis = build_time_axis(sample_rate_hz=1000.0, duration_seconds=1.0)
    good = dict(heart_rate_bpm=120.0)
    with pytest.raises(ValueError):
        generate_heartbeat_envelope(time_axis, **{**good, **bad_kwarg})


def test_generate_heartbeat_envelope_branch_wrong_ndim_raises() -> None:
    """Branch: 2-D time_axis raises ValueError."""
    with pytest.raises(ValueError, match="one-dimensional"):
        generate_heartbeat_envelope(np.ones((2, 3)), heart_rate_bpm=120.0)


def test_generate_heartbeat_envelope_branch_too_short_raises() -> None:
    """Branch: single-sample time_axis raises ValueError."""
    with pytest.raises(ValueError, match="at least two samples"):
        generate_heartbeat_envelope(np.array([0.0]), heart_rate_bpm=120.0)


# ===================================================================
# apply_amplitude_scaling
# ===================================================================


def test_apply_amplitude_scaling_one_case_identity() -> None:
    """One-case: scale=1.0 preserves the signal."""
    signal = np.array([0.5, -0.3, 0.8])
    result = apply_amplitude_scaling(signal, amplitude_scale=1.0)
    assert np.allclose(result, signal)


def test_apply_amplitude_scaling_two_case_zero_and_half() -> None:
    """Two-case: scale=0.0 zeros the signal, scale=0.5 halves it."""
    signal = np.array([2.0, -4.0, 6.0])

    zeroed = apply_amplitude_scaling(signal, amplitude_scale=0.0)
    assert np.allclose(zeroed, np.zeros(3))

    halved = apply_amplitude_scaling(signal, amplitude_scale=0.5)
    assert np.allclose(halved, np.array([1.0, -2.0, 3.0]))


def test_apply_amplitude_scaling_many_case_monotonic_energy() -> None:
    """Many-case: increasing scale monotonically increases RMS energy."""
    signal = np.array([1.0, -1.0, 0.5, -0.5, 0.0])
    scales = [0.0, 0.25, 0.5, 1.0, 2.0]
    energies = [np.sqrt(np.mean(apply_amplitude_scaling(signal, s) ** 2)) for s in scales]

    for i in range(len(energies) - 1):
        assert energies[i] <= energies[i + 1] + 1e-12


def test_apply_amplitude_scaling_branch_negative_raises() -> None:
    """Branch: negative amplitude_scale raises ValueError."""
    with pytest.raises(ValueError, match="amplitude_scale must be non-negative"):
        apply_amplitude_scaling(np.array([1.0, 2.0]), amplitude_scale=-0.1)


def test_apply_amplitude_scaling_branch_wrong_ndim_raises() -> None:
    """Branch: 2-D signal raises ValueError."""
    with pytest.raises(ValueError, match="one-dimensional"):
        apply_amplitude_scaling(np.ones((2, 3)), amplitude_scale=1.0)


def test_apply_amplitude_scaling_branch_too_short_raises() -> None:
    """Branch: single-sample signal raises ValueError."""
    with pytest.raises(ValueError, match="at least two samples"):
        apply_amplitude_scaling(np.array([1.0]), amplitude_scale=1.0)


# ===================================================================
# add_baseline_noise
# ===================================================================


def test_add_baseline_noise_one_case_zero_std_copies() -> None:
    """One-case: noise_std=0 returns an exact copy."""
    rng = build_rng(0)
    signal = np.array([1.0, 2.0, 3.0])
    result = add_baseline_noise(signal, noise_std=0.0, rng=rng)
    assert np.array_equal(result, signal)
    # Must be a copy, not the same object.
    assert result is not signal


def test_add_baseline_noise_two_case_deterministic_under_same_seed() -> None:
    """Two-case: same seed produces identical noise; different seed differs."""
    signal = np.zeros(1000)

    result_a = add_baseline_noise(signal, noise_std=1.0, rng=build_rng(42))
    result_b = add_baseline_noise(signal, noise_std=1.0, rng=build_rng(42))
    result_c = add_baseline_noise(signal, noise_std=1.0, rng=build_rng(99))

    assert np.array_equal(result_a, result_b)
    assert not np.array_equal(result_a, result_c)


def test_add_baseline_noise_many_case_std_proportional() -> None:
    """Many-case: empirical std of the noise grows with noise_std parameter."""
    signal = np.zeros(50_000)
    stds_requested = [0.01, 0.1, 0.5, 1.0]
    stds_measured = []

    for s in stds_requested:
        noisy = add_baseline_noise(signal, noise_std=s, rng=build_rng(0))
        stds_measured.append(float(np.std(noisy)))

    for i in range(len(stds_measured) - 1):
        assert stds_measured[i] < stds_measured[i + 1]

    # Each measured std should be within 10% of the requested std (law of large numbers).
    for req, meas in zip(stds_requested, stds_measured):
        assert abs(meas - req) / req < 0.10


def test_add_baseline_noise_branch_negative_std_raises() -> None:
    """Branch: negative noise_std raises ValueError."""
    with pytest.raises(ValueError, match="noise_std must be non-negative"):
        add_baseline_noise(np.array([1.0, 2.0]), noise_std=-1.0, rng=build_rng(0))


def test_add_baseline_noise_branch_wrong_ndim_raises() -> None:
    """Branch: 2-D signal raises ValueError."""
    with pytest.raises(ValueError, match="one-dimensional"):
        add_baseline_noise(np.ones((2, 3)), noise_std=0.1, rng=build_rng(0))


def test_add_baseline_noise_branch_too_short_raises() -> None:
    """Branch: single-sample signal raises ValueError."""
    with pytest.raises(ValueError, match="at least two samples"):
        add_baseline_noise(np.array([1.0]), noise_std=0.1, rng=build_rng(0))


# ===================================================================
# DopplerSignalResult (datatype)
# ===================================================================


def _make_result(**overrides: object) -> DopplerSignalResult:
    """Helper to build a minimal valid DopplerSignalResult with overrides."""
    defaults = dict(
        signal=np.array([1.0, 2.0, 3.0]),
        time_axis=np.array([0.0, 1.0, 2.0]),
        sample_rate_hz=48000.0,
        carrier_frequency_hz=18000.0,
        heart_rate_bpm=140.0,
        modulation_depth=0.35,
        amplitude_scale=1.0,
        noise_std=0.0,
        seed=0,
    )
    defaults.update(overrides)
    return DopplerSignalResult(**defaults)  # type: ignore[arg-type]


def test_doppler_signal_result_one_case_valid_construction() -> None:
    """One-case: a valid result can be constructed and fields are accessible."""
    result = _make_result()
    assert result.sample_count() == 3
    assert result.heart_rate_bpm == 140.0


def test_doppler_signal_result_two_case_metadata() -> None:
    """Two-case: metadata dict contains expected keys for two configurations."""
    r1 = _make_result(noise_std=0.0, seed=0)
    r2 = _make_result(noise_std=0.5, seed=42)

    m1 = r1.metadata()
    m2 = r2.metadata()

    assert m1["noise_std"] == 0.0 and m1["seed"] == 0
    assert m2["noise_std"] == 0.5 and m2["seed"] == 42
    # Both have the same keys.
    assert set(m1.keys()) == set(m2.keys())


def test_doppler_signal_result_many_case_invariants_hold() -> None:
    """Many-case: several valid constructions all pass invariants."""
    configs = [
        dict(heart_rate_bpm=60.0, amplitude_scale=0.0),
        dict(heart_rate_bpm=140.0, amplitude_scale=1.0),
        dict(heart_rate_bpm=220.0, amplitude_scale=3.0, noise_std=0.5),
    ]
    for cfg in configs:
        r = _make_result(**cfg)
        assert r.signal.ndim == 1
        assert r.signal.size >= 2


@pytest.mark.parametrize(
    "bad_field",
    [
        dict(sample_rate_hz=-1.0),
        dict(carrier_frequency_hz=0.0),
        dict(heart_rate_bpm=-10.0),
        dict(modulation_depth=1.5),
        dict(amplitude_scale=-0.01),
        dict(noise_std=-1.0),
        dict(seed=-1),
    ],
)
def test_doppler_signal_result_branch_invalid_fields_raise(
    bad_field: dict,
) -> None:
    """Branch: each invalid scalar field triggers ValueError."""
    with pytest.raises(ValueError):
        _make_result(**bad_field)


def test_doppler_signal_result_branch_shape_mismatch_raises() -> None:
    """Branch: mismatched signal/time_axis shapes raise ValueError."""
    with pytest.raises(ValueError):
        DopplerSignalResult(
            signal=np.array([1.0, 2.0, 3.0]),
            time_axis=np.array([0.0, 1.0]),
            sample_rate_hz=48000.0,
            carrier_frequency_hz=18000.0,
            heart_rate_bpm=140.0,
            modulation_depth=0.35,
            amplitude_scale=1.0,
            noise_std=0.0,
            seed=0,
        )


# ===================================================================
# generate_doppler_signal  (high-level entry point)
# ===================================================================


def test_generate_doppler_signal_one_case_shape_and_defaults() -> None:
    """One-case: output shape matches sample_rate * duration and defaults are 48kHz/18kHz."""
    result = generate_doppler_signal(
        heart_rate_bpm=140.0,
        duration_seconds=0.5,
    )
    expected_samples = int(math.floor(DEFAULT_SAMPLE_RATE_HZ * 0.5))

    assert result.signal.shape == (expected_samples,)
    assert result.time_axis.shape == (expected_samples,)
    assert result.sample_rate_hz == DEFAULT_SAMPLE_RATE_HZ
    assert result.carrier_frequency_hz == DEFAULT_CARRIER_FREQUENCY_HZ


def test_generate_doppler_signal_two_case_determinism_under_seed() -> None:
    """Two-case: identical seeds produce bit-identical signals; different seeds differ."""
    kwargs = dict(heart_rate_bpm=140.0, duration_seconds=0.5, noise_std=0.1)

    r1 = generate_doppler_signal(**kwargs, seed=7)
    r2 = generate_doppler_signal(**kwargs, seed=7)
    r3 = generate_doppler_signal(**kwargs, seed=99)

    assert np.array_equal(r1.signal, r2.signal)
    assert not np.array_equal(r1.signal, r3.signal)


def test_generate_doppler_signal_two_case_no_noise_is_deterministic() -> None:
    """Two-case: with noise_std=0, any two seeds produce the same signal."""
    kwargs = dict(heart_rate_bpm=140.0, duration_seconds=0.5, noise_std=0.0)

    r1 = generate_doppler_signal(**kwargs, seed=0)
    r2 = generate_doppler_signal(**kwargs, seed=999)

    assert np.array_equal(r1.signal, r2.signal)


def test_generate_doppler_signal_many_case_parameter_effects() -> None:
    """Many-case: verify that each parameter has a measurable effect on the output."""
    base = dict(
        heart_rate_bpm=140.0,
        duration_seconds=2.0,
        sample_rate_hz=4800.0,
        carrier_frequency_hz=1800.0,
        modulation_depth=0.35,
        amplitude_scale=1.0,
        noise_std=0.0,
        seed=0,
    )
    base_signal = generate_doppler_signal(**base).signal

    # Changing heart_rate_bpm changes the envelope → different signal.
    alt_hr = generate_doppler_signal(**{**base, "heart_rate_bpm": 180.0}).signal
    assert not np.array_equal(base_signal, alt_hr)

    # Changing carrier_frequency_hz changes the carrier → different signal.
    alt_cf = generate_doppler_signal(**{**base, "carrier_frequency_hz": 2000.0}).signal
    assert not np.array_equal(base_signal, alt_cf)

    # Changing modulation_depth changes envelope weighting.
    alt_md = generate_doppler_signal(**{**base, "modulation_depth": 0.8}).signal
    assert not np.array_equal(base_signal, alt_md)

    # amplitude_scale=0.5 halves the RMS.
    half_scale = generate_doppler_signal(**{**base, "amplitude_scale": 0.5}).signal
    rms_base = np.sqrt(np.mean(base_signal ** 2))
    rms_half = np.sqrt(np.mean(half_scale ** 2))
    assert abs(rms_half - 0.5 * rms_base) / (rms_base + 1e-12) < 0.01


def test_generate_doppler_signal_many_case_amplitude_scale_monotonic() -> None:
    """Many-case: RMS energy is monotonic in amplitude_scale."""
    scales = [0.0, 0.1, 0.5, 1.0, 2.0]
    energies = []
    for s in scales:
        sig = generate_doppler_signal(
            heart_rate_bpm=140.0, duration_seconds=0.1,
            sample_rate_hz=4800.0, carrier_frequency_hz=1800.0,
            amplitude_scale=s, noise_std=0.0,
        ).signal
        energies.append(np.sqrt(np.mean(sig ** 2)))

    for i in range(len(energies) - 1):
        assert energies[i] <= energies[i + 1] + 1e-12


def test_generate_doppler_signal_metadata_round_trips() -> None:
    """Statement: all generation parameters appear in metadata()."""
    result = generate_doppler_signal(
        heart_rate_bpm=120.0,
        duration_seconds=0.25,
        sample_rate_hz=4800.0,
        carrier_frequency_hz=1800.0,
        modulation_depth=0.5,
        amplitude_scale=0.8,
        noise_std=0.02,
        seed=77,
    )
    meta = result.metadata()

    assert meta["heart_rate_bpm"] == 120.0
    assert meta["sample_rate_hz"] == 4800.0
    assert meta["carrier_frequency_hz"] == 1800.0
    assert meta["modulation_depth"] == 0.5
    assert meta["amplitude_scale"] == 0.8
    assert meta["noise_std"] == 0.02
    assert meta["seed"] == 77


@pytest.mark.parametrize(
    "bad_kwarg",
    [
        dict(heart_rate_bpm=-1.0),
        dict(heart_rate_bpm=0.0),
        dict(duration_seconds=-1.0),
        dict(sample_rate_hz=0.0),
        dict(carrier_frequency_hz=-10.0),
        dict(modulation_depth=-0.01),
        dict(modulation_depth=1.01),
        dict(amplitude_scale=-0.5),
        dict(noise_std=-0.1),
        dict(seed=-1),
    ],
)
def test_generate_doppler_signal_branch_invalid_params_raise(
    bad_kwarg: dict,
) -> None:
    """Branch: every invalid parameter triggers ValueError at the top level."""
    good = dict(heart_rate_bpm=140.0, duration_seconds=0.5)
    with pytest.raises(ValueError):
        generate_doppler_signal(**{**good, **bad_kwarg})
