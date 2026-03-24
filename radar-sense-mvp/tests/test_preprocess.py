"""Tests for preprocessing helpers.

Covers:
- load_waveform_npy     (disk round-trip)
- normalize_signal      (zero-mean, unit-peak normalization)
- bandpass_filter       (Butterworth bandpass around carrier)
- extract_envelope      (analytic-signal envelope via Hilbert transform)
- rectify_signal        (pointwise absolute value)
- moving_average        (uniform-kernel smoothing)

Each group provides 1-case, 2-case, many-case, branch-coverage, and
statement-coverage tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from estimation.preprocess import (
    bandpass_filter,
    extract_envelope,
    load_waveform_npy,
    moving_average,
    normalize_signal,
    rectify_signal,
)


# ===================================================================
# load_waveform_npy
# ===================================================================


def test_load_waveform_npy_one_case_round_trip(tmp_path: Path) -> None:
    """One-case: save and reload a waveform."""
    waveform = np.array([1.0, -0.5, 0.3, 0.7], dtype=np.float64)
    npy_path = tmp_path / "sig.npy"
    np.save(npy_path, waveform)
    loaded = load_waveform_npy(npy_path)
    assert np.array_equal(loaded, waveform)
    assert loaded.dtype == np.float64


def test_load_waveform_npy_two_case_different_signals(tmp_path: Path) -> None:
    """Two-case: two different waveforms load to different arrays."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    pa = tmp_path / "a.npy"
    pb = tmp_path / "b.npy"
    np.save(pa, a)
    np.save(pb, b)
    assert not np.array_equal(load_waveform_npy(pa), load_waveform_npy(pb))


def test_load_waveform_npy_many_case_preserves_length(tmp_path: Path) -> None:
    """Many-case: loaded waveforms preserve their original lengths."""
    for n in [2, 10, 100]:
        w = np.random.default_rng(0).normal(size=n)
        p = tmp_path / f"w{n}.npy"
        np.save(p, w)
        assert load_waveform_npy(p).shape == (n,)


def test_load_waveform_npy_branch_rejects_2d(tmp_path: Path) -> None:
    """Branch: 2-D array on disk raises ValueError."""
    p = tmp_path / "bad.npy"
    np.save(p, np.ones((2, 3)))
    with pytest.raises(ValueError, match="one-dimensional"):
        load_waveform_npy(p)


def test_load_waveform_npy_branch_rejects_too_short(tmp_path: Path) -> None:
    """Branch: single-element waveform raises ValueError."""
    p = tmp_path / "short.npy"
    np.save(p, np.array([1.0]))
    with pytest.raises(ValueError, match="at least two"):
        load_waveform_npy(p)


# ===================================================================
# normalize_signal
# ===================================================================


def test_normalize_signal_one_case_zero_mean_unit_peak() -> None:
    """One-case: output has zero mean and unit peak."""
    signal = np.array([2.0, 4.0, 6.0, 8.0])
    normed = normalize_signal(signal)
    assert abs(float(np.mean(normed))) < 1e-12
    assert abs(float(np.max(np.abs(normed))) - 1.0) < 1e-12


def test_normalize_signal_two_case_different_scales() -> None:
    """Two-case: signals with different scales both normalize to unit peak."""
    for scale in [0.001, 1000.0]:
        signal = np.array([-1.0, 0.0, 1.0]) * scale
        normed = normalize_signal(signal)
        assert abs(float(np.max(np.abs(normed))) - 1.0) < 1e-12


def test_normalize_signal_many_case_shape_preserved() -> None:
    """Many-case: normalization preserves shape for various lengths."""
    rng = np.random.default_rng(42)
    for n in [2, 5, 50, 500]:
        signal = rng.normal(size=n)
        normed = normalize_signal(signal)
        assert normed.shape == signal.shape


def test_normalize_signal_branch_constant_signal() -> None:
    """Branch: constant signal (peak == 0 after centering) returns zeros."""
    signal = np.array([3.0, 3.0, 3.0, 3.0])
    normed = normalize_signal(signal)
    assert np.allclose(normed, np.zeros(4))


def test_normalize_signal_branch_rejects_wrong_ndim() -> None:
    """Branch: 2-D signal raises ValueError."""
    with pytest.raises(ValueError, match="one-dimensional"):
        normalize_signal(np.ones((2, 3)))


def test_normalize_signal_branch_rejects_too_short() -> None:
    """Branch: single-element signal raises ValueError."""
    with pytest.raises(ValueError, match="at least two"):
        normalize_signal(np.array([1.0]))


def test_normalize_signal_statement_deterministic() -> None:
    """Statement: normalization is deterministic (same input, same output)."""
    signal = np.array([1.0, -3.0, 2.0, 0.5])
    a = normalize_signal(signal)
    b = normalize_signal(signal)
    assert np.array_equal(a, b)


# ===================================================================
# bandpass_filter
# ===================================================================


def _make_mixed_signal(
    sample_rate_hz: float,
    duration_seconds: float,
    freq_target_hz: float,
    freq_reject_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_axis, signal) with two sine components."""
    t = np.arange(0, duration_seconds, 1.0 / sample_rate_hz)
    target = np.sin(2.0 * np.pi * freq_target_hz * t)
    reject = np.sin(2.0 * np.pi * freq_reject_hz * t)
    return t, target + reject


def test_bandpass_filter_one_case_passes_target_frequency() -> None:
    """One-case: a tone inside the passband is largely preserved."""
    fs = 1000.0
    _, mixed = _make_mixed_signal(fs, 1.0, freq_target_hz=100.0, freq_reject_hz=400.0)
    filtered = bandpass_filter(mixed, fs, low_hz=80.0, high_hz=120.0)
    # Energy of filtered signal should be dominated by the 100 Hz component.
    rms_filtered = float(np.sqrt(np.mean(filtered ** 2)))
    assert rms_filtered > 0.3  # 100 Hz tone has RMS ~0.707


def test_bandpass_filter_two_case_rejects_out_of_band() -> None:
    """Two-case: tones outside passband are attenuated relative to in-band."""
    fs = 1000.0
    t = np.arange(0, 1.0, 1.0 / fs)
    in_band = np.sin(2.0 * np.pi * 100.0 * t)
    out_band = np.sin(2.0 * np.pi * 400.0 * t)

    filt_in = bandpass_filter(in_band, fs, low_hz=80.0, high_hz=120.0)
    filt_out = bandpass_filter(out_band, fs, low_hz=80.0, high_hz=120.0)

    rms_in = float(np.sqrt(np.mean(filt_in ** 2)))
    rms_out = float(np.sqrt(np.mean(filt_out ** 2)))
    assert rms_in > 10.0 * rms_out  # in-band >> out-of-band


def test_bandpass_filter_many_case_shape_preserved() -> None:
    """Many-case: output shape matches input for various lengths."""
    fs = 1000.0
    rng = np.random.default_rng(7)
    for n in [100, 500, 1000]:
        signal = rng.normal(size=n)
        filtered = bandpass_filter(signal, fs, low_hz=80.0, high_hz=120.0)
        assert filtered.shape == signal.shape


def test_bandpass_filter_statement_deterministic() -> None:
    """Statement: filtering is deterministic (same input, same output)."""
    fs = 1000.0
    signal = np.random.default_rng(0).normal(size=500)
    a = bandpass_filter(signal, fs, low_hz=50.0, high_hz=150.0)
    b = bandpass_filter(signal, fs, low_hz=50.0, high_hz=150.0)
    assert np.array_equal(a, b)


def test_bandpass_filter_branch_rejects_wrong_ndim() -> None:
    """Branch: 2-D signal raises ValueError."""
    with pytest.raises(ValueError, match="one-dimensional"):
        bandpass_filter(np.ones((2, 3)), 1000.0, low_hz=80.0, high_hz=120.0)


def test_bandpass_filter_branch_rejects_too_short() -> None:
    """Branch: single-element signal raises ValueError."""
    with pytest.raises(ValueError, match="at least two"):
        bandpass_filter(np.array([1.0]), 1000.0, low_hz=80.0, high_hz=120.0)


def test_bandpass_filter_branch_rejects_non_positive_sample_rate() -> None:
    """Branch: non-positive sample rate raises ValueError."""
    with pytest.raises(ValueError, match="sample_rate_hz"):
        bandpass_filter(np.ones(100), 0.0, low_hz=80.0, high_hz=120.0)


def test_bandpass_filter_branch_rejects_non_positive_low_hz() -> None:
    """Branch: non-positive low_hz raises ValueError."""
    with pytest.raises(ValueError, match="low_hz"):
        bandpass_filter(np.ones(100), 1000.0, low_hz=0.0, high_hz=120.0)


def test_bandpass_filter_branch_rejects_low_ge_high() -> None:
    """Branch: low_hz >= high_hz raises ValueError."""
    with pytest.raises(ValueError, match="high_hz"):
        bandpass_filter(np.ones(100), 1000.0, low_hz=120.0, high_hz=80.0)


def test_bandpass_filter_branch_rejects_high_ge_nyquist() -> None:
    """Branch: high_hz >= Nyquist raises ValueError."""
    with pytest.raises(ValueError, match="high_hz must be less than Nyquist"):
        bandpass_filter(np.ones(100), 1000.0, low_hz=80.0, high_hz=500.0)


def test_bandpass_filter_branch_rejects_order_below_one() -> None:
    """Branch: order < 1 raises ValueError."""
    with pytest.raises(ValueError, match="order"):
        bandpass_filter(np.ones(100), 1000.0, low_hz=80.0, high_hz=120.0, order=0)


# ===================================================================
# extract_envelope
# ===================================================================


def test_extract_envelope_one_case_recovers_modulation() -> None:
    """One-case: envelope of an AM signal recovers the modulating waveform.

    A carrier at 200 Hz amplitude-modulated by a 5 Hz sinusoid.  The
    extracted envelope should oscillate at ~5 Hz with amplitude close to 1.
    """
    fs = 2000.0
    t = np.arange(0, 1.0, 1.0 / fs)
    modulator = 0.5 + 0.5 * np.cos(2.0 * np.pi * 5.0 * t)  # [0, 1]
    carrier = np.sin(2.0 * np.pi * 200.0 * t)
    am_signal = modulator * carrier

    envelope = extract_envelope(am_signal)
    # The envelope should closely match |modulator| (which is modulator
    # itself since it is non-negative).  Allow some edge-effect tolerance.
    mid = len(t) // 4
    end = 3 * len(t) // 4
    correlation = float(np.corrcoef(envelope[mid:end], modulator[mid:end])[0, 1])
    assert correlation > 0.95


def test_extract_envelope_two_case_non_negative() -> None:
    """Two-case: envelope is non-negative for both sine and noise signals."""
    t = np.linspace(0, 1.0, 500)
    sine = np.sin(2.0 * np.pi * 50.0 * t)
    noise = np.random.default_rng(0).normal(size=500)
    assert np.all(extract_envelope(sine) >= 0.0)
    assert np.all(extract_envelope(noise) >= 0.0)


def test_extract_envelope_many_case_shape_preserved() -> None:
    """Many-case: envelope preserves shape for various signal lengths."""
    rng = np.random.default_rng(7)
    for n in [10, 100, 1000]:
        signal = rng.normal(size=n)
        assert extract_envelope(signal).shape == (n,)


def test_extract_envelope_statement_deterministic() -> None:
    """Statement: envelope extraction is deterministic."""
    signal = np.random.default_rng(0).normal(size=200)
    a = extract_envelope(signal)
    b = extract_envelope(signal)
    assert np.array_equal(a, b)


def test_extract_envelope_branch_rejects_wrong_ndim() -> None:
    """Branch: 2-D signal raises ValueError."""
    with pytest.raises(ValueError, match="one-dimensional"):
        extract_envelope(np.ones((2, 3)))


def test_extract_envelope_branch_rejects_too_short() -> None:
    """Branch: single-element signal raises ValueError."""
    with pytest.raises(ValueError, match="at least two"):
        extract_envelope(np.array([1.0]))


# ===================================================================
# rectify_signal (legacy)
# ===================================================================


def test_rectify_signal_one_case_returns_non_negative_values() -> None:
    """One-case: rectification makes all values non-negative."""
    rectified = rectify_signal(np.array([-1.0, 0.0, 2.0]))
    assert np.allclose(rectified, np.array([1.0, 0.0, 2.0]))


def test_rectify_signal_two_case_shape_and_dtype() -> None:
    """Two-case: rectification preserves shape and outputs float64."""
    for signal in [np.array([-3.0, 1.0]), np.zeros(5)]:
        rectified = rectify_signal(signal)
        assert rectified.shape == signal.shape
        assert rectified.dtype == np.float64


def test_rectify_signal_branch_rejects_wrong_ndim() -> None:
    """Branch: 2-D signal raises ValueError."""
    with pytest.raises(ValueError, match="one-dimensional"):
        rectify_signal(np.ones((2, 3)))


# ===================================================================
# moving_average (legacy)
# ===================================================================


def test_moving_average_one_case_window_one_identity() -> None:
    """One-case: window_size 1 returns the same signal."""
    signal = np.array([1.0, 3.0, 5.0])
    assert np.allclose(moving_average(signal, window_size=1), signal)


def test_moving_average_two_case_has_expected_behavior() -> None:
    """Two-case: window 1 preserves, window 3 averages."""
    signal = np.array([0.0, 2.0, 0.0])
    smoothed_1 = moving_average(signal, window_size=1)
    smoothed_3 = moving_average(signal, window_size=3)
    assert np.allclose(smoothed_1, signal)
    assert np.allclose(smoothed_3, np.array([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]))


def test_moving_average_many_case_preserves_shape() -> None:
    """Many-case: output shape matches input for various window sizes."""
    signal = np.arange(20, dtype=np.float64)
    for w in [1, 3, 5, 10]:
        assert moving_average(signal, window_size=w).shape == signal.shape


def test_moving_average_branch_rejects_zero_window() -> None:
    """Branch: window_size 0 raises ValueError."""
    with pytest.raises(ValueError, match="window_size"):
        moving_average(np.array([1.0, 2.0, 3.0]), window_size=0)


def test_moving_average_branch_rejects_wrong_ndim() -> None:
    """Branch: 2-D signal raises ValueError."""
    with pytest.raises(ValueError, match="one-dimensional"):
        moving_average(np.ones((2, 3)), window_size=3)
