"""Tests for the simple MVP fetal-heart-rate estimator."""

from __future__ import annotations

import numpy as np
import pytest

from estimation.peak_estimator import (
    compute_peak_threshold,
    detect_peak_indices,
    estimate_bpm_from_intervals,
    estimate_heart_rate_bpm,
    peak_distance_seconds_to_samples,
    peak_indices_to_intervals_seconds,
    preprocess_waveform,
    scale_to_unit_interval,
    smoothing_seconds_to_window_size,
)
from simulation.generator import generate_clean_signal_example


def test_smoothing_seconds_to_window_size_one_case_matches_expected_value() -> None:
    """One-case test for smoothing-window conversion."""
    assert smoothing_seconds_to_window_size(100.0, 0.05) == 5


def test_peak_distance_seconds_to_samples_two_case_matches_expected_values() -> None:
    """Two-case test for minimum peak-distance conversion."""
    assert peak_distance_seconds_to_samples(100.0, 0.25) == 25
    assert peak_distance_seconds_to_samples(200.0, 0.30) == 60


def test_scale_to_unit_interval_many_case_maps_values_to_unit_interval() -> None:
    """Many-case test for unit-interval scaling."""
    for signal in [
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([-2.0, 0.0, 2.0], dtype=np.float64),
        np.array([5.0], dtype=np.float64),
    ]:
        scaled = scale_to_unit_interval(signal)
        assert scaled.shape == signal.shape
        assert np.all(scaled >= 0.0)
        assert np.all(scaled <= 1.0)


def test_preprocess_waveform_statement_case_constant_signal_maps_to_zero() -> None:
    """Statement-coverage test for constant-waveform preprocessing."""
    signal = np.ones(200, dtype=np.float64)
    processed = preprocess_waveform(signal=signal, sample_rate_hz=100.0)
    assert np.allclose(processed, np.zeros_like(processed))


def test_compute_peak_threshold_two_case_changes_with_ratio() -> None:
    """Two-case test for threshold-ratio behavior."""
    signal = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    low_ratio_threshold = compute_peak_threshold(signal, threshold_ratio=0.0)
    high_ratio_threshold = compute_peak_threshold(signal, threshold_ratio=1.0)
    assert low_ratio_threshold == pytest.approx(np.mean(signal))
    assert high_ratio_threshold == pytest.approx(np.max(signal))


def test_detect_peak_indices_one_case_finds_expected_peaks() -> None:
    """One-case test for local-maximum detection."""
    signal = np.array([0.0, 1.0, 0.0, 0.0, 2.0, 0.0], dtype=np.float64)
    peak_indices = detect_peak_indices(
        signal=signal,
        threshold=0.5,
        min_peak_distance_samples=1,
    )
    assert np.array_equal(peak_indices, np.array([1, 4], dtype=np.int64))


def test_peak_indices_to_intervals_seconds_two_case_matches_expected_values() -> None:
    """Two-case test for interval conversion."""
    peaks_a = np.array([10, 30], dtype=np.int64)
    peaks_b = np.array([10, 25, 55], dtype=np.int64)
    assert np.allclose(peak_indices_to_intervals_seconds(peaks_a, 10.0), np.array([2.0]))
    assert np.allclose(
        peak_indices_to_intervals_seconds(peaks_b, 10.0),
        np.array([1.5, 3.0]),
    )


def test_estimate_bpm_from_intervals_many_case_uses_median_interval() -> None:
    """Many-case test for median-based interval inversion."""
    intervals = np.array([0.5, 0.5, 0.4, 2.0], dtype=np.float64)
    estimated_bpm = estimate_bpm_from_intervals(intervals)
    assert estimated_bpm == pytest.approx(120.0)


def test_estimate_heart_rate_bpm_many_case_is_reasonably_close_on_clean_examples() -> None:
    """Many-case test on synthetic clean examples with a generous MVP tolerance."""
    tolerance_bpm = 15.0
    for target_bpm in [120.0, 135.0, 150.0]:
        example = generate_clean_signal_example(
            example_id=f"clean-{int(target_bpm)}",
            sample_rate_hz=500.0,
            duration_seconds=8.0,
            heart_rate_bpm=target_bpm,
            carrier_frequency_hz=18.0,
        )
        estimated_bpm = estimate_heart_rate_bpm(
            signal=example.signal,
            sample_rate_hz=example.sample_rate_hz,
        )
        assert abs(estimated_bpm - target_bpm) <= tolerance_bpm


def test_estimate_heart_rate_bpm_many_case_preserves_ordering_on_clean_examples() -> None:
    """Many-case test for monotonic ordering on clean examples."""
    estimates: list[float] = []
    for target_bpm in [110.0, 130.0, 150.0]:
        example = generate_clean_signal_example(
            example_id=f"order-{int(target_bpm)}",
            sample_rate_hz=500.0,
            duration_seconds=8.0,
            heart_rate_bpm=target_bpm,
            carrier_frequency_hz=18.0,
        )
        estimates.append(
            estimate_heart_rate_bpm(
                signal=example.signal,
                sample_rate_hz=example.sample_rate_hz,
            )
        )
    assert estimates[0] <= estimates[1]
    assert estimates[1] <= estimates[2]


@pytest.mark.parametrize(
    ("sample_rate_hz", "value"),
    [
        (0.0, 0.05),
        (100.0, 0.0),
    ],
)
def test_window_and_distance_conversion_branch_cases_raise_value_error(
    sample_rate_hz: float,
    value: float,
) -> None:
    """Branch-coverage test for conversion preconditions."""
    with pytest.raises(ValueError):
        smoothing_seconds_to_window_size(sample_rate_hz, value)
    with pytest.raises(ValueError):
        peak_distance_seconds_to_samples(sample_rate_hz, value)


def test_scale_to_unit_interval_branch_case_rejects_wrong_dimension() -> None:
    """Branch-coverage test for invalid unit-scaling input shape."""
    with pytest.raises(ValueError):
        scale_to_unit_interval(np.ones((2, 2), dtype=np.float64))


def test_compute_peak_threshold_branch_case_rejects_invalid_ratio() -> None:
    """Branch-coverage test for invalid threshold ratio."""
    with pytest.raises(ValueError):
        compute_peak_threshold(np.array([0.0, 1.0], dtype=np.float64), threshold_ratio=1.5)


def test_detect_peak_indices_branch_case_rejects_invalid_spacing() -> None:
    """Branch-coverage test for invalid minimum spacing."""
    with pytest.raises(ValueError):
        detect_peak_indices(
            signal=np.array([0.0, 1.0, 0.0], dtype=np.float64),
            threshold=0.5,
            min_peak_distance_samples=0,
        )


def test_peak_indices_to_intervals_seconds_branch_case_rejects_too_few_peaks() -> None:
    """Branch-coverage test for interval conversion with too few peaks."""
    with pytest.raises(ValueError):
        peak_indices_to_intervals_seconds(np.array([10], dtype=np.int64), sample_rate_hz=100.0)


def test_estimate_bpm_from_intervals_branch_case_rejects_non_positive_interval() -> None:
    """Branch-coverage test for invalid intervals."""
    with pytest.raises(ValueError):
        estimate_bpm_from_intervals(np.array([0.5, 0.0], dtype=np.float64))


def test_estimate_heart_rate_bpm_statement_case_returns_zero_when_too_few_peaks() -> None:
    """Statement-coverage test for the early return when fewer than two peaks are detected."""
    flat_signal = np.ones(1000, dtype=np.float64)
    assert estimate_heart_rate_bpm(flat_signal, sample_rate_hz=200.0) == 0.0


def test_estimate_heart_rate_bpm_branch_case_rejects_non_positive_sample_rate() -> None:
    """Branch-coverage test for invalid sample rate."""
    with pytest.raises(ValueError):
        estimate_heart_rate_bpm(np.ones(100, dtype=np.float64), sample_rate_hz=0.0)
