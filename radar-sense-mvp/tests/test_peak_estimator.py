"""Tests for preprocessing and peak-based heart-rate estimation."""

from __future__ import annotations

import numpy as np
import pytest

from estimation.peak_estimator import count_threshold_peaks, estimate_heart_rate_bpm
from estimation.preprocess import moving_average, rectify_signal
from simulation.generator import generate_clean_signal_example


def test_rectify_signal_one_case_returns_non_negative_values() -> None:
    """One-case test for rectification."""
    rectified = rectify_signal(np.array([-1.0, 0.0, 2.0]))
    assert np.allclose(rectified, np.array([1.0, 0.0, 2.0]))


def test_moving_average_two_case_has_expected_behavior() -> None:
    """Two-case test for two window sizes."""
    signal = np.array([0.0, 2.0, 0.0])

    smoothed_window_1 = moving_average(signal, window_size=1)
    smoothed_window_3 = moving_average(signal, window_size=3)

    assert np.allclose(smoothed_window_1, signal)
    assert np.allclose(smoothed_window_3, np.array([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]))


def test_estimate_heart_rate_bpm_many_case_returns_positive_estimates() -> None:
    """Many-case test across several generated clean examples."""
    examples = [
        generate_clean_signal_example("p1", 200.0, 4.0, 120.0, 18.0),
        generate_clean_signal_example("p2", 200.0, 4.0, 135.0, 18.0),
        generate_clean_signal_example("p3", 200.0, 4.0, 150.0, 18.0),
    ]

    for example in examples:
        estimate = estimate_heart_rate_bpm(example.signal, example.sample_rate_hz)
        assert estimate >= 0.0


def test_count_threshold_peaks_branch_case_rejects_short_signal() -> None:
    """Branch-coverage test for a too-short signal."""
    with pytest.raises(ValueError):
        count_threshold_peaks(np.array([1.0, 2.0]), threshold=1.0)


def test_moving_average_statement_case_rejects_zero_window() -> None:
    """Statement-coverage test for moving-average precondition failure."""
    with pytest.raises(ValueError):
        moving_average(np.array([1.0, 2.0, 3.0]), window_size=0)
