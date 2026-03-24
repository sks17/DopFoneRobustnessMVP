"""Simple fetal-heart-rate estimator for synthetic signals."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from estimation.preprocess import moving_average, rectify_signal


FloatArray = npt.NDArray[np.float64]


# Spec:
# - General description: Count strict local maxima above a threshold.
# - Params: `signal`, one-dimensional waveform; `threshold`, scalar threshold.
# - Pre: `signal` is one-dimensional with at least three samples.
# - Post: Returns the number of indices i such that signal[i] exceeds neighbors and threshold.
# - Mathematical definition: count = |{i : 1 <= i <= n-2, x_i > x_{i-1}, x_i >= x_{i+1}, x_i >= threshold}|.
def count_threshold_peaks(signal: FloatArray, threshold: float) -> int:
    """Return the number of local peaks above a threshold."""
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if signal.size < 3:
        raise ValueError("signal must contain at least three samples.")
    peak_count = 0
    for index in range(1, signal.size - 1):
        if (
            signal[index] > signal[index - 1]
            and signal[index] >= signal[index + 1]
            and signal[index] >= threshold
        ):
            peak_count += 1
    return peak_count


# Spec:
# - General description: Estimate heart rate from a waveform using rectification, smoothing, and peak counting.
# - Params: `signal`, one-dimensional waveform; `sample_rate_hz`, positive sample rate.
# - Pre: `signal` is one-dimensional with at least three samples and `sample_rate_hz > 0`.
# - Post: Returns a non-negative heart-rate estimate in beats per minute.
# - Mathematical definition: bpm_hat = 60 * peak_count / duration, where peaks are counted on the smoothed rectified signal.
def estimate_heart_rate_bpm(signal: FloatArray, sample_rate_hz: float) -> float:
    """Estimate heart rate in beats per minute from the input signal."""
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if signal.size < 3:
        raise ValueError("signal must contain at least three samples.")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    rectified_signal = rectify_signal(signal)
    smoothing_window_size = max(1, int(round(sample_rate_hz * 0.12)))
    smoothed_signal = moving_average(rectified_signal, window_size=smoothing_window_size)
    threshold = float(np.mean(smoothed_signal))
    peak_count = count_threshold_peaks(smoothed_signal, threshold=threshold)
    duration_seconds = signal.size / sample_rate_hz
    if peak_count == 0:
        return 0.0
    return 60.0 * peak_count / duration_seconds
