"""Simple, interpretable MVP fetal-heart-rate estimator."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from estimation.preprocess import extract_envelope, moving_average


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

DEFAULT_SMOOTHING_SECONDS: float = 0.08
DEFAULT_MIN_PEAK_DISTANCE_SECONDS: float = 0.30
DEFAULT_THRESHOLD_RATIO: float = 0.35


# Spec:
# - General description: Convert a smoothing duration in seconds to a moving-average window size in samples.
# - Params: `sample_rate_hz`, positive sample rate; `smoothing_seconds`, positive smoothing duration.
# - Pre: `sample_rate_hz > 0` and `smoothing_seconds > 0`.
# - Post: Returns an integer window size greater than or equal to 1.
# - Mathematical definition: window_size = max(1, round(sample_rate_hz * smoothing_seconds)).
def smoothing_seconds_to_window_size(
    sample_rate_hz: float,
    smoothing_seconds: float,
) -> int:
    """Return a moving-average window size in samples."""
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    if smoothing_seconds <= 0.0:
        raise ValueError("smoothing_seconds must be positive.")
    return max(1, int(round(sample_rate_hz * smoothing_seconds)))


# Spec:
# - General description: Convert a minimum peak spacing in seconds to a minimum spacing in samples.
# - Params: `sample_rate_hz`, positive sample rate; `min_peak_distance_seconds`, positive minimum peak spacing.
# - Pre: `sample_rate_hz > 0` and `min_peak_distance_seconds > 0`.
# - Post: Returns an integer minimum spacing greater than or equal to 1.
# - Mathematical definition: min_peak_distance_samples = max(1, round(sample_rate_hz * min_peak_distance_seconds)).
def peak_distance_seconds_to_samples(
    sample_rate_hz: float,
    min_peak_distance_seconds: float,
) -> int:
    """Return a minimum peak spacing in samples."""
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    if min_peak_distance_seconds <= 0.0:
        raise ValueError("min_peak_distance_seconds must be positive.")
    return max(1, int(round(sample_rate_hz * min_peak_distance_seconds)))


# Spec:
# - General description: Scale a waveform to the unit interval [0, 1].
# - Params: `signal`, one-dimensional waveform with at least one sample.
# - Pre: `signal` is one-dimensional and `signal.size >= 1`.
# - Post: Returns a float64 array with the same shape and values in [0, 1]; constant signals map to zeros.
# - Mathematical definition: y = (x - min(x)) / (max(x) - min(x)) when max(x) > min(x), else y = 0.
def scale_to_unit_interval(signal: FloatArray) -> FloatArray:
    """Return a unit-interval-scaled copy of the input signal."""
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if signal.size < 1:
        raise ValueError("signal must contain at least one sample.")
    minimum_value = float(np.min(signal))
    maximum_value = float(np.max(signal))
    if maximum_value == minimum_value:
        return np.zeros_like(signal, dtype=np.float64)
    return np.asarray(
        (np.asarray(signal, dtype=np.float64) - minimum_value)
        / (maximum_value - minimum_value),
        dtype=np.float64,
    )


# Spec:
# - General description: Extract and smooth a heartbeat-related amplitude envelope from a waveform.
# - Params: `signal`, one-dimensional waveform; `sample_rate_hz`, positive sample rate.
# - Pre: `signal` is one-dimensional with at least two samples and `sample_rate_hz > 0`.
# - Post: Returns a float64 array with the same shape as `signal` and values in [0, 1]; constant waveforms map to zeros.
# - Mathematical definition: processed = scale_to_unit_interval(moving_average(extract_envelope(signal), k)).
def preprocess_waveform(signal: FloatArray, sample_rate_hz: float) -> FloatArray:
    """Return a smoothed, unit-scaled heartbeat envelope."""
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if signal.size < 2:
        raise ValueError("signal must contain at least two samples.")
    if np.allclose(signal, signal[0]):
        return np.zeros_like(signal, dtype=np.float64)
    window_size = smoothing_seconds_to_window_size(
        sample_rate_hz=sample_rate_hz,
        smoothing_seconds=DEFAULT_SMOOTHING_SECONDS,
    )
    envelope = extract_envelope(signal)
    smoothed_envelope = moving_average(envelope, window_size=window_size)
    return scale_to_unit_interval(smoothed_envelope)


# Spec:
# - General description: Compute a scalar peak-detection threshold from a preprocessed envelope.
# - Params: `signal`, one-dimensional unit-scaled waveform; `threshold_ratio`, ratio in [0, 1].
# - Pre: `signal` is one-dimensional with at least one sample and `0 <= threshold_ratio <= 1`.
# - Post: Returns a float threshold in [0, 1].
# - Mathematical definition: threshold = mean(signal) + threshold_ratio * (max(signal) - mean(signal)).
def compute_peak_threshold(signal: FloatArray, threshold_ratio: float) -> float:
    """Return a peak-detection threshold derived from the signal mean and max."""
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if signal.size < 1:
        raise ValueError("signal must contain at least one sample.")
    if not 0.0 <= threshold_ratio <= 1.0:
        raise ValueError("threshold_ratio must lie in [0, 1].")
    mean_value = float(np.mean(signal))
    max_value = float(np.max(signal))
    return mean_value + threshold_ratio * (max_value - mean_value)


# Spec:
# - General description: Detect local maxima above a threshold with a minimum sample spacing.
# - Params: `signal`, one-dimensional waveform; `threshold`, scalar threshold; `min_peak_distance_samples`, integer spacing >= 1.
# - Pre: `signal` is one-dimensional with at least three samples and `min_peak_distance_samples >= 1`.
# - Post: Returns a sorted one-dimensional integer array of accepted peak indices.
# - Mathematical definition: P = {i : x_i > x_{i-1}, x_i >= x_{i+1}, x_i >= threshold, i - last_peak >= min_peak_distance_samples}.
def detect_peak_indices(
    signal: FloatArray,
    threshold: float,
    min_peak_distance_samples: int,
) -> IntArray:
    """Return sorted local-maximum indices that satisfy threshold and spacing constraints."""
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if signal.size < 3:
        raise ValueError("signal must contain at least three samples.")
    if min_peak_distance_samples < 1:
        raise ValueError("min_peak_distance_samples must be at least 1.")
    detected_peaks: list[int] = []
    last_peak_index = -(min_peak_distance_samples + 1)
    for index in range(1, signal.size - 1):
        is_local_maximum = signal[index] > signal[index - 1] and signal[index] >= signal[index + 1]
        is_far_enough = (index - last_peak_index) >= min_peak_distance_samples
        if is_local_maximum and signal[index] >= threshold and is_far_enough:
            detected_peaks.append(index)
            last_peak_index = index
    return np.asarray(detected_peaks, dtype=np.int64)


# Spec:
# - General description: Convert sorted peak sample indices to inter-peak intervals in seconds.
# - Params: `peak_indices`, one-dimensional integer array; `sample_rate_hz`, positive sample rate.
# - Pre: `peak_indices` is one-dimensional with at least two elements and `sample_rate_hz > 0`.
# - Post: Returns a float64 array of length `peak_indices.size - 1`.
# - Mathematical definition: intervals_j = (peak_indices_{j+1} - peak_indices_j) / sample_rate_hz.
def peak_indices_to_intervals_seconds(
    peak_indices: IntArray,
    sample_rate_hz: float,
) -> FloatArray:
    """Return inter-peak intervals in seconds."""
    if peak_indices.ndim != 1:
        raise ValueError("peak_indices must be one-dimensional.")
    if peak_indices.size < 2:
        raise ValueError("peak_indices must contain at least two peaks.")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    return np.diff(peak_indices).astype(np.float64) / sample_rate_hz


# Spec:
# - General description: Estimate BPM from positive inter-peak intervals using the median interval.
# - Params: `intervals_seconds`, one-dimensional array of positive intervals in seconds.
# - Pre: `intervals_seconds` is one-dimensional, non-empty, and all values are positive.
# - Post: Returns a positive BPM estimate.
# - Mathematical definition: bpm = 60 / median(intervals_seconds).
def estimate_bpm_from_intervals(intervals_seconds: FloatArray) -> float:
    """Return BPM from the median inter-peak interval."""
    if intervals_seconds.ndim != 1:
        raise ValueError("intervals_seconds must be one-dimensional.")
    if intervals_seconds.size < 1:
        raise ValueError("intervals_seconds must be non-empty.")
    if np.any(intervals_seconds <= 0.0):
        raise ValueError("intervals_seconds must contain only positive values.")
    return 60.0 / float(np.median(intervals_seconds))


# Spec:
# - General description: Estimate fetal heart rate from a waveform using preprocessing, peak detection, and interval inversion.
# - Params: `signal`, one-dimensional waveform; `sample_rate_hz`, positive sample rate.
# - Pre: `signal` is one-dimensional with at least three samples and `sample_rate_hz > 0`.
# - Post: Returns a non-negative BPM estimate; returns 0.0 when fewer than two peaks are detected.
# - Mathematical definition: bpm_hat = 60 / median(diff(peaks) / sample_rate_hz), where peaks are detected on the preprocessed envelope.
def estimate_heart_rate_bpm(signal: FloatArray, sample_rate_hz: float) -> float:
    """Return a simple heart-rate estimate in beats per minute."""
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if signal.size < 3:
        raise ValueError("signal must contain at least three samples.")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    processed_signal = preprocess_waveform(signal=signal, sample_rate_hz=sample_rate_hz)
    threshold = compute_peak_threshold(
        signal=processed_signal,
        threshold_ratio=DEFAULT_THRESHOLD_RATIO,
    )
    min_peak_distance_samples = peak_distance_seconds_to_samples(
        sample_rate_hz=sample_rate_hz,
        min_peak_distance_seconds=DEFAULT_MIN_PEAK_DISTANCE_SECONDS,
    )
    peak_indices = detect_peak_indices(
        signal=processed_signal,
        threshold=threshold,
        min_peak_distance_samples=min_peak_distance_samples,
    )
    if peak_indices.size < 2:
        return 0.0
    intervals_seconds = peak_indices_to_intervals_seconds(
        peak_indices=peak_indices,
        sample_rate_hz=sample_rate_hz,
    )
    return estimate_bpm_from_intervals(intervals_seconds)
