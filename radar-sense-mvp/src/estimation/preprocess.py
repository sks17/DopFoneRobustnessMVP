"""Preprocessing helpers for simple heart-rate estimation.

Provides a composable pipeline for DopFone-like Doppler signals:

1. **load_waveform_npy** — read a persisted ``.npy`` waveform from disk.
2. **normalize_signal** — zero-mean, unit-peak normalization.
3. **bandpass_filter** — Butterworth bandpass around the carrier frequency.
4. **extract_envelope** — analytic-signal envelope via Hilbert transform.
5. **rectify_signal** — pointwise absolute value (legacy helper).
6. **moving_average** — uniform-kernel smoothing (legacy helper).

All functions operate on one-dimensional float64 arrays and use only
NumPy / SciPy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, hilbert, sosfiltfilt


FloatArray = npt.NDArray[np.float64]


# ------------------------------------------------------------------
# load_waveform_npy
# ------------------------------------------------------------------

# Spec:
# - General description: Load a one-dimensional float64 waveform from a
#   ``.npy`` file.
# - Params: `path`, path to a ``.npy`` file.
# - Pre: `path` exists and contains a one-dimensional array with at least
#   two samples.
# - Post: Returns a float64 array.
# - Mathematical definition: Inverse of ``np.save``.
def load_waveform_npy(path: str | Path) -> FloatArray:
    """Load a waveform array from a .npy file."""
    waveform = np.load(Path(path))
    waveform = np.asarray(waveform, dtype=np.float64)
    if waveform.ndim != 1:
        raise ValueError("Loaded waveform must be one-dimensional.")
    if waveform.size < 2:
        raise ValueError("Loaded waveform must contain at least two samples.")
    return waveform


# ------------------------------------------------------------------
# normalize_signal
# ------------------------------------------------------------------

# Spec:
# - General description: Normalize a signal to zero mean and unit peak
#   absolute value.  If the signal is constant (peak amplitude is zero after
#   mean removal), it is returned as all zeros.
# - Params: `signal`, one-dimensional waveform.
# - Pre: `signal` is one-dimensional with at least two samples.
# - Post: Returns a float64 array with the same shape, zero mean, and
#   max(|y|) == 1 (or all zeros if the input is constant).
# - Mathematical definition: y = (x - mean(x)) / max(|x - mean(x)|).
def normalize_signal(signal: FloatArray) -> FloatArray:
    """Return a zero-mean, unit-peak-normalized copy of the signal."""
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if signal.size < 2:
        raise ValueError("signal must contain at least two samples.")
    centered = np.asarray(signal, dtype=np.float64) - np.mean(signal)
    peak = float(np.max(np.abs(centered)))
    if peak == 0.0:
        return np.zeros_like(centered)
    return centered / peak


# ------------------------------------------------------------------
# bandpass_filter
# ------------------------------------------------------------------

# Spec:
# - General description: Apply a zero-phase Butterworth bandpass filter to
#   isolate frequencies around the carrier.  Uses second-order sections
#   (``sosfiltfilt``) for numerical stability.
# - Params: `signal`, one-dimensional waveform; `sample_rate_hz`, positive
#   sample rate; `low_hz`, lower cutoff frequency; `high_hz`, upper cutoff
#   frequency; `order`, filter order (default 4).
# - Pre: `signal` is one-dimensional with at least two samples,
#   `sample_rate_hz > 0`, `0 < low_hz < high_hz < sample_rate_hz / 2`,
#   and `order >= 1`.
# - Post: Returns a float64 array with the same shape as `signal`.
# - Mathematical definition: y = sosfiltfilt(sos, x) where sos encodes a
#   Butterworth bandpass of order `order` with critical frequencies
#   [low_hz, high_hz] relative to the Nyquist rate.
def bandpass_filter(
    signal: FloatArray,
    sample_rate_hz: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> FloatArray:
    """Apply a zero-phase Butterworth bandpass filter."""
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if signal.size < 2:
        raise ValueError("signal must contain at least two samples.")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    nyquist = sample_rate_hz / 2.0
    if low_hz <= 0.0:
        raise ValueError("low_hz must be positive.")
    if high_hz <= low_hz:
        raise ValueError("high_hz must be greater than low_hz.")
    if high_hz >= nyquist:
        raise ValueError("high_hz must be less than Nyquist (sample_rate_hz / 2).")
    if order < 1:
        raise ValueError("order must be at least 1.")
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=sample_rate_hz, output="sos")
    filtered = sosfiltfilt(sos, np.asarray(signal, dtype=np.float64))
    return np.asarray(filtered, dtype=np.float64)


# ------------------------------------------------------------------
# extract_envelope
# ------------------------------------------------------------------

# Spec:
# - General description: Extract the amplitude envelope of a signal using the
#   analytic signal (Hilbert transform).  The envelope captures the
#   instantaneous amplitude modulation — for a carrier modulated by a
#   heartbeat, this recovers the heartbeat waveform.
# - Params: `signal`, one-dimensional waveform.
# - Pre: `signal` is one-dimensional with at least two samples.
# - Post: Returns a non-negative float64 array with the same shape as
#   `signal`.
# - Mathematical definition: envelope(x) = |x + j * H(x)|, where H is the
#   Hilbert transform.
def extract_envelope(signal: FloatArray) -> FloatArray:
    """Return the amplitude envelope via the analytic signal."""
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if signal.size < 2:
        raise ValueError("signal must contain at least two samples.")
    analytic = hilbert(np.asarray(signal, dtype=np.float64))
    envelope = np.abs(analytic)
    return np.asarray(envelope, dtype=np.float64)


# ------------------------------------------------------------------
# Legacy helpers (retained for backward compatibility with peak_estimator)
# ------------------------------------------------------------------

# Spec:
# - General description: Take the pointwise absolute value of a waveform.
# - Params: `signal`, one-dimensional waveform.
# - Pre: `signal` is one-dimensional.
# - Post: Returns a float64 array with the same shape and non-negative values.
# - Mathematical definition: y_i = |x_i|.
def rectify_signal(signal: FloatArray) -> FloatArray:
    """Return the rectified signal."""
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    return np.abs(np.asarray(signal, dtype=np.float64))


# Spec:
# - General description: Smooth a waveform with a moving-average window.
# - Params: `signal`, one-dimensional waveform; `window_size`, positive integer.
# - Pre: `signal` is one-dimensional and `window_size >= 1`.
# - Post: Returns a float64 array with the same shape as `signal`.
# - Mathematical definition: y_i = (1 / k) * sum_{j=0}^{k-1} x_{i-j} using convolution with a length-k uniform kernel.
def moving_average(signal: FloatArray, window_size: int) -> FloatArray:
    """Return a moving-average smoothed copy of the signal."""
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    return np.convolve(np.asarray(signal, dtype=np.float64), kernel, mode="same")
