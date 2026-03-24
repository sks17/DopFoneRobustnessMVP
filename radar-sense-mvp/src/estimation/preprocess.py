"""Preprocessing helpers for simple heart-rate estimation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


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
