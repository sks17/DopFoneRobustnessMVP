"""Synthetic heartbeat envelope generation."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


# Spec:
# - General description: Construct a uniform time axis for a finite sampled signal.
# - Params: `sample_rate_hz`, positive samples per second; `duration_seconds`, positive signal duration.
# - Pre: `sample_rate_hz > 0` and `duration_seconds > 0`.
# - Post: Returns a float64 array `t` with length `floor(sample_rate_hz * duration_seconds)` and spacing `1 / sample_rate_hz`.
# - Mathematical definition: t_i = i / f_s for i in {0, ..., n - 1}, where n = floor(f_s * d).
def build_time_axis(sample_rate_hz: float, duration_seconds: float) -> FloatArray:
    """Return a uniform time axis for the requested sampling configuration."""
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive.")
    sample_count: int = int(math.floor(sample_rate_hz * duration_seconds))
    if sample_count < 2:
        raise ValueError("time axis must contain at least two samples.")
    return np.arange(sample_count, dtype=np.float64) / sample_rate_hz


# Spec:
# - General description: Convert a heart rate in beats per minute to beats per second.
# - Params: `heart_rate_bpm`, positive heart rate in beats per minute.
# - Pre: `heart_rate_bpm > 0`.
# - Post: Returns a positive frequency in hertz.
# - Mathematical definition: f = bpm / 60.
def heart_rate_to_hz(heart_rate_bpm: float) -> float:
    """Convert beats per minute to hertz."""
    if heart_rate_bpm <= 0.0:
        raise ValueError("heart_rate_bpm must be positive.")
    return heart_rate_bpm / 60.0


# Spec:
# - General description: Generate a non-negative heartbeat amplitude envelope with one pulse train.
# - Params: `time_axis`, one-dimensional sample times; `heart_rate_bpm`, positive label; `pulse_width_seconds`, positive pulse width.
# - Pre: `time_axis` is one-dimensional with at least two samples, `heart_rate_bpm > 0`, and `pulse_width_seconds > 0`.
# - Post: Returns a float64 array with the same shape as `time_axis`; all values are in [0, 1].
# - Mathematical definition: envelope(t) = 0.5 * (1 + sin(2 * pi * f * t)) raised pointwise to power p, where p = max(1, 0.25 / pulse_width_seconds).
def generate_heartbeat_envelope(
    time_axis: FloatArray,
    heart_rate_bpm: float,
    pulse_width_seconds: float = 0.12,
) -> FloatArray:
    """Return a smooth heartbeat-like envelope."""
    if time_axis.ndim != 1:
        raise ValueError("time_axis must be one-dimensional.")
    if time_axis.size < 2:
        raise ValueError("time_axis must contain at least two samples.")
    if pulse_width_seconds <= 0.0:
        raise ValueError("pulse_width_seconds must be positive.")
    heartbeat_hz: float = heart_rate_to_hz(heart_rate_bpm)
    pulse_sharpness: float = max(1.0, 0.25 / pulse_width_seconds)
    base_wave: FloatArray = 0.5 * (1.0 + np.sin(2.0 * np.pi * heartbeat_hz * time_axis))
    return np.power(base_wave, pulse_sharpness, dtype=np.float64)
