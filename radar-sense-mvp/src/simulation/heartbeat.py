"""Small heartbeat-timing and pulse-train helpers."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


# Spec:
# - General description: Convert a target heart rate in beats per minute to seconds per beat.
# - Params: `target_bpm`, positive target heart rate in beats per minute.
# - Pre: `target_bpm > 0`.
# - Post: Returns a positive beat interval in seconds.
# - Mathematical definition: interval_seconds = 60 / target_bpm.
def bpm_to_seconds_per_beat(target_bpm: float) -> float:
    """Return the nominal beat interval in seconds."""
    if target_bpm <= 0.0:
        raise ValueError("target_bpm must be positive.")
    return 60.0 / target_bpm


# Spec:
# - General description: Generate regularly spaced beat timestamps over a finite duration.
# - Params: `target_bpm`, positive target heart rate in beats per minute; `duration_seconds`, positive total duration.
# - Pre: `target_bpm > 0` and `duration_seconds > 0`.
# - Post: Returns a one-dimensional float64 array of beat timestamps `t` such that each value lies in [0, duration_seconds).
# - Mathematical definition: t_k = k * (60 / target_bpm) for all k where t_k < duration_seconds.
def generate_beat_timestamps(target_bpm: float, duration_seconds: float) -> FloatArray:
    """Return nominal beat timestamps for the requested rate and duration."""
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive.")
    seconds_per_beat = bpm_to_seconds_per_beat(target_bpm)
    beat_count = int(math.floor(duration_seconds / seconds_per_beat))
    return np.arange(beat_count, dtype=np.float64) * seconds_per_beat


# Spec:
# - General description: Add bounded Gaussian jitter to beat timestamps and keep them sorted inside the valid duration.
# - Params: `beat_timestamps`, one-dimensional nondecreasing timestamps; `jitter_std_seconds`, non-negative jitter scale; `duration_seconds`, positive total duration; `rng`, NumPy generator.
# - Pre: `beat_timestamps` is one-dimensional, `jitter_std_seconds >= 0`, and `duration_seconds > 0`.
# - Post: Returns a one-dimensional sorted float64 array with values clipped to [0, duration_seconds).
# - Mathematical definition: y_k = clip(x_k + epsilon_k, 0, duration_seconds - delta), where epsilon_k ~ N(0, jitter_std_seconds^2), followed by sorting.
def add_physiological_jitter(
    beat_timestamps: FloatArray,
    jitter_std_seconds: float,
    duration_seconds: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Return beat timestamps with small physiological jitter."""
    if beat_timestamps.ndim != 1:
        raise ValueError("beat_timestamps must be one-dimensional.")
    if jitter_std_seconds < 0.0:
        raise ValueError("jitter_std_seconds must be non-negative.")
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive.")
    if beat_timestamps.size == 0 or jitter_std_seconds == 0.0:
        return np.asarray(beat_timestamps, dtype=np.float64).copy()
    jitter = rng.normal(loc=0.0, scale=jitter_std_seconds, size=beat_timestamps.shape)
    clipped_timestamps = np.clip(
        beat_timestamps + jitter,
        a_min=0.0,
        a_max=np.nextafter(duration_seconds, 0.0),
    )
    return np.sort(np.asarray(clipped_timestamps, dtype=np.float64))


# Spec:
# - General description: Build a uniform time axis for a sampled pulse train.
# - Params: `sample_rate_hz`, positive samples per second; `duration_seconds`, positive total duration.
# - Pre: `sample_rate_hz > 0` and `duration_seconds > 0`.
# - Post: Returns a one-dimensional float64 array with at least one sample and spacing `1 / sample_rate_hz`.
# - Mathematical definition: t_i = i / sample_rate_hz for i in {0, ..., floor(sample_rate_hz * duration_seconds) - 1}.
def build_time_axis(sample_rate_hz: float, duration_seconds: float) -> FloatArray:
    """Return a uniform time axis for sampled pulse-train generation."""
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive.")
    sample_count = int(math.floor(sample_rate_hz * duration_seconds))
    if sample_count < 1:
        raise ValueError("time axis must contain at least one sample.")
    return np.arange(sample_count, dtype=np.float64) / sample_rate_hz


# Spec:
# - General description: Generate a binary pulse train from beat timestamps on a sampled time axis.
# - Params: `time_axis`, one-dimensional sampled times; `beat_timestamps`, one-dimensional timestamps; `pulse_width_seconds`, positive pulse width.
# - Pre: `time_axis` and `beat_timestamps` are one-dimensional and `pulse_width_seconds > 0`.
# - Post: Returns a one-dimensional float64 array with the same shape as `time_axis`, containing values only in {0.0, 1.0}.
# - Mathematical definition: pulse_i = 1 if there exists beat time b with 0 <= time_axis_i - b < pulse_width_seconds, else 0.
def generate_pulse_train(
    time_axis: FloatArray,
    beat_timestamps: FloatArray,
    pulse_width_seconds: float,
) -> FloatArray:
    """Return a binary pulse train aligned to the provided beat timestamps."""
    if time_axis.ndim != 1:
        raise ValueError("time_axis must be one-dimensional.")
    if beat_timestamps.ndim != 1:
        raise ValueError("beat_timestamps must be one-dimensional.")
    if pulse_width_seconds <= 0.0:
        raise ValueError("pulse_width_seconds must be positive.")
    pulse_train = np.zeros_like(time_axis, dtype=np.float64)
    for beat_timestamp in beat_timestamps:
        pulse_mask = (time_axis >= beat_timestamp) & (
            time_axis < beat_timestamp + pulse_width_seconds
        )
        pulse_train[pulse_mask] = 1.0
    return pulse_train
