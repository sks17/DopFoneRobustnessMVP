"""Simple DopFone-inspired Doppler-like signal synthesis.

Generates a synthetic Doppler-like signal inspired by DopFone's 18 kHz
smartphone fetal heart rate sensing approach.  The model composes:

    carrier -> heartbeat-modulated envelope -> amplitude scaling -> additive noise

All parameters are explicit so the signal is fully reproducible from a seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import numpy.typing as npt

from simulation.heartbeat import (
    FloatArray,
    build_time_axis,
    generate_beat_timestamps,
    generate_pulse_train,
)
from utils.seed import build_rng


# Spec:
# - General description: Build a smooth heartbeat envelope on a sampled time axis by
#   generating a binary pulse train from beat timestamps and convolving with a Gaussian
#   kernel to produce a non-negative envelope in [0, 1].
# - Params: `time_axis`, one-dimensional sample times; `heart_rate_bpm`, positive heart
#   rate label; `pulse_width_seconds`, positive pulse width (default 0.06 s);
#   `smooth_std_seconds`, positive Gaussian smoothing std-dev (default 0.04 s).
# - Pre: `time_axis` is one-dimensional with at least two samples; scalar params positive.
# - Post: Returns a float64 array of the same shape as `time_axis` with values in [0, 1].
# - Mathematical definition:
#     p_i = pulse_train(t_i, beat_timestamps, pulse_width)
#     g_j = exp(-j^2 / (2 * (smooth_std * f_s)^2))  (Gaussian kernel)
#     raw_i = (p * g)_i                                (convolution)
#     envelope_i = raw_i / max(raw)                    (peak-normalise to [0, 1])
def generate_heartbeat_envelope(
    time_axis: FloatArray,
    heart_rate_bpm: float,
    pulse_width_seconds: float = 0.06,
    smooth_std_seconds: float = 0.04,
) -> FloatArray:
    """Return a smooth heartbeat envelope normalised to [0, 1]."""
    if time_axis.ndim != 1:
        raise ValueError("time_axis must be one-dimensional.")
    if time_axis.size < 2:
        raise ValueError("time_axis must contain at least two samples.")
    if heart_rate_bpm <= 0.0:
        raise ValueError("heart_rate_bpm must be positive.")
    if pulse_width_seconds <= 0.0:
        raise ValueError("pulse_width_seconds must be positive.")
    if smooth_std_seconds <= 0.0:
        raise ValueError("smooth_std_seconds must be positive.")

    duration_seconds: float = float(time_axis[-1]) + float(time_axis[1] - time_axis[0])
    beat_timestamps: FloatArray = generate_beat_timestamps(
        target_bpm=heart_rate_bpm,
        duration_seconds=duration_seconds,
    )
    pulse_train: FloatArray = generate_pulse_train(
        time_axis=time_axis,
        beat_timestamps=beat_timestamps,
        pulse_width_seconds=pulse_width_seconds,
    )

    # Gaussian smoothing kernel (truncated at 4 sigma, capped so kernel <= signal length).
    sample_spacing: float = float(time_axis[1] - time_axis[0])
    kernel_half_width: int = max(1, int(4.0 * smooth_std_seconds / sample_spacing))
    # Cap so the full kernel (2*h+1) never exceeds the signal length; this keeps
    # np.convolve(mode="same") returning the signal's length, not the kernel's.
    max_half: int = (time_axis.size - 1) // 2
    kernel_half_width = min(kernel_half_width, max_half)

    kernel_indices: FloatArray = np.arange(
        -kernel_half_width, kernel_half_width + 1, dtype=np.float64
    )
    kernel: FloatArray = np.exp(
        -0.5 * (kernel_indices * sample_spacing / smooth_std_seconds) ** 2
    )
    kernel = kernel / kernel.sum()

    smoothed: FloatArray = np.convolve(pulse_train, kernel, mode="same").astype(np.float64)

    peak: float = float(smoothed.max())
    if peak > 0.0:
        smoothed = smoothed / peak
    return smoothed


# Spec:
# - General description: Generate a sinusoidal carrier at the requested frequency.
# - Params: `time_axis`, one-dimensional sample times; `carrier_frequency_hz`, positive carrier frequency.
# - Pre: `time_axis` is one-dimensional with at least two samples and `carrier_frequency_hz > 0`.
# - Post: Returns a float64 array of the same shape as `time_axis` with values in [-1, 1].
# - Mathematical definition: c(t) = sin(2 * pi * f_c * t).
def generate_carrier_wave(time_axis: FloatArray, carrier_frequency_hz: float) -> FloatArray:
    """Return a sinusoidal carrier wave."""
    if time_axis.ndim != 1:
        raise ValueError("time_axis must be one-dimensional.")
    if time_axis.size < 2:
        raise ValueError("time_axis must contain at least two samples.")
    if carrier_frequency_hz <= 0.0:
        raise ValueError("carrier_frequency_hz must be positive.")
    return np.sin(2.0 * np.pi * carrier_frequency_hz * time_axis)


# Spec:
# - General description: Modulate a carrier wave with a heartbeat envelope to mimic Doppler-like observation intensity.
# - Params: `carrier_wave`, one-dimensional carrier samples; `heartbeat_envelope`, one-dimensional non-negative envelope samples; `modulation_depth`, depth in [0, 1].
# - Pre: Arrays are one-dimensional, have the same shape, and `0 <= modulation_depth <= 1`.
# - Post: Returns a float64 array of the same shape as the inputs.
# - Mathematical definition: y_i = c_i * ((1 - m) + m * e_i).
def modulate_carrier_with_envelope(
    carrier_wave: FloatArray,
    heartbeat_envelope: FloatArray,
    modulation_depth: float = 0.35,
) -> FloatArray:
    """Return a carrier wave scaled by the heartbeat envelope."""
    if carrier_wave.ndim != 1 or heartbeat_envelope.ndim != 1:
        raise ValueError("carrier_wave and heartbeat_envelope must be one-dimensional.")
    if carrier_wave.shape != heartbeat_envelope.shape:
        raise ValueError("carrier_wave and heartbeat_envelope must have the same shape.")
    if not 0.0 <= modulation_depth <= 1.0:
        raise ValueError("modulation_depth must lie in [0, 1].")
    amplitude_scale: FloatArray = (1.0 - modulation_depth) + (
        modulation_depth * heartbeat_envelope
    )
    return carrier_wave * amplitude_scale


# Spec:
# - General description: Scale a signal array by a non-negative amplitude factor to mimic
#   distance-dependent attenuation of the Doppler return.
# - Params: `signal`, one-dimensional float64 array; `amplitude_scale`, non-negative scalar.
# - Pre: `signal` is one-dimensional with at least two samples and `amplitude_scale >= 0`.
# - Post: Returns a float64 array of the same shape as `signal`.
# - Mathematical definition: y_i = a * x_i.
def apply_amplitude_scaling(signal: FloatArray, amplitude_scale: float) -> FloatArray:
    """Return the signal scaled by an amplitude factor."""
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if signal.size < 2:
        raise ValueError("signal must contain at least two samples.")
    if amplitude_scale < 0.0:
        raise ValueError("amplitude_scale must be non-negative.")
    return np.asarray(signal * amplitude_scale, dtype=np.float64)


# Spec:
# - General description: Add zero-mean Gaussian noise to a signal, using a seeded RNG for
#   reproducibility.
# - Params: `signal`, one-dimensional float64 array; `noise_std`, non-negative standard
#   deviation; `rng`, a NumPy random Generator.
# - Pre: `signal` is one-dimensional with at least two samples and `noise_std >= 0`.
# - Post: Returns a float64 array of the same shape as `signal`.  When `noise_std == 0`
#   the output equals the input exactly.
# - Mathematical definition: y_i = x_i + n_i where n_i ~ N(0, sigma^2).
def add_baseline_noise(
    signal: FloatArray,
    noise_std: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Return the signal with additive Gaussian noise."""
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if signal.size < 2:
        raise ValueError("signal must contain at least two samples.")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative.")
    if noise_std == 0.0:
        return signal.copy()
    noise: FloatArray = rng.normal(loc=0.0, scale=noise_std, size=signal.shape).astype(
        np.float64
    )
    return np.asarray(signal + noise, dtype=np.float64)


# ---------------------------------------------------------------------------
# Datatype: DopplerSignalResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DopplerSignalResult:
    """Immutable container for a generated Doppler-like waveform and its metadata.

    Abstract state:
    - A sampled Doppler-like waveform together with all generation parameters so the
      signal is fully reproducible.

    Concrete state:
    - `signal`: float64 array of shape ``(n,)`` — the final waveform.
    - `time_axis`: float64 array of shape ``(n,)`` — sample times in seconds.
    - `sample_rate_hz`: positive sampling rate used.
    - `carrier_frequency_hz`: positive carrier frequency used.
    - `heart_rate_bpm`: positive heart-rate label embedded in the signal.
    - `modulation_depth`: modulation depth in [0, 1].
    - `amplitude_scale`: non-negative amplitude scaling factor applied.
    - `noise_std`: non-negative noise standard deviation applied.
    - `seed`: non-negative integer seed used for noise generation.

    Representation invariants:
    - ``signal.ndim == 1`` and ``signal.size >= 2``.
    - ``time_axis.shape == signal.shape``.
    - ``sample_rate_hz > 0``.
    - ``carrier_frequency_hz > 0``.
    - ``heart_rate_bpm > 0``.
    - ``0 <= modulation_depth <= 1``.
    - ``amplitude_scale >= 0``.
    - ``noise_std >= 0``.
    - ``seed >= 0``.

    Abstraction function:
    - Maps stored arrays and scalars to one fully-described synthetic Doppler observation.

    Subtype / supertype clarity:
    - More specific than a raw waveform: carries all generation provenance.
    - Not a benchmark example — use ``SignalExample`` for benchmark-labelled data.
    """

    signal: FloatArray
    time_axis: FloatArray
    sample_rate_hz: float
    carrier_frequency_hz: float
    heart_rate_bpm: float
    modulation_depth: float
    amplitude_scale: float
    noise_std: float
    seed: int

    # Spec:
    # - General description: Validate representation invariants after construction.
    # - Params: `self`.
    # - Pre: Fields are populated by the dataclass constructor.
    # - Post: All invariants hold or `ValueError` is raised.
    def __post_init__(self) -> None:
        """Validate representation invariants."""
        sig = np.asarray(self.signal, dtype=np.float64)
        ta = np.asarray(self.time_axis, dtype=np.float64)
        object.__setattr__(self, "signal", sig)
        object.__setattr__(self, "time_axis", ta)
        if sig.ndim != 1 or sig.size < 2:
            raise ValueError("signal must be one-dimensional with at least two samples.")
        if ta.shape != sig.shape:
            raise ValueError("time_axis must have the same shape as signal.")
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive.")
        if self.carrier_frequency_hz <= 0.0:
            raise ValueError("carrier_frequency_hz must be positive.")
        if self.heart_rate_bpm <= 0.0:
            raise ValueError("heart_rate_bpm must be positive.")
        if not 0.0 <= self.modulation_depth <= 1.0:
            raise ValueError("modulation_depth must lie in [0, 1].")
        if self.amplitude_scale < 0.0:
            raise ValueError("amplitude_scale must be non-negative.")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be non-negative.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")

    # Spec:
    # - General description: Return the number of samples in the waveform.
    # - Params: `self`.
    # - Pre: Invariants hold.
    # - Post: Returns a positive integer.
    # - Mathematical definition: n = |signal|.
    def sample_count(self) -> int:
        """Return the number of samples."""
        return int(self.signal.size)

    # Spec:
    # - General description: Return the generation parameters as a plain dict (useful for logging).
    # - Params: `self`.
    # - Pre: Invariants hold.
    # - Post: Returns a dict with string keys and scalar values.
    def metadata(self) -> Dict[str, float | int]:
        """Return generation metadata as a dictionary."""
        return {
            "sample_rate_hz": self.sample_rate_hz,
            "carrier_frequency_hz": self.carrier_frequency_hz,
            "heart_rate_bpm": self.heart_rate_bpm,
            "modulation_depth": self.modulation_depth,
            "amplitude_scale": self.amplitude_scale,
            "noise_std": self.noise_std,
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# High-level generation entry point
# ---------------------------------------------------------------------------

# Default constants matching the DopFone sensing setup.
DEFAULT_SAMPLE_RATE_HZ: float = 48_000.0
DEFAULT_CARRIER_FREQUENCY_HZ: float = 18_000.0


# Spec:
# - General description: Generate a complete synthetic Doppler-like waveform.  Composes
#   time-axis construction, heartbeat envelope generation, carrier generation,
#   amplitude-modulation, amplitude scaling, and optional additive noise into a single
#   deterministic pipeline.
# - Params:
#     `heart_rate_bpm` — positive fetal heart rate label (beats per minute).
#     `duration_seconds` — positive signal duration in seconds.
#     `sample_rate_hz` — positive sampling rate (default 48 000 Hz).
#     `carrier_frequency_hz` — positive carrier frequency (default 18 000 Hz).
#     `modulation_depth` — depth in [0, 1] (default 0.35).
#     `amplitude_scale` — non-negative attenuation factor (default 1.0).
#     `noise_std` — non-negative additive noise std-dev (default 0.0).
#     `seed` — non-negative integer for deterministic noise (default 0).
# - Pre: All scalar constraints above are satisfied.
# - Post: Returns a `DopplerSignalResult` whose `signal` field has length
#   `floor(sample_rate_hz * duration_seconds)` and whose metadata records every
#   generation parameter.
# - Mathematical definition:
#     t_i = i / f_s                                          (time axis)
#     e_i = heartbeat_envelope(t_i, bpm)                     (envelope)
#     c_i = sin(2 * pi * f_c * t_i)                          (carrier)
#     m_i = c_i * ((1 - d) + d * e_i)                        (modulated)
#     s_i = a * m_i + n_i,   n_i ~ N(0, sigma^2)            (scaled + noise)
def generate_doppler_signal(
    heart_rate_bpm: float,
    duration_seconds: float,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    carrier_frequency_hz: float = DEFAULT_CARRIER_FREQUENCY_HZ,
    modulation_depth: float = 0.35,
    amplitude_scale: float = 1.0,
    noise_std: float = 0.0,
    seed: int = 0,
) -> DopplerSignalResult:
    """Generate a synthetic Doppler-like signal and return it with metadata.

    This is the primary public entry point for the Doppler signal simulation layer.
    """
    # --- input validation (fail-fast before any computation) ---
    if heart_rate_bpm <= 0.0:
        raise ValueError("heart_rate_bpm must be positive.")
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive.")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    if carrier_frequency_hz <= 0.0:
        raise ValueError("carrier_frequency_hz must be positive.")
    if not 0.0 <= modulation_depth <= 1.0:
        raise ValueError("modulation_depth must lie in [0, 1].")
    if amplitude_scale < 0.0:
        raise ValueError("amplitude_scale must be non-negative.")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    # --- deterministic pipeline ---
    rng: np.random.Generator = build_rng(seed)

    time_axis: FloatArray = build_time_axis(
        sample_rate_hz=sample_rate_hz,
        duration_seconds=duration_seconds,
    )
    envelope: FloatArray = generate_heartbeat_envelope(
        time_axis=time_axis,
        heart_rate_bpm=heart_rate_bpm,
    )
    carrier: FloatArray = generate_carrier_wave(
        time_axis=time_axis,
        carrier_frequency_hz=carrier_frequency_hz,
    )
    modulated: FloatArray = modulate_carrier_with_envelope(
        carrier_wave=carrier,
        heartbeat_envelope=envelope,
        modulation_depth=modulation_depth,
    )
    scaled: FloatArray = apply_amplitude_scaling(
        signal=modulated,
        amplitude_scale=amplitude_scale,
    )
    final_signal: FloatArray = add_baseline_noise(
        signal=scaled,
        noise_std=noise_std,
        rng=rng,
    )

    return DopplerSignalResult(
        signal=final_signal,
        time_axis=time_axis,
        sample_rate_hz=sample_rate_hz,
        carrier_frequency_hz=carrier_frequency_hz,
        heart_rate_bpm=heart_rate_bpm,
        modulation_depth=modulation_depth,
        amplitude_scale=amplitude_scale,
        noise_std=noise_std,
        seed=seed,
    )
