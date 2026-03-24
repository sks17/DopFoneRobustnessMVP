"""Simple DopFone-inspired Doppler-like signal synthesis."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from simulation.heartbeat import FloatArray


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
