"""Signal attenuation perturbation helpers.

Provides raw-waveform and ``SignalExample``-level scalar attenuation, plus a
unified severity interface where ``severity`` in [0, 1] maps to
``attenuation_factor`` = ``1 - severity`` (higher severity -> more attenuation).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from datatypes.signal_example import SignalExample


FloatArray = npt.NDArray[np.float64]


# Spec:
# - General description: Multiply a waveform by a scalar attenuation factor.
# - Params: `signal`, one-dimensional waveform; `attenuation_factor`, scalar in [0, 1].
# - Pre: `signal` is one-dimensional with at least two samples and `attenuation_factor`
#   lies in [0, 1].
# - Post: Returns a float64 array with the same shape as `signal`.
# - Mathematical definition: y_i = a * x_i.
def apply_scalar_attenuation(signal: FloatArray, attenuation_factor: float) -> FloatArray:
    """Return an attenuated copy of the signal."""
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if signal.size < 2:
        raise ValueError("signal must contain at least two samples.")
    if not 0.0 <= attenuation_factor <= 1.0:
        raise ValueError("attenuation_factor must lie in [0, 1].")
    return np.asarray(signal * attenuation_factor, dtype=np.float64)


# Spec:
# - General description: Convert a severity value in [0, 1] to an attenuation factor.
#   Higher severity means more attenuation (lower retained amplitude).
# - Params: `severity`, float in [0, 1].
# - Pre: `0 <= severity <= 1`.
# - Post: Returns a float in [0, 1].
# - Mathematical definition: attenuation_factor = 1 - severity.
def severity_to_attenuation_factor(severity: float) -> float:
    """Map a [0, 1] severity to an attenuation factor (1 = no attenuation)."""
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must lie in [0, 1].")
    return 1.0 - severity


# Spec:
# - General description: Apply scalar attenuation to a signal example.
# - Params: `signal_example`, clean source example; `attenuation_factor`, scalar in [0, 1].
# - Pre: `attenuation_factor` lies in [0, 1] and `signal_example` is valid.
# - Post: Returns a perturbed `SignalExample` with metadata updated to `attenuation`.
# - Mathematical definition: Uses `apply_scalar_attenuation` on the stored waveform.
def apply_attenuation_perturbation(
    signal_example: SignalExample,
    attenuation_factor: float,
) -> SignalExample:
    """Return a new signal example with scalar attenuation applied."""
    attenuated_signal = apply_scalar_attenuation(
        signal_example.signal,
        attenuation_factor=attenuation_factor,
    )
    return SignalExample(
        example_id=f"{signal_example.example_id}-attenuation",
        signal=attenuated_signal,
        sample_rate_hz=signal_example.sample_rate_hz,
        heart_rate_bpm=signal_example.heart_rate_bpm,
        is_perturbed=True,
        perturbation_name="attenuation",
    )


# Spec:
# - General description: Apply attenuation at a given severity level.
# - Params: `signal_example`, clean source; `severity`, float in [0, 1].
# - Pre: `0 <= severity <= 1` and `signal_example` is valid.
# - Post: Returns a perturbed `SignalExample`.  RMS energy decreases monotonically
#   with increasing severity.
# - Mathematical definition: attenuation_factor = 1 - severity, then
#   `apply_attenuation_perturbation(signal_example, attenuation_factor)`.
def apply_attenuation_with_severity(
    signal_example: SignalExample,
    severity: float,
) -> SignalExample:
    """Apply scalar attenuation at a normalised severity level."""
    factor = severity_to_attenuation_factor(severity)
    return apply_attenuation_perturbation(signal_example, attenuation_factor=factor)
