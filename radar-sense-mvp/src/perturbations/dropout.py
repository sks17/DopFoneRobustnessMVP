"""Signal dropout perturbation helpers.

Provides raw-waveform and ``SignalExample``-level contiguous-segment dropout,
plus a unified severity interface where ``severity`` in [0, 1] maps linearly
to ``dropout_fraction`` via a configurable ``max_dropout_fraction``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from datatypes.signal_example import SignalExample


FloatArray = npt.NDArray[np.float64]

# Default ceiling for severity → dropout_fraction mapping.
# Set below 1.0 so some signal always remains even at max severity.
DEFAULT_MAX_DROPOUT_FRACTION: float = 0.8


# Spec:
# - General description: Replace one contiguous fraction of a signal with zeros.
# - Params: `signal`, one-dimensional waveform; `dropout_fraction`, fraction in [0, 1);
#   `start_fraction`, fraction in [0, 1).
# - Pre: `signal` is one-dimensional with at least two samples and fractions lie in [0, 1).
# - Post: Returns a float64 array with the same shape as `signal`.
# - Mathematical definition: y_i = 0 for i in [s, s + k), else y_i = x_i,
#   where s = floor(start_fraction * n) and k = floor(dropout_fraction * n).
def apply_contiguous_dropout(
    signal: FloatArray,
    dropout_fraction: float,
    start_fraction: float,
) -> FloatArray:
    """Return a copy of the signal with one contiguous zeroed segment."""
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if signal.size < 2:
        raise ValueError("signal must contain at least two samples.")
    if not 0.0 <= dropout_fraction < 1.0:
        raise ValueError("dropout_fraction must lie in [0, 1).")
    if not 0.0 <= start_fraction < 1.0:
        raise ValueError("start_fraction must lie in [0, 1).")
    output = np.asarray(signal, dtype=np.float64).copy()
    dropout_count = int(np.floor(dropout_fraction * signal.size))
    start_index = int(np.floor(start_fraction * signal.size))
    end_index = min(signal.size, start_index + dropout_count)
    output[start_index:end_index] = 0.0
    return output


# Spec:
# - General description: Convert a severity value in [0, 1] to a dropout fraction.
# - Params: `severity`, float in [0, 1]; `max_dropout_fraction`, positive ceiling in (0, 1).
# - Pre: `0 <= severity <= 1` and `0 < max_dropout_fraction < 1`.
# - Post: Returns a float in [0, max_dropout_fraction).
# - Mathematical definition: dropout_fraction = severity * max_dropout_fraction.
def severity_to_dropout_fraction(
    severity: float,
    max_dropout_fraction: float = DEFAULT_MAX_DROPOUT_FRACTION,
) -> float:
    """Map a [0, 1] severity to a dropout fraction."""
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must lie in [0, 1].")
    if not 0.0 < max_dropout_fraction < 1.0:
        raise ValueError("max_dropout_fraction must lie in (0, 1).")
    return severity * max_dropout_fraction


# Spec:
# - General description: Apply contiguous dropout perturbation to a signal example.
# - Params: `signal_example`, clean source example; `dropout_fraction`, fraction in
#   [0, 1); `start_fraction`, fraction in [0, 1).
# - Pre: Fractions lie in [0, 1) and `signal_example` is valid.
# - Post: Returns a perturbed `SignalExample` with metadata updated to `dropout`.
# - Mathematical definition: Uses `apply_contiguous_dropout` on the stored waveform.
def apply_dropout_perturbation(
    signal_example: SignalExample,
    dropout_fraction: float,
    start_fraction: float,
) -> SignalExample:
    """Return a new signal example with contiguous dropout applied."""
    dropped_signal = apply_contiguous_dropout(
        signal_example.signal,
        dropout_fraction=dropout_fraction,
        start_fraction=start_fraction,
    )
    return SignalExample(
        example_id=f"{signal_example.example_id}-dropout",
        signal=dropped_signal,
        sample_rate_hz=signal_example.sample_rate_hz,
        heart_rate_bpm=signal_example.heart_rate_bpm,
        is_perturbed=True,
        perturbation_name="dropout",
    )


# Spec:
# - General description: Apply contiguous dropout at a given severity level.
# - Params: `signal_example`, clean source; `severity`, float in [0, 1];
#   `start_fraction`, fraction in [0, 1); `max_dropout_fraction`, positive ceiling.
# - Pre: `0 <= severity <= 1`, `0 <= start_fraction < 1`,
#   `0 < max_dropout_fraction < 1`.
# - Post: Returns a perturbed `SignalExample`.  The number of zeroed samples grows
#   monotonically with severity.
# - Mathematical definition: dropout_fraction = severity * max_dropout_fraction, then
#   `apply_dropout_perturbation(signal_example, dropout_fraction, start_fraction)`.
def apply_dropout_with_severity(
    signal_example: SignalExample,
    severity: float,
    start_fraction: float = 0.0,
    max_dropout_fraction: float = DEFAULT_MAX_DROPOUT_FRACTION,
) -> SignalExample:
    """Apply contiguous dropout at a normalised severity level."""
    dropout_fraction = severity_to_dropout_fraction(severity, max_dropout_fraction)
    return apply_dropout_perturbation(
        signal_example,
        dropout_fraction=dropout_fraction,
        start_fraction=start_fraction,
    )
