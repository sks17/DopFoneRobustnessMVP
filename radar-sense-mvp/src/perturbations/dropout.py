"""Signal dropout perturbation helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from datatypes.signal_example import SignalExample


FloatArray = npt.NDArray[np.float64]


# Spec:
# - General description: Replace one contiguous fraction of a signal with zeros.
# - Params: `signal`, one-dimensional waveform; `dropout_fraction`, fraction in [0, 1); `start_fraction`, fraction in [0, 1).
# - Pre: `signal` is one-dimensional with at least two samples and fractions lie in [0, 1).
# - Post: Returns a float64 array with the same shape as `signal`.
# - Mathematical definition: y_i = 0 for i in [s, s + k), else y_i = x_i, where s = floor(start_fraction * n) and k = floor(dropout_fraction * n).
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
# - General description: Apply contiguous dropout perturbation to a signal example.
# - Params: `signal_example`, clean source example; `dropout_fraction`, fraction in [0, 1); `start_fraction`, fraction in [0, 1).
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
