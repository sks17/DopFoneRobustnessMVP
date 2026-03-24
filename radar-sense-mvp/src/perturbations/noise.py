"""Additive noise perturbation helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from datatypes.signal_example import SignalExample


FloatArray = npt.NDArray[np.float64]


# Spec:
# - General description: Add zero-mean Gaussian noise to a waveform.
# - Params: `signal`, one-dimensional waveform; `noise_std`, non-negative standard deviation; `rng`, NumPy generator.
# - Pre: `signal` is one-dimensional, `noise_std >= 0`.
# - Post: Returns a float64 array with the same shape as `signal`.
# - Mathematical definition: y_i = x_i + epsilon_i, where epsilon_i ~ N(0, sigma^2).
def add_gaussian_noise(
    signal: FloatArray,
    noise_std: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Return a noisy copy of the input signal."""
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative.")
    noise = rng.normal(loc=0.0, scale=noise_std, size=signal.shape)
    return np.asarray(signal + noise, dtype=np.float64)


# Spec:
# - General description: Apply Gaussian noise perturbation to a signal example.
# - Params: `signal_example`, clean source example; `noise_std`, non-negative standard deviation; `rng`, NumPy generator.
# - Pre: `noise_std >= 0` and `signal_example` is valid.
# - Post: Returns a perturbed `SignalExample` with the same label and metadata updated to `gaussian_noise`.
# - Mathematical definition: Uses `add_gaussian_noise` on the stored waveform.
def apply_noise_perturbation(
    signal_example: SignalExample,
    noise_std: float,
    rng: np.random.Generator,
) -> SignalExample:
    """Return a new signal example with additive Gaussian noise."""
    noisy_signal = add_gaussian_noise(signal_example.signal, noise_std=noise_std, rng=rng)
    return SignalExample(
        example_id=f"{signal_example.example_id}-noise",
        signal=noisy_signal,
        sample_rate_hz=signal_example.sample_rate_hz,
        heart_rate_bpm=signal_example.heart_rate_bpm,
        is_perturbed=True,
        perturbation_name="gaussian_noise",
    )
