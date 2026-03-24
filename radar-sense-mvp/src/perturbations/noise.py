"""Additive noise perturbation helpers.

Provides raw-waveform and ``SignalExample``-level Gaussian noise injection,
plus a unified severity interface where ``severity`` in [0, 1] maps linearly
to ``noise_std`` via a configurable ``max_noise_std``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from datatypes.signal_example import SignalExample


FloatArray = npt.NDArray[np.float64]

# Default ceiling for the severity → noise_std linear mapping.
DEFAULT_MAX_NOISE_STD: float = 0.5


# Spec:
# - General description: Add zero-mean Gaussian noise to a waveform.
# - Params: `signal`, one-dimensional waveform; `noise_std`, non-negative standard
#   deviation; `rng`, NumPy generator.
# - Pre: `signal` is one-dimensional with at least two samples and `noise_std >= 0`.
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
    if signal.size < 2:
        raise ValueError("signal must contain at least two samples.")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative.")
    noise = rng.normal(loc=0.0, scale=noise_std, size=signal.shape)
    return np.asarray(signal + noise, dtype=np.float64)


# Spec:
# - General description: Convert a severity value in [0, 1] to a noise standard deviation.
# - Params: `severity`, float in [0, 1]; `max_noise_std`, positive ceiling.
# - Pre: `0 <= severity <= 1` and `max_noise_std > 0`.
# - Post: Returns a non-negative float.
# - Mathematical definition: noise_std = severity * max_noise_std.
def severity_to_noise_std(
    severity: float,
    max_noise_std: float = DEFAULT_MAX_NOISE_STD,
) -> float:
    """Map a [0, 1] severity to a noise standard deviation."""
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must lie in [0, 1].")
    if max_noise_std <= 0.0:
        raise ValueError("max_noise_std must be positive.")
    return severity * max_noise_std


# Spec:
# - General description: Apply Gaussian noise perturbation to a signal example.
# - Params: `signal_example`, clean source example; `noise_std`, non-negative standard
#   deviation; `rng`, NumPy generator.
# - Pre: `noise_std >= 0` and `signal_example` is valid.
# - Post: Returns a perturbed `SignalExample` with the same label and metadata updated
#   to `gaussian_noise`.
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


# Spec:
# - General description: Apply Gaussian noise at a given severity level.
# - Params: `signal_example`, clean source; `severity`, float in [0, 1]; `seed`,
#   non-negative integer; `max_noise_std`, positive ceiling.
# - Pre: `0 <= severity <= 1`, `seed >= 0`, `max_noise_std > 0`.
# - Post: Returns a perturbed `SignalExample`.  The perturbation magnitude grows
#   monotonically with severity.
# - Mathematical definition: noise_std = severity * max_noise_std, then
#   `apply_noise_perturbation(signal_example, noise_std, rng(seed))`.
def apply_noise_with_severity(
    signal_example: SignalExample,
    severity: float,
    seed: int,
    max_noise_std: float = DEFAULT_MAX_NOISE_STD,
) -> SignalExample:
    """Apply Gaussian noise at a normalised severity level."""
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    noise_std = severity_to_noise_std(severity, max_noise_std)
    rng = np.random.default_rng(seed)
    return apply_noise_perturbation(signal_example, noise_std=noise_std, rng=rng)
