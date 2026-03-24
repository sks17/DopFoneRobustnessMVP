"""Tests for additive noise perturbations."""

from __future__ import annotations

import numpy as np
import pytest

from perturbations.noise import add_gaussian_noise, apply_noise_perturbation
from simulation.generator import generate_clean_signal_example


def test_add_gaussian_noise_one_case_with_zero_std_keeps_signal() -> None:
    """One-case test for zero-variance noise."""
    signal = np.array([0.0, 1.0, -1.0])
    rng = np.random.default_rng(7)

    noisy_signal = add_gaussian_noise(signal, noise_std=0.0, rng=rng)

    assert np.allclose(noisy_signal, signal)


def test_apply_noise_perturbation_two_case_sets_metadata() -> None:
    """Two-case test covering two clean examples under the same perturbation."""
    rng = np.random.default_rng(7)
    example_a = generate_clean_signal_example("n1", 200.0, 2.0, 120.0, 18.0)
    example_b = generate_clean_signal_example("n2", 200.0, 2.0, 135.0, 18.0)

    perturbed_a = apply_noise_perturbation(example_a, noise_std=0.01, rng=rng)
    perturbed_b = apply_noise_perturbation(example_b, noise_std=0.01, rng=rng)

    assert perturbed_a.perturbation_name == "gaussian_noise"
    assert perturbed_b.perturbation_name == "gaussian_noise"


def test_add_gaussian_noise_many_case_is_shape_preserving() -> None:
    """Many-case test over several signal lengths."""
    rng = np.random.default_rng(9)

    for signal in [
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0, -1.0, 0.5]),
        np.linspace(-1.0, 1.0, num=10),
    ]:
        noisy_signal = add_gaussian_noise(signal, noise_std=0.05, rng=rng)
        assert noisy_signal.shape == signal.shape


def test_add_gaussian_noise_branch_case_rejects_negative_std() -> None:
    """Branch-coverage test for invalid noise standard deviation."""
    with pytest.raises(ValueError):
        add_gaussian_noise(np.array([0.0, 1.0]), noise_std=-0.1, rng=np.random.default_rng(1))
