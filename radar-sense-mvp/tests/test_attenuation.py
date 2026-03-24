"""Tests for attenuation perturbations."""

from __future__ import annotations

import numpy as np
import pytest

from perturbations.attenuation import (
    apply_attenuation_perturbation,
    apply_scalar_attenuation,
)
from simulation.generator import generate_clean_signal_example


def test_apply_scalar_attenuation_one_case_scales_signal() -> None:
    """One-case test for scalar attenuation."""
    signal = np.array([2.0, -2.0])

    attenuated_signal = apply_scalar_attenuation(signal, attenuation_factor=0.5)

    assert np.allclose(attenuated_signal, np.array([1.0, -1.0]))


def test_apply_attenuation_perturbation_two_case_sets_metadata() -> None:
    """Two-case test for attenuation metadata."""
    example_a = generate_clean_signal_example("a1", 200.0, 2.0, 120.0, 18.0)
    example_b = generate_clean_signal_example("a2", 200.0, 2.0, 135.0, 18.0)

    perturbed_a = apply_attenuation_perturbation(example_a, attenuation_factor=0.5)
    perturbed_b = apply_attenuation_perturbation(example_b, attenuation_factor=0.5)

    assert perturbed_a.perturbation_name == "attenuation"
    assert perturbed_b.perturbation_name == "attenuation"


def test_apply_scalar_attenuation_many_case_preserves_shape() -> None:
    """Many-case test over several attenuation factors."""
    signal = np.linspace(-1.0, 1.0, num=10)

    for attenuation_factor in [0.0, 0.5, 1.0]:
        attenuated_signal = apply_scalar_attenuation(signal, attenuation_factor=attenuation_factor)
        assert attenuated_signal.shape == signal.shape


@pytest.mark.parametrize("attenuation_factor", [-0.1, 1.1])
def test_apply_scalar_attenuation_branch_cases_raise_value_error(
    attenuation_factor: float,
) -> None:
    """Branch-coverage test for invalid attenuation factors."""
    with pytest.raises(ValueError):
        apply_scalar_attenuation(np.array([1.0, 2.0]), attenuation_factor=attenuation_factor)
