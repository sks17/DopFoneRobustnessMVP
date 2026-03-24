"""Tests for dropout perturbations."""

from __future__ import annotations

import numpy as np
import pytest

from perturbations.dropout import (
    apply_contiguous_dropout,
    apply_dropout_perturbation,
)
from simulation.generator import generate_clean_signal_example


def test_apply_contiguous_dropout_one_case_zeroes_expected_segment() -> None:
    """One-case test for contiguous dropout."""
    signal = np.array([1.0, 2.0, 3.0, 4.0])

    dropped_signal = apply_contiguous_dropout(signal, dropout_fraction=0.5, start_fraction=0.25)

    assert np.allclose(dropped_signal, np.array([1.0, 0.0, 0.0, 4.0]))


def test_apply_dropout_perturbation_two_case_sets_metadata() -> None:
    """Two-case test for dropout perturbation metadata."""
    example_a = generate_clean_signal_example("d1", 200.0, 2.0, 120.0, 18.0)
    example_b = generate_clean_signal_example("d2", 200.0, 2.0, 135.0, 18.0)

    perturbed_a = apply_dropout_perturbation(example_a, dropout_fraction=0.1, start_fraction=0.2)
    perturbed_b = apply_dropout_perturbation(example_b, dropout_fraction=0.1, start_fraction=0.2)

    assert perturbed_a.perturbation_name == "dropout"
    assert perturbed_b.perturbation_name == "dropout"


def test_apply_contiguous_dropout_many_case_preserves_length() -> None:
    """Many-case test over multiple dropout placements."""
    signal = np.arange(10, dtype=np.float64)

    for start_fraction in [0.0, 0.2, 0.5]:
        dropped_signal = apply_contiguous_dropout(
            signal,
            dropout_fraction=0.2,
            start_fraction=start_fraction,
        )
        assert dropped_signal.shape == signal.shape


@pytest.mark.parametrize(
    ("dropout_fraction", "start_fraction"),
    [
        (-0.1, 0.0),
        (1.0, 0.0),
        (0.2, -0.1),
        (0.2, 1.0),
    ],
)
def test_apply_contiguous_dropout_branch_cases_raise_value_error(
    dropout_fraction: float,
    start_fraction: float,
) -> None:
    """Branch-coverage test for invalid dropout parameters."""
    with pytest.raises(ValueError):
        apply_contiguous_dropout(
            np.array([1.0, 2.0, 3.0]),
            dropout_fraction=dropout_fraction,
            start_fraction=start_fraction,
        )
