"""Tests for the `SignalExample` datatype."""

from __future__ import annotations

import numpy as np
import pytest

from datatypes.signal_example import SignalExample, validate_signal_example


def test_signal_example_one_case_has_expected_duration() -> None:
    """One-case test for a short clean signal."""
    example = SignalExample(
        example_id="clean-1",
        signal=np.array([0.0, 1.0, 0.0, -1.0]),
        sample_rate_hz=2.0,
        heart_rate_bpm=120.0,
        is_perturbed=False,
        perturbation_name=None,
    )

    assert example.sample_count() == 4
    assert example.duration_seconds() == 2.0


def test_signal_example_two_case_perturbation_branch_is_valid() -> None:
    """Two-case test covering clean and perturbed examples."""
    clean_example = SignalExample(
        example_id="clean-2",
        signal=np.array([1.0, 2.0]),
        sample_rate_hz=4.0,
        heart_rate_bpm=130.0,
        is_perturbed=False,
        perturbation_name=None,
    )
    perturbed_example = SignalExample(
        example_id="noise-2",
        signal=np.array([1.0, 2.0]),
        sample_rate_hz=4.0,
        heart_rate_bpm=130.0,
        is_perturbed=True,
        perturbation_name="gaussian_noise",
    )

    validate_signal_example(clean_example)
    validate_signal_example(perturbed_example)


def test_signal_example_many_case_normalizes_dtype() -> None:
    """Many-case test covering multiple sample values and dtype normalization."""
    integer_signal = np.array([1, 2, 3, 4, 5, 6])
    example = SignalExample(
        example_id="many-1",
        signal=integer_signal,
        sample_rate_hz=6.0,
        heart_rate_bpm=140.0,
        is_perturbed=False,
        perturbation_name=None,
    )

    assert example.signal.dtype == np.float64
    assert np.allclose(example.signal, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


@pytest.mark.parametrize(
    ("example_id", "signal", "sample_rate_hz", "heart_rate_bpm", "is_perturbed", "perturbation_name"),
    [
        ("", np.array([0.0, 1.0]), 1.0, 120.0, False, None),
        ("bad-shape", np.array([[0.0, 1.0]]), 1.0, 120.0, False, None),
        ("too-short", np.array([0.0]), 1.0, 120.0, False, None),
        ("bad-rate", np.array([0.0, 1.0]), 0.0, 120.0, False, None),
        ("bad-hr", np.array([0.0, 1.0]), 1.0, 0.0, False, None),
        ("bad-branch", np.array([0.0, 1.0]), 1.0, 120.0, True, None),
    ],
)
def test_signal_example_branch_cases_raise_value_error(
    example_id: str,
    signal: np.ndarray,
    sample_rate_hz: float,
    heart_rate_bpm: float,
    is_perturbed: bool,
    perturbation_name: str | None,
) -> None:
    """Branch-coverage test for invariant failures."""
    with pytest.raises(ValueError):
        SignalExample(
            example_id=example_id,
            signal=signal,
            sample_rate_hz=sample_rate_hz,
            heart_rate_bpm=heart_rate_bpm,
            is_perturbed=is_perturbed,
            perturbation_name=perturbation_name,
        )
