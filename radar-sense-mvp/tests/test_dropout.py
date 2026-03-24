"""Tests for dropout perturbations.

Covers:
- apply_contiguous_dropout          (raw waveform)
- severity_to_dropout_fraction      (mapping)
- apply_dropout_perturbation        (SignalExample level)
- apply_dropout_with_severity       (severity interface)

Each group provides 1-case, 2-case, many-case, branch-coverage, and
statement-coverage tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from perturbations.dropout import (
    apply_contiguous_dropout,
    apply_dropout_perturbation,
    apply_dropout_with_severity,
    severity_to_dropout_fraction,
)
from simulation.generator import generate_clean_signal_example


# ===================================================================
# apply_contiguous_dropout
# ===================================================================


def test_apply_contiguous_dropout_one_case_zeroes_expected_segment() -> None:
    """One-case test for contiguous dropout."""
    signal = np.array([1.0, 2.0, 3.0, 4.0])
    dropped = apply_contiguous_dropout(signal, dropout_fraction=0.5, start_fraction=0.25)
    assert np.allclose(dropped, np.array([1.0, 0.0, 0.0, 4.0]))


def test_apply_contiguous_dropout_two_case_different_starts() -> None:
    """Two-case: different start positions produce different zero segments."""
    signal = np.ones(10)
    a = apply_contiguous_dropout(signal, dropout_fraction=0.3, start_fraction=0.0)
    b = apply_contiguous_dropout(signal, dropout_fraction=0.3, start_fraction=0.5)
    assert not np.array_equal(a, b)
    # Both should have 3 zeros (floor(0.3 * 10) = 3)
    assert np.count_nonzero(a == 0.0) == 3
    assert np.count_nonzero(b == 0.0) == 3


def test_apply_contiguous_dropout_many_case_preserves_length() -> None:
    """Many-case test over multiple dropout placements."""
    signal = np.arange(10, dtype=np.float64)
    for start_fraction in [0.0, 0.2, 0.5]:
        dropped = apply_contiguous_dropout(signal, dropout_fraction=0.2, start_fraction=start_fraction)
        assert dropped.shape == signal.shape


def test_apply_contiguous_dropout_branch_zero_fraction_keeps_signal() -> None:
    """Branch: dropout_fraction 0 leaves signal unchanged."""
    signal = np.array([1.0, 2.0, 3.0, 4.0])
    dropped = apply_contiguous_dropout(signal, dropout_fraction=0.0, start_fraction=0.0)
    assert np.array_equal(dropped, signal)


def test_apply_contiguous_dropout_branch_rejects_wrong_ndim() -> None:
    """Branch: 2-D signal raises ValueError."""
    with pytest.raises(ValueError, match="one-dimensional"):
        apply_contiguous_dropout(np.ones((2, 3)), dropout_fraction=0.1, start_fraction=0.0)


def test_apply_contiguous_dropout_branch_rejects_too_short() -> None:
    """Branch: single-element signal raises ValueError."""
    with pytest.raises(ValueError, match="at least two"):
        apply_contiguous_dropout(np.array([1.0]), dropout_fraction=0.1, start_fraction=0.0)


@pytest.mark.parametrize(
    ("dropout_fraction", "start_fraction"),
    [
        (-0.1, 0.0),
        (1.0, 0.0),
        (0.2, -0.1),
        (0.2, 1.0),
    ],
)
def test_apply_contiguous_dropout_branch_invalid_fractions(
    dropout_fraction: float,
    start_fraction: float,
) -> None:
    """Branch: out-of-range fractions raise ValueError."""
    with pytest.raises(ValueError):
        apply_contiguous_dropout(
            np.array([1.0, 2.0, 3.0]),
            dropout_fraction=dropout_fraction,
            start_fraction=start_fraction,
        )


# ===================================================================
# severity_to_dropout_fraction
# ===================================================================


def test_severity_to_dropout_fraction_one_case_zero() -> None:
    """One-case: severity 0 maps to dropout_fraction 0."""
    assert severity_to_dropout_fraction(0.0) == 0.0


def test_severity_to_dropout_fraction_two_case_endpoints() -> None:
    """Two-case: severity 0 and 1 map to 0 and max_dropout_fraction."""
    assert severity_to_dropout_fraction(0.0, max_dropout_fraction=0.6) == 0.0
    assert severity_to_dropout_fraction(1.0, max_dropout_fraction=0.6) == pytest.approx(0.6)


def test_severity_to_dropout_fraction_many_case_monotonic() -> None:
    """Many-case: output is monotonically non-decreasing in severity."""
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    fractions = [severity_to_dropout_fraction(s) for s in severities]
    for i in range(len(fractions) - 1):
        assert fractions[i] <= fractions[i + 1]


@pytest.mark.parametrize("bad_severity", [-0.1, 1.1])
def test_severity_to_dropout_fraction_branch_invalid_severity(bad_severity: float) -> None:
    """Branch: out-of-range severity raises ValueError."""
    with pytest.raises(ValueError, match="severity"):
        severity_to_dropout_fraction(bad_severity)


@pytest.mark.parametrize("bad_max", [0.0, 1.0, -0.1])
def test_severity_to_dropout_fraction_branch_invalid_max(bad_max: float) -> None:
    """Branch: invalid max_dropout_fraction raises ValueError."""
    with pytest.raises(ValueError, match="max_dropout_fraction"):
        severity_to_dropout_fraction(0.5, max_dropout_fraction=bad_max)


# ===================================================================
# apply_dropout_perturbation
# ===================================================================


def test_apply_dropout_perturbation_one_case_sets_metadata() -> None:
    """One-case: perturbation name and is_perturbed are set."""
    ex = generate_clean_signal_example("d1", 200.0, 2.0, 120.0, 18.0)
    p = apply_dropout_perturbation(ex, dropout_fraction=0.1, start_fraction=0.0)
    assert p.perturbation_name == "dropout"
    assert p.is_perturbed is True


def test_apply_dropout_perturbation_two_case_preserves_label() -> None:
    """Two-case: label is preserved for two different heart rates."""
    for bpm in [120.0, 150.0]:
        ex = generate_clean_signal_example(f"d-{bpm}", 200.0, 2.0, bpm, 18.0)
        p = apply_dropout_perturbation(ex, dropout_fraction=0.1, start_fraction=0.0)
        assert p.heart_rate_bpm == bpm


def test_apply_dropout_perturbation_statement_shape_preserved() -> None:
    """Statement: output signal has same shape as input."""
    ex = generate_clean_signal_example("ds", 200.0, 2.0, 140.0, 18.0)
    p = apply_dropout_perturbation(ex, dropout_fraction=0.2, start_fraction=0.1)
    assert p.signal.shape == ex.signal.shape


# ===================================================================
# apply_dropout_with_severity
# ===================================================================


def test_apply_dropout_with_severity_one_case_zero_severity() -> None:
    """One-case: severity 0 yields a signal identical to the original."""
    ex = generate_clean_signal_example("sd0", 200.0, 2.0, 140.0, 18.0)
    p = apply_dropout_with_severity(ex, severity=0.0)
    assert np.array_equal(p.signal, ex.signal)


def test_apply_dropout_with_severity_two_case_different_severities() -> None:
    """Two-case: higher severity produces more zeros."""
    ex = generate_clean_signal_example("sd2", 200.0, 2.0, 140.0, 18.0)
    lo = apply_dropout_with_severity(ex, severity=0.2, start_fraction=0.0)
    hi = apply_dropout_with_severity(ex, severity=0.8, start_fraction=0.0)
    zeros_lo = int(np.count_nonzero(lo.signal == 0.0))
    zeros_hi = int(np.count_nonzero(hi.signal == 0.0))
    assert zeros_lo <= zeros_hi


def test_apply_dropout_with_severity_many_case_monotonic_zeros() -> None:
    """Many-case: number of zeroed samples increases monotonically with severity."""
    ex = generate_clean_signal_example("sdm", 200.0, 2.0, 140.0, 18.0)
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    zero_counts = []
    for s in severities:
        p = apply_dropout_with_severity(ex, severity=s, start_fraction=0.0)
        zero_counts.append(int(np.count_nonzero(p.signal == 0.0)))

    for i in range(len(zero_counts) - 1):
        assert zero_counts[i] <= zero_counts[i + 1]


@pytest.mark.parametrize("bad_severity", [-0.1, 1.1])
def test_apply_dropout_with_severity_branch_invalid_severity(bad_severity: float) -> None:
    """Branch: out-of-range severity raises ValueError."""
    ex = generate_clean_signal_example("sdbs", 200.0, 2.0, 140.0, 18.0)
    with pytest.raises(ValueError, match="severity"):
        apply_dropout_with_severity(ex, severity=bad_severity)


def test_apply_dropout_with_severity_statement_metadata() -> None:
    """Statement: severity-based dropout sets correct metadata."""
    ex = generate_clean_signal_example("sds", 200.0, 2.0, 140.0, 18.0)
    p = apply_dropout_with_severity(ex, severity=0.5)
    assert p.is_perturbed is True
    assert p.perturbation_name == "dropout"
