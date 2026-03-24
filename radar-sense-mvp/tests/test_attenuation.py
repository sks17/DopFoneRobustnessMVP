"""Tests for attenuation perturbations.

Covers:
- apply_scalar_attenuation          (raw waveform)
- severity_to_attenuation_factor    (mapping)
- apply_attenuation_perturbation    (SignalExample level)
- apply_attenuation_with_severity   (severity interface)

Each group provides 1-case, 2-case, many-case, branch-coverage, and
statement-coverage tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from perturbations.attenuation import (
    apply_attenuation_perturbation,
    apply_attenuation_with_severity,
    apply_scalar_attenuation,
    severity_to_attenuation_factor,
)
from simulation.generator import generate_clean_signal_example


# ===================================================================
# apply_scalar_attenuation
# ===================================================================


def test_apply_scalar_attenuation_one_case_scales_signal() -> None:
    """One-case test for scalar attenuation."""
    signal = np.array([2.0, -2.0])
    attenuated = apply_scalar_attenuation(signal, attenuation_factor=0.5)
    assert np.allclose(attenuated, np.array([1.0, -1.0]))


def test_apply_scalar_attenuation_two_case_identity_and_zero() -> None:
    """Two-case: factor 1 preserves, factor 0 zeroes."""
    signal = np.array([3.0, -1.0, 0.5])
    identity = apply_scalar_attenuation(signal, attenuation_factor=1.0)
    zeroed = apply_scalar_attenuation(signal, attenuation_factor=0.0)
    assert np.array_equal(identity, signal)
    assert np.allclose(zeroed, np.zeros_like(signal))


def test_apply_scalar_attenuation_many_case_preserves_shape() -> None:
    """Many-case test over several attenuation factors."""
    signal = np.linspace(-1.0, 1.0, num=10)
    for factor in [0.0, 0.25, 0.5, 0.75, 1.0]:
        attenuated = apply_scalar_attenuation(signal, attenuation_factor=factor)
        assert attenuated.shape == signal.shape


def test_apply_scalar_attenuation_branch_rejects_negative_factor() -> None:
    """Branch: negative factor raises ValueError."""
    with pytest.raises(ValueError, match="attenuation_factor"):
        apply_scalar_attenuation(np.array([1.0, 2.0]), attenuation_factor=-0.1)


def test_apply_scalar_attenuation_branch_rejects_factor_above_one() -> None:
    """Branch: factor > 1 raises ValueError."""
    with pytest.raises(ValueError, match="attenuation_factor"):
        apply_scalar_attenuation(np.array([1.0, 2.0]), attenuation_factor=1.1)


def test_apply_scalar_attenuation_branch_rejects_wrong_ndim() -> None:
    """Branch: 2-D signal raises ValueError."""
    with pytest.raises(ValueError, match="one-dimensional"):
        apply_scalar_attenuation(np.ones((2, 3)), attenuation_factor=0.5)


def test_apply_scalar_attenuation_branch_rejects_too_short() -> None:
    """Branch: single-element signal raises ValueError."""
    with pytest.raises(ValueError, match="at least two"):
        apply_scalar_attenuation(np.array([1.0]), attenuation_factor=0.5)


# ===================================================================
# severity_to_attenuation_factor
# ===================================================================


def test_severity_to_attenuation_factor_one_case_zero() -> None:
    """One-case: severity 0 maps to factor 1 (no attenuation)."""
    assert severity_to_attenuation_factor(0.0) == 1.0


def test_severity_to_attenuation_factor_two_case_endpoints() -> None:
    """Two-case: severity 0 → factor 1, severity 1 → factor 0."""
    assert severity_to_attenuation_factor(0.0) == 1.0
    assert severity_to_attenuation_factor(1.0) == 0.0


def test_severity_to_attenuation_factor_many_case_monotonic_decrease() -> None:
    """Many-case: factor decreases monotonically with severity."""
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    factors = [severity_to_attenuation_factor(s) for s in severities]
    for i in range(len(factors) - 1):
        assert factors[i] >= factors[i + 1]


@pytest.mark.parametrize("bad_severity", [-0.1, 1.1])
def test_severity_to_attenuation_factor_branch_invalid_severity(bad_severity: float) -> None:
    """Branch: out-of-range severity raises ValueError."""
    with pytest.raises(ValueError, match="severity"):
        severity_to_attenuation_factor(bad_severity)


# ===================================================================
# apply_attenuation_perturbation
# ===================================================================


def test_apply_attenuation_perturbation_one_case_sets_metadata() -> None:
    """One-case: perturbation name and is_perturbed are set."""
    ex = generate_clean_signal_example("a1", 200.0, 2.0, 120.0, 18.0)
    p = apply_attenuation_perturbation(ex, attenuation_factor=0.5)
    assert p.perturbation_name == "attenuation"
    assert p.is_perturbed is True


def test_apply_attenuation_perturbation_two_case_preserves_label() -> None:
    """Two-case: label is preserved for two different heart rates."""
    for bpm in [120.0, 150.0]:
        ex = generate_clean_signal_example(f"a-{bpm}", 200.0, 2.0, bpm, 18.0)
        p = apply_attenuation_perturbation(ex, attenuation_factor=0.7)
        assert p.heart_rate_bpm == bpm


def test_apply_attenuation_perturbation_statement_shape_preserved() -> None:
    """Statement: output signal has same shape as input."""
    ex = generate_clean_signal_example("as", 200.0, 2.0, 140.0, 18.0)
    p = apply_attenuation_perturbation(ex, attenuation_factor=0.3)
    assert p.signal.shape == ex.signal.shape


# ===================================================================
# apply_attenuation_with_severity
# ===================================================================


def test_apply_attenuation_with_severity_one_case_zero_severity() -> None:
    """One-case: severity 0 yields a signal identical to the original."""
    ex = generate_clean_signal_example("sa0", 200.0, 2.0, 140.0, 18.0)
    p = apply_attenuation_with_severity(ex, severity=0.0)
    assert np.array_equal(p.signal, ex.signal)


def test_apply_attenuation_with_severity_two_case_full_severity() -> None:
    """Two-case: severity 0 preserves, severity 1 zeros."""
    ex = generate_clean_signal_example("sa2", 200.0, 2.0, 140.0, 18.0)
    p0 = apply_attenuation_with_severity(ex, severity=0.0)
    p1 = apply_attenuation_with_severity(ex, severity=1.0)
    assert np.array_equal(p0.signal, ex.signal)
    assert np.allclose(p1.signal, np.zeros_like(ex.signal))


def test_apply_attenuation_with_severity_many_case_monotonic_rms_decrease() -> None:
    """Many-case: RMS energy decreases monotonically with severity."""
    ex = generate_clean_signal_example("sam", 200.0, 2.0, 140.0, 18.0)
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    rms_values = []
    for s in severities:
        p = apply_attenuation_with_severity(ex, severity=s)
        rms_values.append(float(np.sqrt(np.mean(p.signal ** 2))))

    for i in range(len(rms_values) - 1):
        assert rms_values[i] >= rms_values[i + 1] - 1e-12


@pytest.mark.parametrize("bad_severity", [-0.1, 1.1])
def test_apply_attenuation_with_severity_branch_invalid_severity(bad_severity: float) -> None:
    """Branch: out-of-range severity raises ValueError."""
    ex = generate_clean_signal_example("sabs", 200.0, 2.0, 140.0, 18.0)
    with pytest.raises(ValueError, match="severity"):
        apply_attenuation_with_severity(ex, severity=bad_severity)


def test_apply_attenuation_with_severity_statement_metadata() -> None:
    """Statement: severity-based attenuation sets correct metadata."""
    ex = generate_clean_signal_example("sas", 200.0, 2.0, 140.0, 18.0)
    p = apply_attenuation_with_severity(ex, severity=0.5)
    assert p.is_perturbed is True
    assert p.perturbation_name == "attenuation"
