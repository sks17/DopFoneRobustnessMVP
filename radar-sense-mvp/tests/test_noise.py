"""Tests for additive noise perturbations.

Covers:
- add_gaussian_noise             (raw waveform)
- severity_to_noise_std          (mapping)
- apply_noise_perturbation       (SignalExample level)
- apply_noise_with_severity      (severity interface)

Each group provides 1-case, 2-case, many-case, branch-coverage, and
statement-coverage tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from perturbations.noise import (
    add_gaussian_noise,
    apply_noise_perturbation,
    apply_noise_with_severity,
    severity_to_noise_std,
)
from simulation.generator import generate_clean_signal_example


# ===================================================================
# add_gaussian_noise
# ===================================================================


def test_add_gaussian_noise_one_case_with_zero_std_keeps_signal() -> None:
    """One-case test for zero-variance noise."""
    signal = np.array([0.0, 1.0, -1.0])
    rng = np.random.default_rng(7)
    noisy_signal = add_gaussian_noise(signal, noise_std=0.0, rng=rng)
    assert np.allclose(noisy_signal, signal)


def test_add_gaussian_noise_two_case_deterministic_under_seed() -> None:
    """Two-case: same seed identical, different seed differs."""
    signal = np.zeros(500)
    a = add_gaussian_noise(signal, 0.1, np.random.default_rng(42))
    b = add_gaussian_noise(signal, 0.1, np.random.default_rng(42))
    c = add_gaussian_noise(signal, 0.1, np.random.default_rng(99))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


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


def test_add_gaussian_noise_branch_rejects_negative_std() -> None:
    """Branch: negative noise_std raises ValueError."""
    with pytest.raises(ValueError, match="noise_std"):
        add_gaussian_noise(np.array([0.0, 1.0]), noise_std=-0.1, rng=np.random.default_rng(1))


def test_add_gaussian_noise_branch_rejects_wrong_ndim() -> None:
    """Branch: 2-D signal raises ValueError."""
    with pytest.raises(ValueError, match="one-dimensional"):
        add_gaussian_noise(np.ones((2, 3)), noise_std=0.1, rng=np.random.default_rng(0))


def test_add_gaussian_noise_branch_rejects_too_short() -> None:
    """Branch: single-element signal raises ValueError."""
    with pytest.raises(ValueError, match="at least two"):
        add_gaussian_noise(np.array([1.0]), noise_std=0.1, rng=np.random.default_rng(0))


# ===================================================================
# severity_to_noise_std
# ===================================================================


def test_severity_to_noise_std_one_case_zero() -> None:
    """One-case: severity 0 maps to noise_std 0."""
    assert severity_to_noise_std(0.0) == 0.0


def test_severity_to_noise_std_two_case_endpoints() -> None:
    """Two-case: severity 0 and 1 map to 0 and max_noise_std."""
    assert severity_to_noise_std(0.0, max_noise_std=0.5) == 0.0
    assert severity_to_noise_std(1.0, max_noise_std=0.5) == 0.5


def test_severity_to_noise_std_many_case_monotonic() -> None:
    """Many-case: output is monotonically non-decreasing in severity."""
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    stds = [severity_to_noise_std(s) for s in severities]
    for i in range(len(stds) - 1):
        assert stds[i] <= stds[i + 1]


@pytest.mark.parametrize("bad_severity", [-0.1, 1.1])
def test_severity_to_noise_std_branch_invalid_severity(bad_severity: float) -> None:
    """Branch: out-of-range severity raises ValueError."""
    with pytest.raises(ValueError, match="severity"):
        severity_to_noise_std(bad_severity)


def test_severity_to_noise_std_branch_invalid_max() -> None:
    """Branch: non-positive max_noise_std raises ValueError."""
    with pytest.raises(ValueError, match="max_noise_std"):
        severity_to_noise_std(0.5, max_noise_std=0.0)


# ===================================================================
# apply_noise_perturbation
# ===================================================================


def test_apply_noise_perturbation_one_case_sets_metadata() -> None:
    """One-case: perturbation name and is_perturbed are set."""
    example = generate_clean_signal_example("n1", 200.0, 2.0, 120.0, 18.0)
    perturbed = apply_noise_perturbation(example, noise_std=0.01, rng=np.random.default_rng(7))
    assert perturbed.perturbation_name == "gaussian_noise"
    assert perturbed.is_perturbed is True


def test_apply_noise_perturbation_two_case_preserves_label() -> None:
    """Two-case: label is preserved for two different heart rates."""
    for bpm in [120.0, 150.0]:
        ex = generate_clean_signal_example(f"n-{bpm}", 200.0, 2.0, bpm, 18.0)
        p = apply_noise_perturbation(ex, noise_std=0.05, rng=np.random.default_rng(0))
        assert p.heart_rate_bpm == bpm


def test_apply_noise_perturbation_statement_shape_preserved() -> None:
    """Statement: output signal has same shape as input."""
    ex = generate_clean_signal_example("ns", 200.0, 2.0, 140.0, 18.0)
    p = apply_noise_perturbation(ex, noise_std=0.1, rng=np.random.default_rng(0))
    assert p.signal.shape == ex.signal.shape


# ===================================================================
# apply_noise_with_severity
# ===================================================================


def test_apply_noise_with_severity_one_case_zero_severity() -> None:
    """One-case: severity 0 yields a signal very close to the original."""
    ex = generate_clean_signal_example("sn0", 200.0, 2.0, 140.0, 18.0)
    p = apply_noise_with_severity(ex, severity=0.0, seed=0)
    assert np.allclose(p.signal, ex.signal, atol=1e-12)


def test_apply_noise_with_severity_two_case_determinism() -> None:
    """Two-case: same seed+severity is deterministic; different seed differs."""
    ex = generate_clean_signal_example("sn2", 200.0, 2.0, 140.0, 18.0)
    a = apply_noise_with_severity(ex, severity=0.5, seed=7)
    b = apply_noise_with_severity(ex, severity=0.5, seed=7)
    c = apply_noise_with_severity(ex, severity=0.5, seed=99)
    assert np.array_equal(a.signal, b.signal)
    assert not np.array_equal(a.signal, c.signal)


def test_apply_noise_with_severity_many_case_monotonic_rms_increase() -> None:
    """Many-case: noise RMS increases monotonically with severity."""
    ex = generate_clean_signal_example("snm", 200.0, 2.0, 140.0, 18.0)
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    noise_rms_values = []
    for s in severities:
        p = apply_noise_with_severity(ex, severity=s, seed=0)
        diff = p.signal - ex.signal
        noise_rms_values.append(float(np.sqrt(np.mean(diff ** 2))))

    for i in range(len(noise_rms_values) - 1):
        assert noise_rms_values[i] <= noise_rms_values[i + 1] + 1e-12


def test_apply_noise_with_severity_branch_invalid_seed() -> None:
    """Branch: negative seed raises ValueError."""
    ex = generate_clean_signal_example("snb", 200.0, 2.0, 140.0, 18.0)
    with pytest.raises(ValueError, match="seed"):
        apply_noise_with_severity(ex, severity=0.5, seed=-1)


@pytest.mark.parametrize("bad_severity", [-0.1, 1.1])
def test_apply_noise_with_severity_branch_invalid_severity(bad_severity: float) -> None:
    """Branch: out-of-range severity raises ValueError."""
    ex = generate_clean_signal_example("snbs", 200.0, 2.0, 140.0, 18.0)
    with pytest.raises(ValueError, match="severity"):
        apply_noise_with_severity(ex, severity=bad_severity, seed=0)
