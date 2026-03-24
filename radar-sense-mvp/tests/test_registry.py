"""Tests for the perturbation registry.

Covers:
- build_perturbation_registry       (parameter-based)
- build_severity_registry           (severity-based)
- list_perturbation_names           (sorted names)
- apply_registered_perturbation     (parameter dispatch)
- apply_perturbation_with_severity  (severity dispatch)

Each group provides 1-case, 2-case, many-case, branch-coverage, and
statement-coverage tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from perturbations.registry import (
    apply_perturbation_with_severity,
    apply_registered_perturbation,
    build_perturbation_registry,
    build_severity_registry,
    list_perturbation_names,
)
from simulation.generator import generate_clean_signal_example


# ===================================================================
# build_perturbation_registry
# ===================================================================


def test_build_perturbation_registry_one_case_has_noise() -> None:
    """One-case: gaussian_noise is registered."""
    registry = build_perturbation_registry()
    assert "gaussian_noise" in registry


def test_build_perturbation_registry_two_case_has_dropout_and_attenuation() -> None:
    """Two-case: dropout and attenuation are both registered."""
    registry = build_perturbation_registry()
    assert "dropout" in registry
    assert "attenuation" in registry


def test_build_perturbation_registry_many_case_all_callables() -> None:
    """Many-case: every value in the registry is callable."""
    registry = build_perturbation_registry()
    for name, fn in registry.items():
        assert callable(fn), f"{name} is not callable"


# ===================================================================
# build_severity_registry
# ===================================================================


def test_build_severity_registry_one_case_has_noise() -> None:
    """One-case: gaussian_noise is in the severity registry."""
    registry = build_severity_registry()
    assert "gaussian_noise" in registry


def test_build_severity_registry_two_case_matches_param_registry_keys() -> None:
    """Two-case: severity registry has same keys as parameter registry."""
    param_keys = set(build_perturbation_registry().keys())
    severity_keys = set(build_severity_registry().keys())
    assert param_keys == severity_keys


def test_build_severity_registry_many_case_all_callables() -> None:
    """Many-case: every severity function is callable."""
    registry = build_severity_registry()
    for name, fn in registry.items():
        assert callable(fn), f"{name} is not callable"


# ===================================================================
# list_perturbation_names
# ===================================================================


def test_list_perturbation_names_one_case_returns_list() -> None:
    """One-case: returns a list."""
    names = list_perturbation_names()
    assert isinstance(names, list)


def test_list_perturbation_names_two_case_sorted() -> None:
    """Two-case: list is sorted alphabetically."""
    names = list_perturbation_names()
    assert names == sorted(names)


def test_list_perturbation_names_many_case_contains_all() -> None:
    """Many-case: all three expected perturbation names are present."""
    names = list_perturbation_names()
    assert "attenuation" in names
    assert "dropout" in names
    assert "gaussian_noise" in names


# ===================================================================
# apply_registered_perturbation
# ===================================================================


def test_apply_registered_perturbation_one_case_noise() -> None:
    """One-case: noise perturbation via registry."""
    ex = generate_clean_signal_example("rn1", 200.0, 2.0, 140.0, 18.0)
    p = apply_registered_perturbation(
        ex, "gaussian_noise", {"noise_std": 0.05, "rng": np.random.default_rng(0)},
    )
    assert p.perturbation_name == "gaussian_noise"
    assert p.is_perturbed is True


def test_apply_registered_perturbation_two_case_attenuation_and_dropout() -> None:
    """Two-case: attenuation and dropout via registry."""
    ex = generate_clean_signal_example("rp2", 200.0, 2.0, 140.0, 18.0)
    att = apply_registered_perturbation(ex, "attenuation", {"attenuation_factor": 0.5})
    drp = apply_registered_perturbation(
        ex, "dropout", {"dropout_fraction": 0.2, "start_fraction": 0.0},
    )
    assert att.perturbation_name == "attenuation"
    assert drp.perturbation_name == "dropout"


def test_apply_registered_perturbation_branch_unknown_name() -> None:
    """Branch: unknown name raises ValueError."""
    ex = generate_clean_signal_example("rpb", 200.0, 2.0, 140.0, 18.0)
    with pytest.raises(ValueError, match="Unknown perturbation_name"):
        apply_registered_perturbation(ex, "nonexistent", {})


# ===================================================================
# apply_perturbation_with_severity
# ===================================================================


def test_apply_perturbation_with_severity_one_case_noise() -> None:
    """One-case: severity-based noise dispatch."""
    ex = generate_clean_signal_example("sv1", 200.0, 2.0, 140.0, 18.0)
    p = apply_perturbation_with_severity(ex, "gaussian_noise", severity=0.5, seed=0)
    assert p.perturbation_name == "gaussian_noise"
    assert p.is_perturbed is True


def test_apply_perturbation_with_severity_two_case_attenuation_and_dropout() -> None:
    """Two-case: severity-based attenuation and dropout."""
    ex = generate_clean_signal_example("sv2", 200.0, 2.0, 140.0, 18.0)
    att = apply_perturbation_with_severity(ex, "attenuation", severity=0.3)
    drp = apply_perturbation_with_severity(ex, "dropout", severity=0.3)
    assert att.perturbation_name == "attenuation"
    assert drp.perturbation_name == "dropout"


def test_apply_perturbation_with_severity_many_case_all_names() -> None:
    """Many-case: all registered names work with severity dispatch."""
    ex = generate_clean_signal_example("svm", 200.0, 2.0, 140.0, 18.0)
    for name in list_perturbation_names():
        kwargs = {"seed": 0} if name == "gaussian_noise" else {}
        p = apply_perturbation_with_severity(ex, name, severity=0.5, **kwargs)
        assert p.is_perturbed is True


def test_apply_perturbation_with_severity_branch_unknown_name() -> None:
    """Branch: unknown name raises ValueError."""
    ex = generate_clean_signal_example("svb", 200.0, 2.0, 140.0, 18.0)
    with pytest.raises(ValueError, match="Unknown perturbation_name"):
        apply_perturbation_with_severity(ex, "nonexistent", severity=0.5)


@pytest.mark.parametrize("bad_severity", [-0.1, 1.1])
def test_apply_perturbation_with_severity_branch_invalid_severity(bad_severity: float) -> None:
    """Branch: out-of-range severity raises ValueError."""
    ex = generate_clean_signal_example("svbs", 200.0, 2.0, 140.0, 18.0)
    with pytest.raises(ValueError, match="severity"):
        apply_perturbation_with_severity(ex, "attenuation", severity=bad_severity)


def test_apply_perturbation_with_severity_statement_shape_preserved() -> None:
    """Statement: output shape matches input for all perturbation types."""
    ex = generate_clean_signal_example("svs", 200.0, 2.0, 140.0, 18.0)
    for name in list_perturbation_names():
        kwargs = {"seed": 0} if name == "gaussian_noise" else {}
        p = apply_perturbation_with_severity(ex, name, severity=0.5, **kwargs)
        assert p.signal.shape == ex.signal.shape
