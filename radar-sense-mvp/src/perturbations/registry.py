"""Registry for supported perturbations.

Provides both a parameter-based dispatch (``apply_registered_perturbation``) and
a severity-based dispatch (``apply_perturbation_with_severity``).  The severity
interface accepts a single ``severity`` float in [0, 1] and maps it to the
appropriate physical parameter for each perturbation type.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from datatypes.signal_example import SignalExample
from perturbations.attenuation import (
    apply_attenuation_perturbation,
    apply_attenuation_with_severity,
)
from perturbations.dropout import (
    apply_dropout_perturbation,
    apply_dropout_with_severity,
)
from perturbations.noise import (
    apply_noise_perturbation,
    apply_noise_with_severity,
)


PerturbationCallable = Callable[..., SignalExample]
SeverityCallable = Callable[..., SignalExample]


# Spec:
# - General description: Return a mapping from perturbation names to parameter-based
#   perturbation functions.
# - Params: None.
# - Pre: None.
# - Post: Returns a dictionary with entries for ``gaussian_noise``, ``dropout``, and
#   ``attenuation``.
# - Mathematical definition: Finite map R from name strings to callable transformations.
def build_perturbation_registry() -> Dict[str, PerturbationCallable]:
    """Return the supported perturbation registry."""
    return {
        "gaussian_noise": apply_noise_perturbation,
        "dropout": apply_dropout_perturbation,
        "attenuation": apply_attenuation_perturbation,
    }


# Spec:
# - General description: Return a mapping from perturbation names to severity-based
#   perturbation functions.
# - Params: None.
# - Pre: None.
# - Post: Returns a dictionary with entries for ``gaussian_noise``, ``dropout``, and
#   ``attenuation``.  Each value accepts ``(signal_example, severity, **kwargs)``.
# - Mathematical definition: Finite map S from name strings to severity-aware callables.
def build_severity_registry() -> Dict[str, SeverityCallable]:
    """Return the severity-based perturbation registry."""
    return {
        "gaussian_noise": apply_noise_with_severity,
        "dropout": apply_dropout_with_severity,
        "attenuation": apply_attenuation_with_severity,
    }


# Spec:
# - General description: Return the list of registered perturbation names.
# - Params: None.
# - Pre: None.
# - Post: Returns a sorted list of strings.
def list_perturbation_names() -> List[str]:
    """Return sorted names of all registered perturbations."""
    return sorted(build_perturbation_registry().keys())


# Spec:
# - General description: Apply a named perturbation using the parameter-based registry.
# - Params: `signal_example`, source example; `perturbation_name`, registry key;
#   `parameters`, keyword arguments for the perturbation.
# - Pre: `perturbation_name` exists in the registry and parameters match the
#   perturbation signature.
# - Post: Returns a perturbed `SignalExample`.
# - Mathematical definition: result = R[name](signal_example, **parameters).
def apply_registered_perturbation(
    signal_example: SignalExample,
    perturbation_name: str,
    parameters: dict[str, float | np.random.Generator],
) -> SignalExample:
    """Apply one named perturbation with explicit parameters."""
    registry = build_perturbation_registry()
    if perturbation_name not in registry:
        raise ValueError(f"Unknown perturbation_name: {perturbation_name}")
    perturbation_function = registry[perturbation_name]
    return perturbation_function(signal_example, **parameters)


# Spec:
# - General description: Apply a named perturbation at a normalised severity level.
# - Params: `signal_example`, source example; `perturbation_name`, registry key;
#   `severity`, float in [0, 1]; `extra_kwargs`, optional additional keyword arguments
#   forwarded to the severity function (e.g. ``seed`` for noise).
# - Pre: `perturbation_name` exists in the severity registry and `0 <= severity <= 1`.
# - Post: Returns a perturbed `SignalExample`.
# - Mathematical definition: result = S[name](signal_example, severity, **extra_kwargs).
def apply_perturbation_with_severity(
    signal_example: SignalExample,
    perturbation_name: str,
    severity: float,
    **extra_kwargs: object,
) -> SignalExample:
    """Apply one named perturbation at a normalised severity level."""
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must lie in [0, 1].")
    registry = build_severity_registry()
    if perturbation_name not in registry:
        raise ValueError(f"Unknown perturbation_name: {perturbation_name}")
    severity_function = registry[perturbation_name]
    return severity_function(signal_example, severity=severity, **extra_kwargs)
