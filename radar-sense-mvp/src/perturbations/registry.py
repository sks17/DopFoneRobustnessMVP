"""Registry for supported perturbations."""

from __future__ import annotations

from typing import Callable

import numpy as np

from datatypes.signal_example import SignalExample
from perturbations.attenuation import apply_attenuation_perturbation
from perturbations.dropout import apply_dropout_perturbation
from perturbations.noise import apply_noise_perturbation


PerturbationCallable = Callable[..., SignalExample]


# Spec:
# - General description: Return a mapping from perturbation names to perturbation functions.
# - Params: None.
# - Pre: None.
# - Post: Returns a dictionary with entries for `gaussian_noise`, `dropout`, and `attenuation`.
# - Mathematical definition: Finite map R from name strings to callable transformations.
def build_perturbation_registry() -> dict[str, PerturbationCallable]:
    """Return the supported perturbation registry."""
    return {
        "gaussian_noise": apply_noise_perturbation,
        "dropout": apply_dropout_perturbation,
        "attenuation": apply_attenuation_perturbation,
    }


# Spec:
# - General description: Apply a named perturbation using the perturbation registry.
# - Params: `signal_example`, source example; `perturbation_name`, registry key; `parameters`, keyword arguments for the perturbation.
# - Pre: `perturbation_name` exists in the registry and parameters match the perturbation signature.
# - Post: Returns a perturbed `SignalExample`.
# - Mathematical definition: result = R[name](signal_example, **parameters), where R is the perturbation registry.
def apply_registered_perturbation(
    signal_example: SignalExample,
    perturbation_name: str,
    parameters: dict[str, float | np.random.Generator],
) -> SignalExample:
    """Apply one named perturbation."""
    registry = build_perturbation_registry()
    if perturbation_name not in registry:
        raise ValueError(f"Unknown perturbation_name: {perturbation_name}")
    perturbation_function = registry[perturbation_name]
    return perturbation_function(signal_example, **parameters)
