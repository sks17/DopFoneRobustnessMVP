"""Clean and perturbed dataset construction."""

from __future__ import annotations

import numpy as np

from datatypes.signal_example import SignalExample
from perturbations.registry import apply_registered_perturbation
from simulation.generator import generate_clean_signal_example


# Spec:
# - General description: Build a list of clean synthetic signal examples from a small config dictionary.
# - Params: `generation_config`, dictionary containing sample rate, duration, carrier frequency, heart rates, and examples per rate.
# - Pre: Required config keys exist and carry positive values.
# - Post: Returns a non-empty list of clean `SignalExample` values.
# - Mathematical definition: Produces one example for each pair in heart_rate_bpm_values x {0, ..., examples_per_rate - 1}.
def build_clean_dataset(generation_config: dict[str, object]) -> list[SignalExample]:
    """Return a list of clean synthetic signal examples."""
    sample_rate_hz = float(generation_config["sample_rate_hz"])
    duration_seconds = float(generation_config["duration_seconds"])
    carrier_frequency_hz = float(generation_config["carrier_frequency_hz"])
    heart_rate_bpm_values = list(generation_config["heart_rate_bpm_values"])
    examples_per_rate = int(generation_config["examples_per_rate"])

    dataset: list[SignalExample] = []
    for heart_rate_bpm in heart_rate_bpm_values:
        for example_index in range(examples_per_rate):
            example_id = f"clean-{int(float(heart_rate_bpm))}-{example_index}"
            dataset.append(
                generate_clean_signal_example(
                    example_id=example_id,
                    sample_rate_hz=sample_rate_hz,
                    duration_seconds=duration_seconds,
                    heart_rate_bpm=float(heart_rate_bpm),
                    carrier_frequency_hz=carrier_frequency_hz,
                )
            )
    return dataset


# Spec:
# - General description: Build a perturbed dataset by applying one perturbation to each clean example.
# - Params: `clean_dataset`, list of clean examples; `perturbation_name`, registry key; `parameters`, perturbation parameters.
# - Pre: `clean_dataset` is non-empty and `perturbation_name` is registered.
# - Post: Returns a list with the same length as `clean_dataset`.
# - Mathematical definition: Output list y where y_i = P(x_i; parameters).
def build_perturbed_dataset(
    clean_dataset: list[SignalExample],
    perturbation_name: str,
    parameters: dict[str, object],
) -> list[SignalExample]:
    """Return a perturbed dataset with one perturbation applied per example."""
    if len(clean_dataset) == 0:
        raise ValueError("clean_dataset must be non-empty.")
    return [
        apply_registered_perturbation(
            signal_example=example,
            perturbation_name=perturbation_name,
            parameters=parameters,
        )
        for example in clean_dataset
    ]


# Spec:
# - General description: Build a default perturbation-parameter dictionary for one named perturbation.
# - Params: `perturbation_name`, registry key; `seed`, integer random seed.
# - Pre: `perturbation_name` is one of the supported perturbation names.
# - Post: Returns a dictionary with the fields required by the named perturbation.
# - Mathematical definition: Piecewise selection over supported perturbation names.
def build_default_perturbation_parameters(
    perturbation_name: str,
    seed: int,
) -> dict[str, object]:
    """Return default parameters for the requested perturbation."""
    if perturbation_name == "gaussian_noise":
        return {"noise_std": 0.05, "rng": np.random.default_rng(seed)}
    if perturbation_name == "dropout":
        return {"dropout_fraction": 0.2, "start_fraction": 0.25}
    if perturbation_name == "attenuation":
        return {"attenuation_factor": 0.5}
    raise ValueError(f"Unknown perturbation_name: {perturbation_name}")
