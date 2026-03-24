"""Datatype for one synthetic Doppler-like signal example."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SignalExample:
    """Represent one labeled time-series example for the benchmark.

    Abstract state:
    - A sampled one-dimensional signal with a known heart-rate label and metadata.

    Concrete state:
    - `example_id`: stable string identifier.
    - `signal`: NumPy float64 array of shape `(n,)`.
    - `sample_rate_hz`: positive sampling rate.
    - `heart_rate_bpm`: positive scalar label.
    - `is_perturbed`: whether a perturbation has been applied.
    - `perturbation_name`: optional perturbation identifier.

    Representation invariants:
    - `example_id` is non-empty.
    - `signal.ndim == 1`.
    - `signal.size >= 2`.
    - `sample_rate_hz > 0`.
    - `heart_rate_bpm > 0`.
    - `perturbation_name is None` if and only if `is_perturbed` is False.

    Abstraction function:
    - Maps the stored array and metadata to a single benchmark-ready observation with an objective heart-rate label.

    Subtype and supertype clarity:
    - This is a domain datatype, not a generic container.
    - It is more specific than a raw waveform because it always includes benchmark label semantics.
    """

    example_id: str
    signal: FloatArray
    sample_rate_hz: float
    heart_rate_bpm: float
    is_perturbed: bool
    perturbation_name: str | None

    # Spec:
    # - General description: Validate and normalize the datatype immediately after construction.
    # - Params: `self`, the newly constructed signal example.
    # - Pre: Constructor fields are populated.
    # - Post: `self.signal` has dtype float64 and all representation invariants hold.
    # - Mathematical definition: Let x be the stored signal. Enforce x in R^n with n >= 2 and scalar metadata constraints.
    def __post_init__(self) -> None:
        """Validate representation invariants for the signal example."""
        normalized_signal: FloatArray = np.asarray(self.signal, dtype=np.float64)
        object.__setattr__(self, "signal", normalized_signal)
        validate_signal_example(self)

    # Spec:
    # - General description: Return the number of samples in the stored signal.
    # - Params: `self`, a valid signal example.
    # - Pre: Representation invariants hold.
    # - Post: Returns an integer n equal to `self.signal.size`.
    # - Mathematical definition: n = |x| where x is the stored sample vector.
    def sample_count(self) -> int:
        """Return the number of samples in the signal."""
        return int(self.signal.size)

    # Spec:
    # - General description: Return the duration of the signal in seconds.
    # - Params: `self`, a valid signal example.
    # - Pre: Representation invariants hold and `sample_rate_hz > 0`.
    # - Post: Returns a positive float equal to sample count divided by sample rate.
    # - Mathematical definition: duration = n / f_s.
    def duration_seconds(self) -> float:
        """Return the signal duration in seconds."""
        return self.sample_count() / self.sample_rate_hz


# Spec:
# - General description: Check that a `SignalExample` satisfies all representation invariants.
# - Params: `signal_example`, the datatype instance to validate.
# - Pre: `signal_example` exists.
# - Post: Returns `None` when valid; raises `ValueError` otherwise.
# - Mathematical definition: Valid iff all invariant predicates over identifier, signal domain, and metadata evaluate to true.
def validate_signal_example(signal_example: SignalExample) -> None:
    """Raise `ValueError` if the signal example violates its invariants."""
    if signal_example.example_id.strip() == "":
        raise ValueError("example_id must be non-empty.")
    if signal_example.signal.ndim != 1:
        raise ValueError("signal must be one-dimensional.")
    if signal_example.signal.size < 2:
        raise ValueError("signal must contain at least two samples.")
    if signal_example.sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    if signal_example.heart_rate_bpm <= 0.0:
        raise ValueError("heart_rate_bpm must be positive.")
    has_name: bool = signal_example.perturbation_name is not None
    if signal_example.is_perturbed != has_name:
        raise ValueError(
            "perturbation_name must be present exactly when is_perturbed is True."
        )
