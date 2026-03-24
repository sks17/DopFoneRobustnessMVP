"""Datatype for one benchmark evaluation result."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkResult:
    """Represent estimation quality for one example.

    Abstract state:
    - A comparison between a true heart-rate label and an estimated heart-rate output.

    Concrete state:
    - `example_id`: stable identifier for the evaluated example.
    - `true_heart_rate_bpm`: positive label.
    - `estimated_heart_rate_bpm`: non-negative estimate.
    - `absolute_error_bpm`: non-negative absolute error.
    - `within_tolerance`: whether the estimate falls within a fixed tolerance.

    Representation invariants:
    - `example_id` is non-empty.
    - `true_heart_rate_bpm > 0`.
    - `estimated_heart_rate_bpm >= 0`.
    - `absolute_error_bpm >= 0`.

    Abstraction function:
    - Maps stored scalar values to a single scored comparison for one benchmark example.

    Subtype and supertype clarity:
    - This is not a dataset-level summary.
    - It is a per-example evaluation record from which aggregate metrics can be computed.
    """

    example_id: str
    true_heart_rate_bpm: float
    estimated_heart_rate_bpm: float
    absolute_error_bpm: float
    within_tolerance: bool

    # Spec:
    # - General description: Validate benchmark-result invariants after construction.
    # - Params: `self`, the newly constructed benchmark result.
    # - Pre: Constructor fields are populated.
    # - Post: All representation invariants hold.
    # - Mathematical definition: absolute_error_bpm >= 0, true rate lies in R_{>0}, and estimated rate lies in R_{>=0}.
    def __post_init__(self) -> None:
        """Validate representation invariants for the benchmark result."""
        validate_benchmark_result(self)

    # Spec:
    # - General description: Return the signed estimation error in beats per minute.
    # - Params: `self`, a valid benchmark result.
    # - Pre: Representation invariants hold.
    # - Post: Returns `estimated_heart_rate_bpm - true_heart_rate_bpm`.
    # - Mathematical definition: e = y_hat - y.
    def signed_error_bpm(self) -> float:
        """Return the signed heart-rate error."""
        return self.estimated_heart_rate_bpm - self.true_heart_rate_bpm


# Spec:
# - General description: Construct a validated `BenchmarkResult` from a true label and estimate.
# - Params: `example_id`, `true_heart_rate_bpm`, `estimated_heart_rate_bpm`, `tolerance_bpm`.
# - Pre: True heart-rate and tolerance are positive, estimated heart-rate is non-negative, and `example_id` is non-empty.
# - Post: Returns a `BenchmarkResult` whose absolute error and tolerance flag are consistent with the inputs.
# - Mathematical definition: error = |y_hat - y| and within_tolerance = [error <= tolerance].
def create_benchmark_result(
    example_id: str,
    true_heart_rate_bpm: float,
    estimated_heart_rate_bpm: float,
    tolerance_bpm: float,
) -> BenchmarkResult:
    """Create one validated benchmark result."""
    if tolerance_bpm <= 0.0:
        raise ValueError("tolerance_bpm must be positive.")
    absolute_error_bpm: float = abs(estimated_heart_rate_bpm - true_heart_rate_bpm)
    within_tolerance: bool = absolute_error_bpm <= tolerance_bpm
    return BenchmarkResult(
        example_id=example_id,
        true_heart_rate_bpm=true_heart_rate_bpm,
        estimated_heart_rate_bpm=estimated_heart_rate_bpm,
        absolute_error_bpm=absolute_error_bpm,
        within_tolerance=within_tolerance,
    )


# Spec:
# - General description: Check that a `BenchmarkResult` satisfies all representation invariants.
# - Params: `benchmark_result`, the datatype instance to validate.
# - Pre: `benchmark_result` exists.
# - Post: Returns `None` when valid; raises `ValueError` otherwise.
# - Mathematical definition: Valid iff identifier and scalar constraints hold.
def validate_benchmark_result(benchmark_result: BenchmarkResult) -> None:
    """Raise `ValueError` if the benchmark result violates its invariants."""
    if benchmark_result.example_id.strip() == "":
        raise ValueError("example_id must be non-empty.")
    if benchmark_result.true_heart_rate_bpm <= 0.0:
        raise ValueError("true_heart_rate_bpm must be positive.")
    if benchmark_result.estimated_heart_rate_bpm < 0.0:
        raise ValueError("estimated_heart_rate_bpm must be non-negative.")
    if benchmark_result.absolute_error_bpm < 0.0:
        raise ValueError("absolute_error_bpm must be non-negative.")
