"""Aggregate benchmark metrics for RADAR-Sense-MVP."""

from __future__ import annotations

from datatypes.benchmark_result import BenchmarkResult


# Spec:
# - General description: Compute the absolute error between a true BPM label and an estimated BPM value.
# - Params: `true_heart_rate_bpm`, positive true BPM; `estimated_heart_rate_bpm`, non-negative estimated BPM.
# - Pre: `true_heart_rate_bpm > 0` and `estimated_heart_rate_bpm >= 0`.
# - Post: Returns a non-negative absolute error in BPM.
# - Mathematical definition: error = |estimated_heart_rate_bpm - true_heart_rate_bpm|.
def compute_absolute_error_bpm(
    true_heart_rate_bpm: float,
    estimated_heart_rate_bpm: float,
) -> float:
    """Return the absolute estimation error in beats per minute."""
    if true_heart_rate_bpm <= 0.0:
        raise ValueError("true_heart_rate_bpm must be positive.")
    if estimated_heart_rate_bpm < 0.0:
        raise ValueError("estimated_heart_rate_bpm must be non-negative.")
    return abs(estimated_heart_rate_bpm - true_heart_rate_bpm)


# Spec:
# - General description: Compute mean absolute error over a non-empty list of benchmark results.
# - Params: `results`, non-empty list of benchmark results.
# - Pre: `results` is non-empty.
# - Post: Returns the arithmetic mean of per-result absolute errors.
# - Mathematical definition: MAE = (1 / n) * sum_{i=1}^{n} result_i.absolute_error_bpm.
def compute_mean_absolute_error_bpm(results: list[BenchmarkResult]) -> float:
    """Return the mean absolute error in beats per minute."""
    if len(results) == 0:
        raise ValueError("results must be non-empty.")
    total_error = sum(result.absolute_error_bpm for result in results)
    return total_error / float(len(results))


# Spec:
# - General description: Compute the fraction of results that fall within tolerance.
# - Params: `results`, non-empty list of benchmark results.
# - Pre: `results` is non-empty.
# - Post: Returns a value in [0, 1].
# - Mathematical definition: success_rate = (1 / n) * sum_{i=1}^{n} 1[result_i.within_tolerance].
def compute_success_rate(results: list[BenchmarkResult]) -> float:
    """Return the fraction of results that fall within tolerance."""
    if len(results) == 0:
        raise ValueError("results must be non-empty.")
    success_count = sum(1 for result in results if result.within_tolerance)
    return success_count / float(len(results))


# Spec:
# - General description: Parse a structured perturbed example identifier into its artifact type and severity components.
# - Params: `example_id`, non-empty example identifier string.
# - Pre: `example_id` follows the perturbed identifier format `"{source_id}-{artifact_type}-sev{severity}-s{seed}"`.
# - Post: Returns a pair `(artifact_type, severity)` where `artifact_type` is a non-empty string and `severity` lies in [0, 1].
# - Mathematical definition: Splits the identifier into suffix tokens and parses the `sev{severity}` token as a float.
def parse_perturbed_example_id(example_id: str) -> tuple[str, float]:
    """Return the artifact type and severity encoded in a perturbed example identifier."""
    if example_id.strip() == "":
        raise ValueError("example_id must be non-empty.")
    parts = example_id.rsplit("-", maxsplit=3)
    if len(parts) != 4 or not parts[2].startswith("sev") or not parts[3].startswith("s"):
        raise ValueError("example_id must follow the perturbed benchmark identifier format.")
    artifact_type = parts[1]
    if artifact_type.strip() == "":
        raise ValueError("artifact_type must be non-empty.")
    severity_text = parts[2].removeprefix("sev")
    try:
        severity = float(severity_text)
    except ValueError as error:
        raise ValueError("severity must be parseable as a float.") from error
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must lie in [0, 1].")
    return artifact_type, severity


# Spec:
# - General description: Extract the artifact-type token from a structured perturbed example identifier.
# - Params: `example_id`, non-empty example identifier string.
# - Pre: `example_id` follows the perturbed identifier format `"{source_id}-{artifact_type}-sev{severity}-s{seed}"`.
# - Post: Returns the artifact-type substring.
# - Mathematical definition: artifact_type is the token immediately before `-sev{severity}-s{seed}`.
def extract_artifact_type_from_example_id(example_id: str) -> str:
    """Return the artifact type encoded in a perturbed example identifier."""
    artifact_type, _ = parse_perturbed_example_id(example_id)
    return artifact_type


# Spec:
# - General description: Extract the severity value from a structured perturbed example identifier.
# - Params: `example_id`, non-empty example identifier string.
# - Pre: `example_id` follows the perturbed identifier format `"{source_id}-{artifact_type}-sev{severity}-s{seed}"`.
# - Post: Returns the parsed severity as a float in [0, 1].
# - Mathematical definition: severity = float(text after the `sev` prefix).
def extract_severity_from_example_id(example_id: str) -> float:
    """Return the severity encoded in a perturbed example identifier."""
    _, severity = parse_perturbed_example_id(example_id)
    return severity


# Spec:
# - General description: Group benchmark results by artifact type using the artifact encoded in each example identifier.
# - Params: `results`, non-empty list of benchmark results.
# - Pre: `results` is non-empty and each result identifier follows the perturbed benchmark identifier format.
# - Post: Returns a dictionary mapping artifact type to a non-empty list of benchmark results.
# - Mathematical definition: groups[a] = { result_i : artifact(result_i.example_id) = a }.
def group_results_by_artifact_type(
    results: list[BenchmarkResult],
) -> dict[str, list[BenchmarkResult]]:
    """Return benchmark results grouped by artifact type."""
    if len(results) == 0:
        raise ValueError("results must be non-empty.")
    grouped_results: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        artifact_type = extract_artifact_type_from_example_id(result.example_id)
        grouped_results.setdefault(artifact_type, []).append(result)
    return grouped_results


# Spec:
# - General description: Group benchmark results by severity using the severity encoded in each example identifier.
# - Params: `results`, non-empty list of benchmark results.
# - Pre: `results` is non-empty and each result identifier follows the perturbed benchmark identifier format.
# - Post: Returns a dictionary mapping severity to a non-empty list of benchmark results.
# - Mathematical definition: groups[s] = { result_i : severity(result_i.example_id) = s }.
def group_results_by_severity(
    results: list[BenchmarkResult],
) -> dict[float, list[BenchmarkResult]]:
    """Return benchmark results grouped by severity."""
    if len(results) == 0:
        raise ValueError("results must be non-empty.")
    grouped_results: dict[float, list[BenchmarkResult]] = {}
    for result in results:
        severity = extract_severity_from_example_id(result.example_id)
        grouped_results.setdefault(severity, []).append(result)
    return grouped_results


# Spec:
# - General description: Compute mean absolute error separately for each artifact type.
# - Params: `results`, non-empty list of benchmark results.
# - Pre: `results` is non-empty and each result identifier follows the perturbed benchmark identifier format.
# - Post: Returns a dictionary mapping artifact type to grouped MAE.
# - Mathematical definition: grouped_mae[a] = MAE(groups[a]) for each artifact type a.
def compute_grouped_mae_by_artifact_type(
    results: list[BenchmarkResult],
) -> dict[str, float]:
    """Return mean absolute error grouped by artifact type."""
    grouped_results = group_results_by_artifact_type(results)
    return {
        artifact_type: compute_mean_absolute_error_bpm(group)
        for artifact_type, group in grouped_results.items()
    }


# Spec:
# - General description: Compute mean absolute error separately for each severity level.
# - Params: `results`, non-empty list of benchmark results.
# - Pre: `results` is non-empty and each result identifier follows the perturbed benchmark identifier format.
# - Post: Returns a dictionary mapping severity to grouped MAE.
# - Mathematical definition: grouped_mae[s] = MAE(groups[s]) for each severity level s.
def compute_grouped_mae_by_severity(
    results: list[BenchmarkResult],
) -> dict[float, float]:
    """Return mean absolute error grouped by severity."""
    grouped_results = group_results_by_severity(results)
    return {
        severity: compute_mean_absolute_error_bpm(group)
        for severity, group in grouped_results.items()
    }
