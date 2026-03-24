"""Tests for RADAR-inspired dataset construction.

Covers:
- build_clean_dataset              (config-driven clean generation)
- expand_to_perturbed_dataset      (artifact x severity x seed expansion)
- perturbed_example_to_record      (provenance-aware serialisation)

Each group provides 1-case, 2-case, many-case, branch-coverage, and
statement-coverage tests.  Special attention is given to the RADAR invariant:
the objective label (heart_rate_bpm) must be preserved through all
perturbations.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from benchmark.dataset_builder import (
    build_clean_dataset,
    expand_to_perturbed_dataset,
    perturbed_example_to_record,
)
from perturbations.registry import list_perturbation_names
from simulation.generator import generate_clean_signal_example


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _small_config(
    heart_rates: list[float] | None = None,
    examples_per_rate: int = 1,
) -> dict[str, object]:
    """Return a minimal generation config for testing."""
    return {
        "sample_rate_hz": 200.0,
        "duration_seconds": 2.0,
        "carrier_frequency_hz": 18.0,
        "heart_rate_bpm_values": heart_rates or [120.0],
        "examples_per_rate": examples_per_rate,
    }


def _make_clean(example_id: str = "c1", bpm: float = 140.0) -> list:
    """Return a one-element list with a single clean example."""
    return [generate_clean_signal_example(example_id, 200.0, 2.0, bpm, 18.0)]


# ===================================================================
# build_clean_dataset
# ===================================================================


def test_build_clean_dataset_one_case_creates_expected_count() -> None:
    """One-case: single rate, single example."""
    dataset = build_clean_dataset(_small_config())
    assert len(dataset) == 1
    assert dataset[0].is_perturbed is False


def test_build_clean_dataset_two_case_two_rates() -> None:
    """Two-case: two heart rates produce two examples."""
    dataset = build_clean_dataset(_small_config(heart_rates=[120.0, 150.0]))
    assert len(dataset) == 2
    bpms = {ex.heart_rate_bpm for ex in dataset}
    assert bpms == {120.0, 150.0}


def test_build_clean_dataset_many_case_rate_times_count() -> None:
    """Many-case: 3 rates x 2 examples = 6 total."""
    dataset = build_clean_dataset(
        _small_config(heart_rates=[120.0, 135.0, 150.0], examples_per_rate=2),
    )
    assert len(dataset) == 6


# ===================================================================
# expand_to_perturbed_dataset
# ===================================================================


def test_expand_one_case_single_combination() -> None:
    """One-case: 1 clean x 1 perturbation x 1 severity x 1 seed = 1 output."""
    clean = _make_clean()
    perturbed = expand_to_perturbed_dataset(
        clean, ["gaussian_noise"], [0.5], [0],
    )
    assert len(perturbed) == 1
    assert perturbed[0].is_perturbed is True
    assert perturbed[0].perturbation_name == "gaussian_noise"


def test_expand_two_case_two_clean_examples() -> None:
    """Two-case: 2 clean examples produce 2x expansion."""
    clean = [
        generate_clean_signal_example("c1", 200.0, 2.0, 120.0, 18.0),
        generate_clean_signal_example("c2", 200.0, 2.0, 150.0, 18.0),
    ]
    perturbed = expand_to_perturbed_dataset(
        clean, ["attenuation"], [0.5], [0],
    )
    assert len(perturbed) == 2
    assert perturbed[0].heart_rate_bpm == 120.0
    assert perturbed[1].heart_rate_bpm == 150.0


def test_expand_many_case_full_grid_count() -> None:
    """Many-case: 2 clean x 3 perturbations x 5 severities x 2 seeds = 60."""
    clean = [
        generate_clean_signal_example("c1", 200.0, 2.0, 120.0, 18.0),
        generate_clean_signal_example("c2", 200.0, 2.0, 150.0, 18.0),
    ]
    names = list_perturbation_names()  # 3 names
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    seeds = [0, 7]
    perturbed = expand_to_perturbed_dataset(clean, names, severities, seeds)
    assert len(perturbed) == 2 * 3 * 5 * 2


def test_expand_label_preservation_radar_invariant() -> None:
    """RADAR invariant: heart_rate_bpm is preserved for every perturbed output."""
    bpm = 135.0
    clean = _make_clean("label-test", bpm=bpm)
    names = list_perturbation_names()
    severities = [0.0, 0.5, 1.0]
    seeds = [0, 42]
    perturbed = expand_to_perturbed_dataset(clean, names, severities, seeds)
    for ex in perturbed:
        assert ex.heart_rate_bpm == bpm, (
            f"Label not preserved: expected {bpm}, got {ex.heart_rate_bpm}"
        )


def test_expand_metadata_correctness() -> None:
    """Statement: every output is perturbed and carries correct perturbation name."""
    clean = _make_clean()
    perturbed = expand_to_perturbed_dataset(
        clean, ["gaussian_noise", "attenuation"], [0.5], [0],
    )
    for ex in perturbed:
        assert ex.is_perturbed is True
        assert ex.perturbation_name in {"gaussian_noise", "attenuation"}


def test_expand_example_id_encodes_provenance() -> None:
    """Statement: example_id encodes source, perturbation, severity, and seed."""
    clean = _make_clean("src1")
    perturbed = expand_to_perturbed_dataset(
        clean, ["attenuation"], [0.75], [42],
    )
    eid = perturbed[0].example_id
    assert "src1" in eid
    assert "attenuation" in eid
    assert "sev0.75" in eid
    assert "s42" in eid


def test_expand_noise_monotonic_severity() -> None:
    """Many-case: for noise, RMS of noise increases monotonically with severity."""
    clean = _make_clean("mono")
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    perturbed = expand_to_perturbed_dataset(
        clean, ["gaussian_noise"], severities, [0],
    )
    rms_values = []
    original_signal = clean[0].signal
    for ex in perturbed:
        diff = ex.signal - original_signal
        rms_values.append(float(np.sqrt(np.mean(diff ** 2))))
    for i in range(len(rms_values) - 1):
        assert rms_values[i] <= rms_values[i + 1] + 1e-12


def test_expand_attenuation_monotonic_severity() -> None:
    """Many-case: for attenuation, RMS of output decreases with severity."""
    clean = _make_clean("att-mono")
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    perturbed = expand_to_perturbed_dataset(
        clean, ["attenuation"], severities, [0],
    )
    rms_values = []
    for ex in perturbed:
        rms_values.append(float(np.sqrt(np.mean(ex.signal ** 2))))
    for i in range(len(rms_values) - 1):
        assert rms_values[i] >= rms_values[i + 1] - 1e-12


def test_expand_branch_empty_clean_raises() -> None:
    """Branch: empty clean_examples raises ValueError."""
    with pytest.raises(ValueError, match="clean_examples"):
        expand_to_perturbed_dataset([], ["gaussian_noise"], [0.5], [0])


def test_expand_branch_empty_names_raises() -> None:
    """Branch: empty perturbation_names raises ValueError."""
    with pytest.raises(ValueError, match="perturbation_names"):
        expand_to_perturbed_dataset(_make_clean(), [], [0.5], [0])


def test_expand_branch_empty_severities_raises() -> None:
    """Branch: empty severities raises ValueError."""
    with pytest.raises(ValueError, match="severities"):
        expand_to_perturbed_dataset(_make_clean(), ["attenuation"], [], [0])


def test_expand_branch_empty_seeds_raises() -> None:
    """Branch: empty seeds raises ValueError."""
    with pytest.raises(ValueError, match="seeds"):
        expand_to_perturbed_dataset(_make_clean(), ["attenuation"], [0.5], [])


def test_expand_branch_unknown_name_raises() -> None:
    """Branch: unknown perturbation name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown perturbation_name"):
        expand_to_perturbed_dataset(_make_clean(), ["nonexistent"], [0.5], [0])


def test_expand_branch_invalid_severity_raises() -> None:
    """Branch: severity outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError, match="severity"):
        expand_to_perturbed_dataset(_make_clean(), ["attenuation"], [1.5], [0])


def test_expand_branch_negative_seed_raises() -> None:
    """Branch: negative seed raises ValueError."""
    with pytest.raises(ValueError, match="seed"):
        expand_to_perturbed_dataset(_make_clean(), ["attenuation"], [0.5], [-1])


# ===================================================================
# perturbed_example_to_record
# ===================================================================


def test_perturbed_record_one_case_has_provenance_keys() -> None:
    """One-case: record contains source_example_id, severity, seed."""
    clean = _make_clean("rec1")
    perturbed = expand_to_perturbed_dataset(clean, ["attenuation"], [0.5], [7])
    record = perturbed_example_to_record(perturbed[0], "rec1", 0.5, 7)
    assert record["source_example_id"] == "rec1"
    assert record["severity"] == 0.5
    assert record["seed"] == 7
    assert record["perturbation_name"] == "attenuation"


def test_perturbed_record_two_case_different_perturbations() -> None:
    """Two-case: different perturbation names are reflected in records."""
    clean = _make_clean("rec2")
    noise = expand_to_perturbed_dataset(clean, ["gaussian_noise"], [0.5], [0])
    att = expand_to_perturbed_dataset(clean, ["attenuation"], [0.5], [0])
    r_noise = perturbed_example_to_record(noise[0], "rec2", 0.5, 0)
    r_att = perturbed_example_to_record(att[0], "rec2", 0.5, 0)
    assert r_noise["perturbation_name"] == "gaussian_noise"
    assert r_att["perturbation_name"] == "attenuation"


def test_perturbed_record_statement_json_serializable() -> None:
    """Statement: every value in the record is JSON-serializable."""
    clean = _make_clean("rec-json")
    perturbed = expand_to_perturbed_dataset(clean, ["gaussian_noise"], [0.25], [0])
    record = perturbed_example_to_record(perturbed[0], "rec-json", 0.25, 0)
    serialized = json.dumps(record)
    assert isinstance(serialized, str)


def test_perturbed_record_statement_label_preserved() -> None:
    """Statement: heart_rate_bpm in record matches original clean example."""
    bpm = 145.0
    clean = _make_clean("rec-bpm", bpm=bpm)
    perturbed = expand_to_perturbed_dataset(clean, ["dropout"], [0.3], [0])
    record = perturbed_example_to_record(perturbed[0], "rec-bpm", 0.3, 0)
    assert record["heart_rate_bpm"] == bpm
