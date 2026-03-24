"""Tests for high-level clean signal generation, serialization, and I/O.

Covers:
- generate_clean_signal_example  (existing, quick sanity)
- signal_example_to_record       (new)
- generate_clean_dataset         (new)
- write_manifest_jsonl / read_manifest_jsonl (new I/O)
- write_waveform_npy             (new I/O)

Each group provides 1-case, 2-case, many-case, branch-coverage, and
statement-coverage tests per the project hard requirements.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simulation.generator import (
    generate_clean_dataset,
    generate_clean_signal_example,
    signal_example_to_record,
)
from utils.io import (
    read_manifest_jsonl,
    write_manifest_jsonl,
    write_waveform_npy,
)


# ===================================================================
# signal_example_to_record
# ===================================================================


def test_signal_example_to_record_one_case_keys() -> None:
    """One-case: record contains the expected keys and no raw signal."""
    example = generate_clean_signal_example(
        example_id="rec-1",
        sample_rate_hz=200.0,
        duration_seconds=1.0,
        heart_rate_bpm=140.0,
        carrier_frequency_hz=18.0,
    )
    record = signal_example_to_record(example)

    assert set(record.keys()) == {
        "example_id",
        "sample_rate_hz",
        "heart_rate_bpm",
        "is_perturbed",
        "perturbation_name",
        "sample_count",
    }
    assert "signal" not in record
    assert record["example_id"] == "rec-1"
    assert record["sample_count"] == example.sample_count()


def test_signal_example_to_record_two_case_clean_vs_perturbed() -> None:
    """Two-case: clean and perturbed examples produce correct is_perturbed."""
    from datatypes.signal_example import SignalExample

    clean = generate_clean_signal_example("c1", 200.0, 1.0, 140.0, 18.0)
    perturbed = SignalExample(
        example_id="p1",
        signal=clean.signal.copy(),
        sample_rate_hz=200.0,
        heart_rate_bpm=140.0,
        is_perturbed=True,
        perturbation_name="noise",
    )

    rc = signal_example_to_record(clean)
    rp = signal_example_to_record(perturbed)

    assert rc["is_perturbed"] is False
    assert rc["perturbation_name"] is None
    assert rp["is_perturbed"] is True
    assert rp["perturbation_name"] == "noise"


def test_signal_example_to_record_many_case_all_json_serializable() -> None:
    """Many-case: all record values are JSON-serializable types."""
    import json

    examples = [
        generate_clean_signal_example(f"m{i}", 200.0, 1.0, bpm, 18.0)
        for i, bpm in enumerate([60.0, 120.0, 180.0])
    ]
    for ex in examples:
        record = signal_example_to_record(ex)
        # json.dumps will raise if any value is not serializable.
        serialized = json.dumps(record)
        assert isinstance(serialized, str)


# ===================================================================
# generate_clean_dataset
# ===================================================================


def test_generate_clean_dataset_one_case_single_rate() -> None:
    """One-case (N=1): single heart rate produces one example."""
    dataset = generate_clean_dataset(
        heart_rate_bpm_values=[140.0],
        sample_rate_hz=200.0,
        duration_seconds=1.0,
        carrier_frequency_hz=18.0,
        base_seed=0,
    )
    assert len(dataset) == 1
    assert dataset[0].heart_rate_bpm == 140.0
    assert dataset[0].is_perturbed is False
    assert dataset[0].example_id == "clean-140-s0"


def test_generate_clean_dataset_two_case_two_rates() -> None:
    """Two-case (N=2): two heart rates produce two examples with unique IDs."""
    dataset = generate_clean_dataset(
        heart_rate_bpm_values=[120.0, 150.0],
        sample_rate_hz=200.0,
        duration_seconds=1.0,
        carrier_frequency_hz=18.0,
        base_seed=10,
    )
    assert len(dataset) == 2
    assert dataset[0].example_id == "clean-120-s10"
    assert dataset[1].example_id == "clean-150-s11"
    assert dataset[0].heart_rate_bpm == 120.0
    assert dataset[1].heart_rate_bpm == 150.0


def test_generate_clean_dataset_many_case_five_rates() -> None:
    """Many-case (N=5): five heart rates produce five valid examples."""
    rates = [60.0, 90.0, 120.0, 150.0, 180.0]
    dataset = generate_clean_dataset(
        heart_rate_bpm_values=rates,
        sample_rate_hz=200.0,
        duration_seconds=2.0,
        carrier_frequency_hz=18.0,
        base_seed=0,
    )
    assert len(dataset) == 5
    for i, ex in enumerate(dataset):
        assert ex.heart_rate_bpm == rates[i]
        assert ex.is_perturbed is False
        assert ex.signal.ndim == 1
        assert ex.signal.size == 400  # 200 * 2.0


def test_generate_clean_dataset_deterministic_across_calls() -> None:
    """Statement: same parameters produce bit-identical datasets."""
    kwargs = dict(
        heart_rate_bpm_values=[120.0, 150.0],
        sample_rate_hz=200.0,
        duration_seconds=1.0,
        carrier_frequency_hz=18.0,
        base_seed=42,
    )
    d1 = generate_clean_dataset(**kwargs)
    d2 = generate_clean_dataset(**kwargs)

    for a, b in zip(d1, d2):
        assert np.array_equal(a.signal, b.signal)
        assert a.example_id == b.example_id


@pytest.mark.parametrize(
    "bad_kwarg",
    [
        dict(heart_rate_bpm_values=[]),
        dict(sample_rate_hz=-1.0),
        dict(sample_rate_hz=0.0),
        dict(duration_seconds=-1.0),
        dict(duration_seconds=0.0),
        dict(carrier_frequency_hz=-1.0),
        dict(carrier_frequency_hz=0.0),
        dict(base_seed=-1),
    ],
)
def test_generate_clean_dataset_branch_invalid_params_raise(
    bad_kwarg: dict,
) -> None:
    """Branch: every invalid parameter triggers ValueError."""
    good = dict(
        heart_rate_bpm_values=[140.0],
        sample_rate_hz=200.0,
        duration_seconds=1.0,
        carrier_frequency_hz=18.0,
        base_seed=0,
    )
    with pytest.raises(ValueError):
        generate_clean_dataset(**{**good, **bad_kwarg})


# ===================================================================
# write_manifest_jsonl / read_manifest_jsonl  (I/O round-trip)
# ===================================================================


def test_write_read_manifest_jsonl_one_case(tmp_path: Path) -> None:
    """One-case: single record round-trips through JSONL."""
    records = [{"example_id": "ex-0", "bpm": 140.0}]
    manifest_path = tmp_path / "manifest.jsonl"

    write_manifest_jsonl(manifest_path, records)
    loaded = read_manifest_jsonl(manifest_path)

    assert loaded == records


def test_write_read_manifest_jsonl_two_case(tmp_path: Path) -> None:
    """Two-case: two records round-trip correctly."""
    records = [
        {"example_id": "ex-0", "bpm": 120.0},
        {"example_id": "ex-1", "bpm": 150.0},
    ]
    manifest_path = tmp_path / "manifest.jsonl"

    write_manifest_jsonl(manifest_path, records)
    loaded = read_manifest_jsonl(manifest_path)

    assert loaded == records
    assert len(loaded) == 2


def test_write_read_manifest_jsonl_many_case(tmp_path: Path) -> None:
    """Many-case: five records round-trip and line count matches."""
    records = [{"id": i, "val": float(i)} for i in range(5)]
    manifest_path = tmp_path / "sub" / "manifest.jsonl"

    write_manifest_jsonl(manifest_path, records)
    loaded = read_manifest_jsonl(manifest_path)

    assert loaded == records
    # Verify file has exactly 5 lines.
    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5


def test_write_manifest_jsonl_branch_empty_raises(tmp_path: Path) -> None:
    """Branch: empty records list raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        write_manifest_jsonl(tmp_path / "bad.jsonl", [])


def test_manifest_jsonl_creates_parent_dirs(tmp_path: Path) -> None:
    """Statement: parent directories are created automatically."""
    deep_path = tmp_path / "a" / "b" / "c" / "manifest.jsonl"
    write_manifest_jsonl(deep_path, [{"k": "v"}])
    assert deep_path.exists()


# ===================================================================
# write_waveform_npy  (I/O)
# ===================================================================


def test_write_waveform_npy_one_case_round_trip(tmp_path: Path) -> None:
    """One-case: waveform saved and loaded back matches original."""
    waveform = np.array([1.0, -1.0, 0.5], dtype=np.float64)
    npy_path = tmp_path / "signal.npy"

    write_waveform_npy(npy_path, waveform)
    loaded = np.load(npy_path)

    assert np.array_equal(loaded, waveform)


def test_write_waveform_npy_two_case_different_signals(tmp_path: Path) -> None:
    """Two-case: two different waveforms written to different files both round-trip."""
    w1 = np.array([1.0, 2.0], dtype=np.float64)
    w2 = np.array([3.0, 4.0, 5.0], dtype=np.float64)

    write_waveform_npy(tmp_path / "w1.npy", w1)
    write_waveform_npy(tmp_path / "w2.npy", w2)

    assert np.array_equal(np.load(tmp_path / "w1.npy"), w1)
    assert np.array_equal(np.load(tmp_path / "w2.npy"), w2)


def test_write_waveform_npy_many_case_from_dataset(tmp_path: Path) -> None:
    """Many-case: waveforms from a generated dataset all round-trip."""
    dataset = generate_clean_dataset(
        heart_rate_bpm_values=[60.0, 120.0, 180.0],
        sample_rate_hz=200.0,
        duration_seconds=1.0,
        carrier_frequency_hz=18.0,
    )
    for ex in dataset:
        path = tmp_path / f"{ex.example_id}.npy"
        write_waveform_npy(path, ex.signal)
        loaded = np.load(path)
        assert np.array_equal(loaded, ex.signal)


def test_write_waveform_npy_branch_wrong_ndim_raises(tmp_path: Path) -> None:
    """Branch: 2-D array raises ValueError."""
    with pytest.raises(ValueError, match="one-dimensional"):
        write_waveform_npy(tmp_path / "bad.npy", np.ones((2, 3)))


def test_write_waveform_npy_branch_too_short_raises(tmp_path: Path) -> None:
    """Branch: single-element array raises ValueError."""
    with pytest.raises(ValueError, match="at least two samples"):
        write_waveform_npy(tmp_path / "bad.npy", np.array([1.0]))


# ===================================================================
# End-to-end: generate + serialize + write + read
# ===================================================================


def test_end_to_end_dataset_to_manifest(tmp_path: Path) -> None:
    """Statement: full pipeline from generation through manifest I/O is consistent."""
    dataset = generate_clean_dataset(
        heart_rate_bpm_values=[120.0, 150.0],
        sample_rate_hz=200.0,
        duration_seconds=1.0,
        carrier_frequency_hz=18.0,
        base_seed=7,
    )

    # Serialize and write.
    records = []
    for ex in dataset:
        waveform_path = tmp_path / "waveforms" / f"{ex.example_id}.npy"
        write_waveform_npy(waveform_path, ex.signal)
        record = signal_example_to_record(ex)
        record["waveform_path"] = str(waveform_path)
        records.append(record)

    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest_jsonl(manifest_path, records)

    # Read back and verify.
    loaded_records = read_manifest_jsonl(manifest_path)
    assert len(loaded_records) == 2

    for lr, ex in zip(loaded_records, dataset):
        assert lr["example_id"] == ex.example_id
        assert lr["heart_rate_bpm"] == ex.heart_rate_bpm
        assert lr["sample_count"] == ex.sample_count()
        # Verify waveform file is loadable and matches.
        waveform = np.load(lr["waveform_path"])
        assert np.array_equal(waveform, ex.signal)
