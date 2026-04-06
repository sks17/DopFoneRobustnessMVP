# RADAR-Sense-MVP

RADAR-Sense-MVP is a research MVP for benchmarking the robustness of DopFone-like smartphone fetal-heart-rate sensing under realistic signal artifacts. The project takes inspiration from two places:

- `DopFone`: smartphone-based fetal heart-rate sensing using an approximately 18 kHz Doppler-like audio setup.
- `RADAR`: a benchmark philosophy where clean source examples are paired with structured perturbations so failure modes can be measured directly instead of inferred informally.

The core idea is simple: start with synthetic signals whose ground-truth heart rate is known exactly, apply controlled corruptions such as attenuation, dropout, and noise, then measure how much a simple estimator degrades as the corruption type and severity change.

## What This Project Is Trying To Solve

DopFone-style sensing is interesting because it could make fetal monitoring more accessible, but it is also fragile. In real usage, signal quality can change because of placement, coupling, motion, attenuation, or missing segments. A model may appear to work in a small, clean study and still fail badly once the signal is imperfect.

This repository addresses that gap by turning robustness into a measurable benchmark problem:

1. Build clean, labeled synthetic waveforms.
2. Expand them into a perturbation grid.
3. Run a transparent estimator on both clean and perturbed data.
4. Report error statistics overall and by artifact type and severity.

The current estimator is intentionally weak and interpretable. The goal of the MVP is not state-of-the-art performance; it is to create a disciplined evaluation harness that makes failure visible.

## End-To-End Architecture

The pipeline is organized into five layers.

### 1. Signal timing and heartbeat structure

`src/simulation/heartbeat.py` creates the temporal skeleton of the signal:

- converts BPM to beat interval
- generates nominal beat timestamps over a duration
- optionally adds small physiological jitter
- builds a sampled time axis
- converts beat timestamps into a pulse train

This layer defines when heartbeats occur before any Doppler-like carrier is introduced.

### 2. Doppler-like waveform synthesis

`src/simulation/doppler_like.py` turns heartbeat timing into a synthetic Doppler-like waveform:

- a pulse train is smoothed into a heartbeat envelope
- a sinusoidal carrier is generated
- the carrier is amplitude-modulated by the heartbeat envelope
- optional amplitude scaling and baseline noise are applied

This gives a synthetic observed waveform that behaves like a simplified DopFone-style signal.

### 3. Dataset generation

`src/simulation/generator.py` and `src/benchmark/dataset_builder.py` build benchmark-ready examples:

- clean examples are generated with known `heart_rate_bpm`
- clean examples are serialized to manifest records
- each clean example can be expanded across perturbation type x severity x seed

This is the key RADAR-style step: the objective label stays fixed while corruption changes, so error under corruption can be measured directly.

### 4. Perturbation layer

`src/perturbations/` provides structured artifact generation:

- `noise.py`: additive Gaussian noise
- `dropout.py`: contiguous missing segments
- `attenuation.py`: scalar amplitude reduction
- `registry.py`: both explicit-parameter and severity-based dispatch

Severity is normalized to `[0, 1]`, which makes it easy to run compact local sweeps or much larger HPC sweeps with the same interface.

### 5. Estimation and benchmark evaluation

The evaluation stack lives in `src/estimation/` and `src/benchmark/`:

- `preprocess.py` loads `.npy` waveforms, normalizes them, filters them, and extracts amplitude envelopes
- `peak_estimator.py` implements a simple interpretable FHR estimator using smoothing, thresholded peak detection, interval extraction, and median-interval BPM estimation
- `runner.py` evaluates clean and perturbed manifests and produces `BenchmarkResult` rows
- `metrics.py` computes absolute error, MAE, grouped MAE by artifact type, and grouped MAE by severity

The benchmark result format is designed so rows can be written out, reloaded, and re-summarized without rerunning the estimator.

## Data Flow In Practice

At a high level, the repository works like this:

1. Generate clean waveforms and write them to `.npy` files.
2. Write a clean JSONL manifest containing metadata and waveform paths.
3. Expand the clean set into perturbed waveforms and write a perturbed JSONL manifest.
4. Read both manifests and evaluate every waveform with the estimator.
5. Write benchmark result rows as JSONL or CSV.
6. Compute summary statistics:
   - overall MAE
   - clean MAE
   - perturbed MAE
   - MAE grouped by artifact type
   - MAE grouped by severity

That separation is deliberate:

- signal generation and perturbation are pure data-creation steps
- estimation is a pure model-evaluation step
- scripts handle file I/O and orchestration

This keeps the code compositional and makes it easier to swap in a stronger estimator later without rewriting the benchmark.

## Repository Layout

- `configs/`: YAML configs for generation, benchmark paths, and testing expectations
- `data/`: generated waveforms, manifests, and summary outputs
- `docs/`: architectural notes and specification stubs
- `src/datatypes/`: explicit benchmark datatypes such as `SignalExample` and `BenchmarkResult`
- `src/simulation/`: heartbeat timing and Doppler-like signal synthesis
- `src/perturbations/`: structured corruption functions and registry logic
- `src/estimation/`: preprocessing and the MVP peak-based estimator
- `src/benchmark/`: dataset expansion, metrics, and benchmark runner logic
- `src/utils/`: YAML/JSON/JSONL I/O, waveform persistence, logging, seeding
- `scripts/`: manifest builders, benchmark execution, and summary recomputation
- `tests/`: unit and small end-to-end tests
- `slurm/`: simple templates for scaling generation and benchmarking on HPC

## How To Run It Locally

From `radar-sense-mvp/`:

1. Install dependencies.
2. Run the tests.
3. Build clean and perturbed manifests.
4. Run the benchmark.
5. Recompute or print the summary.

Typical commands:

```powershell
python -m pip install -r requirements.txt
python -m pytest
python scripts/build_clean_dataset.py
python scripts/build_perturbed_dataset.py
python scripts/run_benchmark.py
python scripts/summarize_results.py
```

For a single local orchestration script that produces the main findings:

```powershell
python run_local_master_stats.py
```

That root-level script is intentionally lightweight. It is meant to give a realistic local feel for the benchmark outputs before launching much larger HPC sweeps.

## Key Output Files

Important outputs are written under `data/`:

- `data/manifests/clean_manifest.jsonl`: metadata for clean examples
- `data/manifests/perturbed_manifest.jsonl`: metadata for perturbed examples
- `data/manifests/benchmark_results.jsonl`: per-example benchmark result rows
- `data/manifests/benchmark_summary.json`: aggregated benchmark metrics
- `data/waveforms/clean/`: clean waveform `.npy` files
- `data/perturbed/`: perturbed waveform `.npy` files

The summary file is the compact “final findings” artifact for a run.

## Why The Current Estimator Looks Weak

The estimator is intentionally simple:

- it extracts an envelope
- smooths it
- detects peaks
- converts peak spacing into BPM

This is useful because it is easy to inspect and easy to break. When the benchmark shows large errors, that is not a bug in the idea of benchmarking; it is the point of the benchmark. The MVP is designed to reveal when a simple method is:

- overly sensitive to attenuation
- unable to handle missing segments
- accidentally helped or confused by noise
- reacting to artifacts instead of robust heartbeat structure

That is exactly the kind of evidence needed before investing in more sophisticated models or larger-scale studies.

## Why This Design Matters

The most important contribution of the project is methodological:

- known labels instead of uncertain labels
- controlled perturbations instead of vague robustness claims
- manifest-driven evaluation instead of ad hoc scripts
- grouped error analysis instead of one headline number

In other words, the repository is not just “a synthetic signal generator” and not just “a weak estimator.” It is a compact robustness benchmark scaffold that can support better estimators and larger experiments later.

## Current Status

The project currently supports:

- synthetic clean signal generation
- structured perturbation sweeps
- manifest-driven evaluation
- grouped robustness metrics
- local summary extraction
- scaling paths toward larger HPC trials

It is best understood as benchmark infrastructure first, sensing model second.

## Acknowledgment

The robustness framing is inspired in part by the RADAR benchmark paper:

```bibtex
@article{gu2025radar,
  title={RADAR: Benchmarking Language Models on Imperfect Tabular Data},
  author={Ken Gu and Zhihan Zhang and Kate Lin and Yuwei Zhang and Akshay Paruchuri and Hong Yu and Mehran Kazemi and Kumar Ayush and A. Ali Heydari and Maxwell A. Xu and Girish Narayanswamy and Yun Liu and Ming-Zher Poh and Yuzhe Yang and Mark Malhotra and Shwetak Patel and Hamid Palangi and Xuhai Xu and Daniel McDuff and Tim Althoff and Xin Liu},
  year={2025},
  eprint={2506.08249},
  archivePrefix={arXiv},
  primaryClass={cs.DB},
  url={https://arxiv.org/abs/2506.08249}
}
```
