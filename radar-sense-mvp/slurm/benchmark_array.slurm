#!/bin/bash
# Minimal SLURM template for benchmark evaluation.

#SBATCH --job-name=radar-sense-benchmark
#SBATCH --output=logs/benchmark-%A_%a.out
#SBATCH --array=0-2

python scripts/run_benchmark.py
