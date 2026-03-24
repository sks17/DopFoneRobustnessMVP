#!/bin/bash
# Minimal SLURM template for synthetic dataset generation.

#SBATCH --job-name=radar-sense-generate
#SBATCH --output=logs/generate-%A_%a.out
#SBATCH --array=0-2

python scripts/build_clean_dataset.py
