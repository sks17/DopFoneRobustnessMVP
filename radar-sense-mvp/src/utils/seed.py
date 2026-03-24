"""Deterministic seeding helpers."""

from __future__ import annotations

import numpy as np


# Spec:
# - General description: Create a NumPy random generator from an integer seed.
# - Params: `seed`, integer seed value.
# - Pre: `seed >= 0`.
# - Post: Returns a deterministic `np.random.Generator`.
# - Mathematical definition: rng = Generator(PCG64(seed)).
def build_rng(seed: int) -> np.random.Generator:
    """Return a deterministic NumPy random generator."""
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    return np.random.default_rng(seed)
