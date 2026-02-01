from __future__ import annotations

import time

import numpy as np

from ..rng import get_np_rng, sample_pm1


def generate_seeds(
    N: int,
    shots: int,
    K_out: int,
    *,
    backend: str,  # "cpu" | "gpu" | "auto" (ignored here)
    rng_seed: int,
    params: dict,
) -> tuple[np.ndarray, dict]:
    """Random seeder baseline.

    Returns K_out iid random ±1 sequences.
    """

    t0 = time.perf_counter()
    rng = get_np_rng(rng_seed)

    K = int(K_out)
    seeds = sample_pm1((K, int(N)), rng)

    info = {
        "seeder": "random",
        "backend": backend,
        "rng_seed": int(rng_seed),
        "compile_s": 0.0,
        "sampling_s": time.perf_counter() - t0,
        "shots": int(shots),
        "K_out": int(K_out),
        "notes": "IID random ±1 baseline",
    }
    return seeds, info
