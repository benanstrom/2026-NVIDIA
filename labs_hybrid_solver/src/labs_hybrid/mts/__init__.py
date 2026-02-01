from __future__ import annotations

import numpy as np


def run_mts(
    seeds_pm1: np.ndarray,
    *,
    budget_s: float | None,
    max_iters: int | None,
    tabu_params: dict,
    device: str,  # "cpu" | "gpu"
    rng_seed: int,
) -> tuple[np.ndarray, float, dict]:
    """Run the fixed Memetic Tabu Search (dispatch CPU/GPU)."""

    if device == "cpu":
        from .mts_cpu import run_mts_cpu

        return run_mts_cpu(
            seeds_pm1,
            budget_s=budget_s,
            max_iters=max_iters,
            tabu_params=tabu_params,
            rng_seed=rng_seed,
        )
    if device == "gpu":
        from .mts_gpu import run_mts_gpu

        return run_mts_gpu(
            seeds_pm1,
            budget_s=budget_s,
            max_iters=max_iters,
            tabu_params=tabu_params,
            rng_seed=rng_seed,
        )
    raise ValueError(f"Unknown device for MTS: {device}")
