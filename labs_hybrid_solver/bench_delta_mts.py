#!/usr/bin/env python
"""Benchmark: delta-energy GPU MTS vs naive (full-eval) GPU MTS."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import time
import numpy as np

try:
    import cupy as cp
except ImportError:
    sys.exit("CuPy not available — GPU benchmark cannot run.")

from labs_hybrid.mts.mts_gpu import run_mts_gpu, _run_mts_gpu_naive


CONFIGS = [
    (1000, 40),
    (4096, 40),
    (4096, 64),
]
ITERS = 100
TABU_PARAMS = {"tenure": 7, "aspiration": True, "trace_stride": ITERS + 1}


def bench(fn, seeds, iters, repeats=3):
    """Run fn `repeats` times and return median wall-clock seconds."""
    times = []
    for _ in range(repeats):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        fn(
            seeds.copy(),
            budget_s=None,
            max_iters=iters,
            tabu_params=TABU_PARAMS,
            rng_seed=0,
        )
        cp.cuda.Device().synchronize()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def main():
    print(f"{'K':>6}  {'N':>4}  {'Naive (s)':>10}  {'Delta (s)':>10}  {'Speedup':>8}")
    print("-" * 50)

    for K, N in CONFIGS:
        rng = np.random.default_rng(0)
        seeds = (rng.integers(0, 2, size=(K, N)) * 2 - 1).astype(np.int8)

        t_naive = bench(_run_mts_gpu_naive, seeds, ITERS)
        t_delta = bench(run_mts_gpu, seeds, ITERS)

        speedup = t_naive / t_delta if t_delta > 0 else float("inf")
        print(f"{K:>6}  {N:>4}  {t_naive:>10.4f}  {t_delta:>10.4f}  {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
