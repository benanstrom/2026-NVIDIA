"""Benchmark: RawKernel vs loop-based GPU LABS energy.

Run on a CUDA machine:
    python bench_energy_kernel.py
"""

from __future__ import annotations

import time

import cupy as cp
import numpy as np

from labs_hybrid.labs_energy_gpu import (
    labs_energy_batch_gpu,
    _labs_energy_batch_gpu_fallback,
)

WARMUP = 5
REPEATS = 50


def bench(fn, seqs_gpu, warmup=WARMUP, repeats=REPEATS):
    # warmup
    for _ in range(warmup):
        fn(seqs_gpu)
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(repeats):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        fn(seqs_gpu)
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def main():
    sizes = [(1000, 40), (4096, 40), (1000, 64)]

    for K, N in sizes:
        rng = np.random.default_rng(0)
        seeds = rng.integers(0, 2, size=(K, N), dtype=np.int8) * 2 - 1
        seqs_gpu = cp.asarray(seeds, dtype=cp.int8)

        # Verify parity before benchmarking
        E_raw = cp.asnumpy(labs_energy_batch_gpu(seqs_gpu))
        E_loop = cp.asnumpy(_labs_energy_batch_gpu_fallback(seqs_gpu))
        assert np.array_equal(E_raw, E_loop), "Parity check failed!"

        t_raw = bench(labs_energy_batch_gpu, seqs_gpu)
        t_loop = bench(_labs_energy_batch_gpu_fallback, seqs_gpu)

        med_raw = sorted(t_raw)[len(t_raw) // 2] * 1e6
        med_loop = sorted(t_loop)[len(t_loop) // 2] * 1e6
        speedup = med_loop / med_raw if med_raw > 0 else float("inf")

        print(f"K={K:>5}, N={N:>3}  |  "
              f"loop: {med_loop:8.1f} us  |  "
              f"raw:  {med_raw:8.1f} us  |  "
              f"speedup: {speedup:.1f}x")


if __name__ == "__main__":
    main()
