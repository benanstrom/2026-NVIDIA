from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..labs_energy_cpu import labs_energy_batch
from ..rng import bits01_to_pm1, get_np_rng, sample_pm1

from .qaoa_seeder import _counts_to_bitstrings, _labs_proxy_pairs, _bitstrings_to_pm1, _set_cudaq_target, _try_import_cudaq


def _fallback_classical_dcqo_like(N: int, shots: int, rng_seed: int, params: dict) -> np.ndarray:
    """Runnable fallback if CUDA-Q isn't available.

    A very lightweight 'digitized evolution' surrogate: repeated deterministic proxy sweeps
    (up to 3 passes of greedy single-bit flips under a 2-local proxy cost).

    **Limitation:** The 2-local proxy (Σ Z_i Z_{i+k} for k ≤ max_lag) is a crude
    approximation of the true 4-local LABS objective. With only 3 sweeps, seed quality
    is marginal — comparable to lightly improved random sampling. This is acceptable
    because the fallback only activates when CUDA-Q is not installed (no GPU available),
    typically in test/CI environments where seed quality is not critical.
    """

    rng = get_np_rng(rng_seed)
    seeds = sample_pm1((shots, N), rng)
    steps = int(params.get("steps", 24))
    max_lag = int(params.get("max_lag", min(6, N - 1)))
    pairs = _labs_proxy_pairs(N, max_lag)

    def proxy_cost(x: np.ndarray) -> int:
        val = 0
        for i, j in pairs:
            val += int(x[i]) * int(x[j])
        return val

    # Repeat a small number of proxy-improvement sweeps (bounded to keep cost small).
    sweeps = min(3, max(1, steps // 8))
    for r in range(shots):
        x = seeds[r].copy()
        for _ in range(sweeps):
            base = proxy_cost(x)
            for i in range(N):
                x[i] *= -1
                new = proxy_cost(x)
                if new <= base:
                    base = new
                else:
                    x[i] *= -1
        seeds[r] = x
    return seeds


def generate_seeds(
    N: int,
    shots: int,
    K_out: int,
    *,
    backend: str,
    rng_seed: int,
    params: dict,
) -> tuple[np.ndarray, dict]:
    """DCQO seeder (Digitized Counterdiabatic Quantum Optimization baseline).

    Implementation note:
      - We implement a digitized time-dependent evolution circuit with Lie–Trotter (first order)
        and right-endpoint time sampling.
      - The underlying Hamiltonian uses a 2-local proxy for LABS for circuit tractability.
    """

    t0 = time.perf_counter()
    cudaq = _try_import_cudaq()

    steps = int(params.get("steps", 24))
    max_lag = int(params.get("max_lag", min(6, N - 1)))
    total_time = float(params.get("total_time", 1.0))
    bit_order = str(params.get("bit_order", "as_returned"))

    if cudaq is None:
        seeds = _fallback_classical_dcqo_like(int(N), int(shots), int(rng_seed), params)
        seeds = seeds[: int(K_out)]
        E = labs_energy_batch(seeds)
        info = {
            "seeder": "dcqo",
            "backend": backend,
            "cudaq_available": False,
            "compile_s": 0.0,
            "sampling_s": time.perf_counter() - t0,
            "shots": int(shots),
            "K_out": int(K_out),
            "raw_energy_min": int(E.min()),
            "raw_energy_mean": float(E.mean()),
            "params": {"steps": steps, "max_lag": max_lag, "total_time": total_time, "bit_order": bit_order},
            "notes": "CUDA-Q not found; using classical fallback (3 greedy sweeps under 2-local proxy). Seed quality ~random.",
        }
        return seeds.astype(np.int8, copy=False), info

    target = _set_cudaq_target(cudaq, backend)

    cudaq_seeded = False
    try:
        if hasattr(cudaq, "set_random_seed"):
            cudaq.set_random_seed(int(rng_seed))
            cudaq_seeded = True
    except Exception:
        cudaq_seeded = False

    pairs = _labs_proxy_pairs(int(N), max_lag)

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(int(N))
    kernel.h(qubits)

    def zz_evolve(theta: float, i: int, j: int):
        kernel.cx(control=qubits[i], target=qubits[j])
        kernel.rz(2.0 * theta, qubits[j])
        kernel.cx(control=qubits[i], target=qubits[j])

    dt = total_time / float(steps)
    # Right-endpoint time grid: t_j = (j+1)/steps
    for j in range(steps):
        s = (j + 1) / float(steps)
        a = 1.0 - s
        b = s

        # Lie–Trotter: driver then problem; fixed Pauli ordering (pairs as built)
        for q in range(int(N)):
            kernel.rx(2.0 * dt * a, qubits[q])
        for i, k in pairs:
            zz_evolve(dt * b, i, k)

    for q in range(int(N)):
        kernel.mz(qubits[q])

    t_compile0 = time.perf_counter()
    _ = cudaq.sample(kernel, shots_count=1)
    compile_s = time.perf_counter() - t_compile0

    t_samp0 = time.perf_counter()
    counts = cudaq.sample(kernel, shots_count=int(shots))
    sampling_s = time.perf_counter() - t_samp0

    bitstrings = _counts_to_bitstrings(counts, int(N))
    pm1 = _bitstrings_to_pm1(bitstrings, int(N), bit_order=bit_order)
    pm1 = pm1[: int(K_out)]

    E = labs_energy_batch(pm1)

    info = {
        "seeder": "dcqo",
        "backend": backend,
        "cudaq_available": True,
        "cudaq_target": target,
        "cudaq_seeded": cudaq_seeded,
        "compile_s": compile_s,
        "sampling_s": sampling_s,
        "shots": int(shots),
        "K_out": int(K_out),
        "qubits": int(N),
        "depth_proxy": steps * (int(N) + len(pairs)),
        "raw_energy_min": int(E.min()),
        "raw_energy_mean": float(E.mean()),
        "params": {"steps": steps, "max_lag": max_lag, "total_time": total_time, "bit_order": bit_order},
    }
    return pm1.astype(np.int8, copy=False), info
