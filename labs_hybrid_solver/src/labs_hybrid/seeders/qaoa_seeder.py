from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..labs_energy_cpu import labs_energy_batch
from ..rng import bits01_to_pm1, get_np_rng, sample_pm1


def _labs_proxy_pairs(N: int, max_lag: int) -> list[tuple[int, int]]:
    """Heuristic 2-local proxy for the LABS objective.

    True LABS contains 4-local terms after squaring autocorrelations.
    For a lightweight circuit baseline, we use a 2-local proxy coupling Z_i Z_{i+k}
    for k up to `max_lag`.
    """

    pairs: list[tuple[int, int]] = []
    L = int(min(max_lag, N - 1))
    for k in range(1, L + 1):
        for i in range(0, N - k):
            pairs.append((i, i + k))
    return pairs


def _try_import_cudaq():
    try:
        import cudaq  # type: ignore

        return cudaq
    except Exception:
        return None


def _set_cudaq_target(cudaq: Any, backend: str) -> str:
    """Best-effort target selection."""

    # CUDA-Q default is a CPU simulator; on many systems `cudaq.set_target('nvidia')`
    # enables GPU simulation.
    try:
        if backend == "gpu":
            cudaq.set_target("nvidia")
            return "nvidia"
        if backend == "cpu":
            cudaq.set_target("qpp")
            return "qpp"
    except Exception:
        pass
    return "default"


def _counts_to_bitstrings(counts: Any, N: int) -> list[str]:
    # `counts` can be dict-like or SampleResult. Ensure deterministic expansion order.
    items = list(counts.items())
    items.sort(key=lambda kv: kv[0])
    out: list[str] = []
    for b, c in items:
        s = str(b).replace(" ", "")
        s = s.zfill(N)
        out.extend([s] * int(c))
    return out


def _bitstrings_to_pm1(bitstrings: list[str], N: int, bit_order: str = "as_returned") -> np.ndarray:
    arr = np.zeros((len(bitstrings), N), dtype=np.int8)
    for r, s in enumerate(bitstrings):
        s = s.zfill(N)
        if bit_order == "reverse":
            s = s[::-1]
        arr[r] = np.fromiter((1 if ch == "1" else 0 for ch in s), dtype=np.int8, count=N)
    return bits01_to_pm1(arr)


def _fallback_classical_qaoa_like(N: int, shots: int, rng_seed: int, params: dict) -> np.ndarray:
    """Runnable fallback if CUDA-Q isn't available.

    Produces valid seeds and applies a tiny deterministic local 'smoothing' under a 2-local proxy.
    """

    rng = get_np_rng(rng_seed)
    seeds = sample_pm1((shots, N), rng)
    max_lag = int(params.get("max_lag", min(6, N - 1)))
    pairs = _labs_proxy_pairs(N, max_lag)

    # One deterministic sweep: try flipping each bit once if it lowers proxy cost.
    def proxy_cost(x: np.ndarray) -> int:
        # H = sum Z_i Z_j; lower is better (more anti-correlated).
        # For ±1, Z_i Z_j -> s_i s_j.
        val = 0
        for i, j in pairs:
            val += int(x[i]) * int(x[j])
        return val

    for r in range(shots):
        x = seeds[r].copy()
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
    """QAOA seeder (canonical baseline).

    Notes:
      - This implementation uses a lightweight 2-local proxy for LABS in the cost layer.
      - When CUDA-Q is not available, it falls back to a deterministic classical sampler.
    """

    t0 = time.perf_counter()
    cudaq = _try_import_cudaq()
    p = int(params.get("p", 1))
    max_lag = int(params.get("max_lag", min(6, N - 1)))
    bit_order = str(params.get("bit_order", "as_returned"))

    if cudaq is None:
        seeds = _fallback_classical_qaoa_like(int(N), int(shots), int(rng_seed), params)
        seeds = seeds[: int(K_out)]
        E = labs_energy_batch(seeds)
        info = {
            "seeder": "qaoa",
            "backend": backend,
            "cudaq_available": False,
            "compile_s": 0.0,
            "sampling_s": time.perf_counter() - t0,
            "shots": int(shots),
            "K_out": int(K_out),
            "raw_energy_min": int(E.min()),
            "raw_energy_mean": float(E.mean()),
            "params": {"p": p, "max_lag": max_lag, "bit_order": bit_order},
            "notes": "CUDA-Q not found; using fallback sampler.",
        }
        return seeds.astype(np.int8, copy=False), info

    # CUDA-Q path
    target = _set_cudaq_target(cudaq, backend)

    cudaq_seeded = False
    try:
        if hasattr(cudaq, "set_random_seed"):
            cudaq.set_random_seed(int(rng_seed))
            cudaq_seeded = True
    except Exception:
        cudaq_seeded = False

    rng = get_np_rng(rng_seed)
    # Deterministic angles from rng_seed unless provided.
    gamma = float(params.get("gamma", float(rng.uniform(0.2, 1.2))))
    beta = float(params.get("beta", float(rng.uniform(0.2, 1.2))))

    pairs = _labs_proxy_pairs(int(N), max_lag)

    # Build kernel dynamically.
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(int(N))

    # Initialize uniform superposition.
    kernel.h(qubits)

    def zz_evolve(theta: float, i: int, j: int):
        # exp(-i theta Z_i Z_j) via CX-RZ-CX (only default ops required)
        kernel.cx(control=qubits[i], target=qubits[j])
        kernel.rz(2.0 * theta, qubits[j])
        kernel.cx(control=qubits[i], target=qubits[j])

    for _layer in range(p):
        for i, j in pairs:
            zz_evolve(gamma, i, j)
        # Mixer: exp(-i beta X) = Rx(2beta)
        for q in range(int(N)):
            kernel.rx(2.0 * beta, qubits[q])

    for q in range(int(N)):
        kernel.mz(qubits[q])

    # Compile warm-up
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
        "seeder": "qaoa",
        "backend": backend,
        "cudaq_available": True,
        "cudaq_target": target,
        "cudaq_seeded": cudaq_seeded,
        "compile_s": compile_s,
        "sampling_s": sampling_s,
        "shots": int(shots),
        "K_out": int(K_out),
        "qubits": int(N),
        "depth_proxy": int(p) * (len(pairs) + int(N)),
        "raw_energy_min": int(E.min()),
        "raw_energy_mean": float(E.mean()),
        "params": {"p": p, "gamma": gamma, "beta": beta, "max_lag": max_lag, "bit_order": bit_order},
    }
    return pm1.astype(np.int8, copy=False), info
