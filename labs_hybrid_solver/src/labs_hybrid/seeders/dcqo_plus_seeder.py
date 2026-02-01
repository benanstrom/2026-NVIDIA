from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

from ..labs_energy_cpu import labs_energy_batch
from ..rng import get_np_rng, sample_pm1

from .qaoa_seeder import (
    _bitstrings_to_pm1,
    _counts_to_bitstrings,
    _labs_proxy_pairs,
    _set_cudaq_target,
    _try_import_cudaq,
)


def _ordering_signature(order: np.ndarray) -> str:
    """Small deterministic fingerprint of an ordering for tests/logging."""

    take = np.concatenate([order[:10], order[-10:]]) if order.size > 20 else order
    h = hashlib.sha256(take.tobytes()).hexdigest()
    return h[:16]


def _compute_trotter_meta(
    *,
    N: int,
    steps: int,
    pairs: list[tuple[int, int]],
    time_grid: str,
    ordering: str,
    K_perm: int,
    rng_seed: int,
) -> dict[str, Any]:
    pair_count = len(pairs)

    if time_grid not in ("right_endpoint", "midpoint"):
        raise ValueError(f"Unknown time_grid: {time_grid}")
    if ordering not in ("fixed", "forward_reverse", "k_permutation"):
        raise ValueError(f"Unknown ordering: {ordering}")

    if time_grid == "right_endpoint":
        t_points = np.array([(j + 1) / float(steps) for j in range(steps)], dtype=np.float64)
    else:
        t_points = np.array([(j + 0.5) / float(steps) for j in range(steps)], dtype=np.float64)

    base_order = np.arange(pair_count, dtype=np.int32)
    rng = get_np_rng(rng_seed)

    if ordering == "fixed":
        orders = [base_order.copy() for _ in range(steps)]
    elif ordering == "forward_reverse":
        rev = base_order[::-1].copy()
        orders = [base_order.copy() if (j % 2 == 0) else rev.copy() for j in range(steps)]
    else:
        if K_perm < 1:
            raise ValueError("K_perm must be >= 1")
        perms = [rng.permutation(base_order) for _ in range(K_perm)]
        orders = [perms[j % K_perm].copy() for j in range(steps)]

    order_sigs = [_ordering_signature(o) for o in orders]

    # For tests/debugging: store a preview of each ordering (full ordering if small).
    previews = []
    full_orders = []
    for o in orders:
        previews.append({"first20": o[:20].tolist(), "last20": o[-20:].tolist()})
        if pair_count <= 128:
            full_orders.append(o.tolist())

    meta: dict[str, Any] = {
        "time_grid": time_grid,
        "ordering": ordering,
        "K_perm": K_perm,
        "t_points": t_points.tolist(),
        "order_sigs": order_sigs,
        "pair_count": pair_count,
        "depth_proxy": steps * (int(N) + pair_count),
        "orders_preview": previews,
    }
    if full_orders:
        meta["orders_full"] = full_orders
    return meta


def _fallback_classical(N: int, shots: int, rng_seed: int) -> np.ndarray:
    rng = get_np_rng(rng_seed)
    return sample_pm1((shots, N), rng)


def generate_seeds(
    N: int,
    shots: int,
    K_out: int,
    *,
    backend: str,
    rng_seed: int,
    params: dict,
) -> tuple[np.ndarray, dict]:
    """DCQO+ seeder: DCQO with *zero added depth* trotterization upgrades.

    Upgrades (must not change term set or depth):
      - time_grid: right_endpoint -> midpoint
      - ordering: fixed -> forward_reverse | k_permutation

    Baseline behavior when knobs are off:
      - Lie–Trotter (first-order)
      - right-endpoint schedule sampling
      - fixed Pauli ordering
    """

    t0 = time.perf_counter()
    cudaq = _try_import_cudaq()

    steps = int(params.get("steps", 24))
    max_lag = int(params.get("max_lag", min(6, N - 1)))
    total_time = float(params.get("total_time", 1.0))
    bit_order = str(params.get("bit_order", "as_returned"))

    trotter_config = params.get("trotter_config", {}) or {}
    time_grid = str(trotter_config.get("time_grid", "right_endpoint"))
    ordering = str(trotter_config.get("ordering", "fixed"))
    K_perm = int(trotter_config.get("K_perm", 1))

    pairs = _labs_proxy_pairs(int(N), max_lag)
    trotter_meta = _compute_trotter_meta(
        N=int(N),
        steps=steps,
        pairs=pairs,
        time_grid=time_grid,
        ordering=ordering,
        K_perm=K_perm,
        rng_seed=int(rng_seed),
    )

    if cudaq is None:
        seeds = _fallback_classical(int(N), int(shots), int(rng_seed))[: int(K_out)]
        E = labs_energy_batch(seeds)
        info = {
            "seeder": "dcqo_plus",
            "backend": backend,
            "cudaq_available": False,
            "compile_s": 0.0,
            "sampling_s": time.perf_counter() - t0,
            "shots": int(shots),
            "K_out": int(K_out),
            "raw_energy_min": int(E.min()),
            "raw_energy_mean": float(E.mean()),
            "params": {
                "steps": steps,
                "max_lag": max_lag,
                "total_time": total_time,
                "bit_order": bit_order,
                "trotter_config": {"time_grid": time_grid, "ordering": ordering, "K_perm": K_perm},
            },
            "trotter_meta": trotter_meta,
            "notes": "CUDA-Q not found; no circuit built.",
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

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(int(N))
    kernel.h(qubits)

    def zz_evolve(theta: float, i: int, j: int):
        kernel.cx(control=qubits[i], target=qubits[j])
        kernel.rz(2.0 * theta, qubits[j])
        kernel.cx(control=qubits[i], target=qubits[j])

    dt = total_time / float(steps)
    pair_list = pairs
    t_points = trotter_meta["t_points"]
    orders = trotter_meta.get("orders_full")
    if orders is None:
        # Reconstruct from previews if full list too large.
        orders = None

    # Build the trotterized evolution circuit following the precomputed schedule.
    base_order = np.arange(len(pair_list), dtype=np.int32)
    for j in range(steps):
        s = float(t_points[j])
        a = 1.0 - s
        b = s

        for q in range(int(N)):
            kernel.rx(2.0 * dt * a, qubits[q])

        # Derive the ordering for this step.
        if ordering == "fixed":
            order_idx = base_order
        elif ordering == "forward_reverse":
            order_idx = base_order if (j % 2 == 0) else base_order[::-1]
        else:
            # k_permutation: use the signature list to reproduce deterministic permutations via RNG.
            # We reuse the same precomputed meta; orders_full exists for tests and small N.
            if trotter_meta.get("orders_full"):
                order_idx = np.array(trotter_meta["orders_full"][j], dtype=np.int32)
            else:
                # For large cases, fall back to a single RNG permutation per step.
                order_idx = get_np_rng(int(rng_seed) + j).permutation(base_order)

        for idx in order_idx:
            i, k = pair_list[int(idx)]
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
    pm1 = _bitstrings_to_pm1(bitstrings, int(N), bit_order=bit_order)[: int(K_out)]

    E = labs_energy_batch(pm1)

    info = {
        "seeder": "dcqo_plus",
        "backend": backend,
        "cudaq_available": True,
        "cudaq_target": target,
        "cudaq_seeded": cudaq_seeded,
        "compile_s": compile_s,
        "sampling_s": sampling_s,
        "shots": int(shots),
        "K_out": int(K_out),
        "qubits": int(N),
        "depth_proxy": int(trotter_meta["depth_proxy"]),
        "raw_energy_min": int(E.min()),
        "raw_energy_mean": float(E.mean()),
        "params": {
            "steps": steps,
            "max_lag": max_lag,
            "total_time": total_time,
            "bit_order": bit_order,
            "trotter_config": {"time_grid": time_grid, "ordering": ordering, "K_perm": K_perm},
        },
        "trotter_meta": trotter_meta,
    }
    return pm1.astype(np.int8, copy=False), info
