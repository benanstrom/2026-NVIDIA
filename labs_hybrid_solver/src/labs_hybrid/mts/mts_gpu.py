from __future__ import annotations

import time

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None  # type: ignore

from ..labs_energy_gpu import labs_energy_batch_gpu


def run_mts_gpu(
    seeds_pm1: np.ndarray,
    *,
    budget_s: float | None,
    max_iters: int | None,
    tabu_params: dict,
    rng_seed: int,
) -> tuple[np.ndarray, float, dict]:
    """Fixed GPU MTS: batched multi-start tabu search with minimal host transfer.

    Zhang et al. motivates the structure (multi-start + batched evaluation),
    but this implementation avoids custom CUDA kernels and seeder-specific heuristics.
    """

    if cp is None:
        raise RuntimeError("CuPy not available")

    t0 = time.perf_counter()

    pop = cp.asarray(seeds_pm1, dtype=cp.int8)
    K, N = pop.shape

    tenure = int(tabu_params.get("tenure", 7))
    aspiration = bool(tabu_params.get("aspiration", True))
    trace_stride = int(tabu_params.get("trace_stride", 5))

    tabu = cp.zeros((K, N), dtype=cp.int32)

    E_cur = labs_energy_batch_gpu(pop)
    best_per = pop.copy()
    E_best_per = E_cur.copy()

    best_idx = int(cp.asnumpy(cp.argmin(E_best_per)))
    best_global_seq = cp.asnumpy(best_per[best_idx]).astype(np.int8, copy=False)
    best_global_E = float(cp.asnumpy(E_best_per[best_idx]))

    trace_best = [best_global_E]

    it = 0
    big = cp.int64(cp.iinfo(cp.int64).max)
    ar = cp.arange(K)
    while True:
        it += 1
        if max_iters is not None and it > int(max_iters):
            break
        if budget_s is not None and (time.perf_counter() - t0) >= float(budget_s):
            break

        neigh = cp.repeat(pop, N, axis=0)
        rows = cp.arange(K * N)
        cols = cp.tile(cp.arange(N), K)
        neigh[rows, cols] *= -1

        E_neigh = labs_energy_batch_gpu(neigh).reshape(K, N)

        if tenure > 0:
            blocked = tabu > 0
            if aspiration:
                aspir = E_neigh < E_best_per[:, None]
                blocked = blocked & (~aspir)
            E_masked = E_neigh.astype(cp.int64, copy=True)
            E_masked[blocked] = big
        else:
            E_masked = E_neigh

        move_pos = cp.argmin(E_masked, axis=1)
        move_E = E_neigh[ar, move_pos]

        if tenure > 0:
            tabu = cp.maximum(tabu - 1, 0)
            tabu[ar, move_pos] = tenure

        pop[ar, move_pos] *= -1
        E_cur = move_E

        improved = E_cur < E_best_per
        if bool(cp.any(improved)):
            best_per[improved] = pop[improved]
            E_best_per[improved] = E_cur[improved]

        if it % trace_stride == 0:
            idx = cp.argmin(E_best_per)
            E_best = E_best_per[idx]
            E_host = float(cp.asnumpy(E_best))
            if E_host < best_global_E:
                best_global_E = E_host
                best_global_seq = cp.asnumpy(best_per[idx]).astype(np.int8, copy=False)
            trace_best.append(best_global_E)

    idx = cp.argmin(E_best_per)
    E_best = float(cp.asnumpy(E_best_per[idx]))
    if E_best < best_global_E:
        best_global_E = E_best
        best_global_seq = cp.asnumpy(best_per[idx]).astype(np.int8, copy=False)

    t_total = time.perf_counter() - t0
    logs = {
        "device": "gpu",
        "iters": it,
        "time_total_s": t_total,
        "trace_stride": trace_stride,
        "best_trace": trace_best,
        "best_energy": best_global_E,
    }
    return best_global_seq, float(best_global_E), logs
