from __future__ import annotations

import time

import numpy as np

from ..labs_energy_cpu import labs_energy_batch


def run_mts_cpu(
    seeds_pm1: np.ndarray,
    *,
    budget_s: float | None,
    max_iters: int | None,
    tabu_params: dict,
    rng_seed: int,
) -> tuple[np.ndarray, float, dict]:
    """Fixed CPU Memetic Tabu Search (MTS).

    This is the correctness oracle for the GPU path.
    The implementation intentionally avoids seeder-specific heuristics.
    """

    t0 = time.perf_counter()

    pop = np.array(seeds_pm1, dtype=np.int8, copy=True)
    K, N = pop.shape

    tenure = int(tabu_params.get("tenure", 7))
    aspiration = bool(tabu_params.get("aspiration", True))
    stagnation_window = tabu_params.get("stagnation_window", None)
    if stagnation_window is not None:
        stagnation_window = int(stagnation_window)

    tabu = np.zeros((K, N), dtype=np.int32)

    E_cur = labs_energy_batch(pop)
    best_per = pop.copy()
    E_best_per = E_cur.copy()

    best_idx = int(np.argmin(E_cur))
    best_global_seq = pop[best_idx].copy()
    best_global_E = int(E_cur[best_idx])

    trace_best = [best_global_E]
    stagnation_count = 0

    it = 0
    while True:
        it += 1
        if max_iters is not None and it > int(max_iters):
            break
        if budget_s is not None and (time.perf_counter() - t0) >= float(budget_s):
            break

        # Full single-bit-flip neighborhood as (K*N, N)
        neigh = np.repeat(pop, N, axis=0)
        rows = np.arange(K * N)
        cols = np.tile(np.arange(N), K)
        neigh[rows, cols] *= -1

        E_neigh = labs_energy_batch(neigh).reshape(K, N)

        if tenure > 0:
            blocked = tabu > 0
            if aspiration:
                aspir = E_neigh < E_best_per[:, None]
                blocked = blocked & (~aspir)
            E_masked = E_neigh.astype(np.int64, copy=True)
            E_masked[blocked] = np.iinfo(np.int64).max
        else:
            E_masked = E_neigh

        move_pos = np.argmin(E_masked, axis=1)
        move_E = E_neigh[np.arange(K), move_pos]

        if tenure > 0:
            tabu = np.maximum(tabu - 1, 0)
            tabu[np.arange(K), move_pos] = tenure

        pop[np.arange(K), move_pos] *= -1
        E_cur = move_E

        improved = E_cur < E_best_per
        if np.any(improved):
            best_per[improved] = pop[improved]
            E_best_per[improved] = E_cur[improved]

        cur_best_idx = int(np.argmin(E_best_per))
        cur_best_E = int(E_best_per[cur_best_idx])
        if cur_best_E < best_global_E:
            best_global_E = cur_best_E
            best_global_seq = best_per[cur_best_idx].copy()
            stagnation_count = 0
        else:
            stagnation_count += 1

        trace_best.append(best_global_E)

        if stagnation_window is not None and stagnation_count >= stagnation_window:
            break

    early_exit = (stagnation_window is not None and stagnation_count >= stagnation_window)
    t_total = time.perf_counter() - t0
    logs = {
        "device": "cpu",
        "iters": it,
        "time_total_s": t_total,
        "best_trace": trace_best,
        "best_energy": float(best_global_E),
        "early_exit_stagnation": early_exit,
    }
    return best_global_seq, float(best_global_E), logs
