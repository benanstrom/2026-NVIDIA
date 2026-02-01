from __future__ import annotations

import time

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None  # type: ignore

from ..labs_energy_gpu import labs_energy_batch_gpu


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Kernel 1 — compute autocorrelations C_k for all K sequences
# ---------------------------------------------------------------------------

_LABS_AUTOCORR_KERNEL_CODE = r"""
extern "C" __global__
void labs_autocorr_kernel(const signed char* seqs,
                          int*               autocorrs,
                          int                K,
                          int                N)
{
    // One block per sequence, one thread per lag (1 .. N-1).
    int seq_idx = blockIdx.x;
    int lag     = threadIdx.x + 1;   // k = 1 .. N-1

    int c_k = 0;
    if (lag < N) {
        const signed char* row = seqs + (long long)seq_idx * N;
        for (int i = 0; i < N - lag; ++i) {
            c_k += row[i] * row[i + lag];
        }
    }

    // Store C_k.  autocorrs layout: (K, N-1), row-major.
    // Thread 0 writes lag=1 at index 0, thread t writes lag=t+1 at index t.
    if (lag < N) {
        autocorrs[(long long)seq_idx * (N - 1) + (lag - 1)] = c_k;
    }
}
"""

_autocorr_kernel = None


def _get_autocorr_kernel():
    global _autocorr_kernel
    if _autocorr_kernel is None:
        _autocorr_kernel = cp.RawKernel(_LABS_AUTOCORR_KERNEL_CODE,
                                        "labs_autocorr_kernel")
    return _autocorr_kernel


# ---------------------------------------------------------------------------
# Kernel 2 — delta-energy: compute E_new(j) for every (seq, flip-position)
# ---------------------------------------------------------------------------

_LABS_DELTA_ENERGY_KERNEL_CODE = r"""
extern "C" __global__
void labs_delta_energy_kernel(const signed char* seqs,
                              const int*         autocorrs,
                              long long*         delta_energies,
                              int                K,
                              int                N)
{
    // One block per sequence, one thread per flip position j = 0 .. N-1.
    int seq_idx = blockIdx.x;
    int j       = threadIdx.x;

    if (j >= N) return;

    const signed char* row = seqs + (long long)seq_idx * N;
    const int* C = autocorrs + (long long)seq_idx * (N - 1);  // C[0]=C_1 .. C[N-2]=C_{N-1}
    int xj = row[j];

    long long energy = 0;
    for (int k = 1; k < N; ++k) {
        int c_k = C[k - 1];  // current autocorrelation at lag k

        // delta_k = -2 * x[j] * ( [j-k>=0]*x[j-k] + [j+k<N]*x[j+k] )
        int contrib = 0;
        if (j - k >= 0) contrib += row[j - k];
        if (j + k < N)  contrib += row[j + k];
        int delta_k = -2 * xj * contrib;

        long long new_ck = (long long)(c_k + delta_k);
        energy += new_ck * new_ck;
    }

    // Store E_new(j).  Layout: (K, N).
    delta_energies[(long long)seq_idx * N + j] = energy;
}
"""

_delta_energy_kernel = None


def _get_delta_energy_kernel():
    global _delta_energy_kernel
    if _delta_energy_kernel is None:
        _delta_energy_kernel = cp.RawKernel(_LABS_DELTA_ENERGY_KERNEL_CODE,
                                            "labs_delta_energy_kernel")
    return _delta_energy_kernel


# ---------------------------------------------------------------------------
# Kernel 3 — update autocorrelations in-place after selecting best move
# ---------------------------------------------------------------------------

_LABS_UPDATE_AUTOCORR_KERNEL_CODE = r"""
extern "C" __global__
void labs_update_autocorr_kernel(const signed char* seqs,
                                 int*               autocorrs,
                                 const int*         move_pos,
                                 int                K,
                                 int                N)
{
    // One block per sequence, one thread per lag (1 .. N-1).
    // MUST be called BEFORE flipping the bit in seqs.
    int seq_idx = blockIdx.x;
    int lag     = threadIdx.x + 1;

    if (lag >= N) return;

    int j = move_pos[seq_idx];
    const signed char* row = seqs + (long long)seq_idx * N;
    int xj = row[j];

    int contrib = 0;
    if (j - lag >= 0) contrib += row[j - lag];
    if (j + lag < N)  contrib += row[j + lag];
    int delta_k = -2 * xj * contrib;

    autocorrs[(long long)seq_idx * (N - 1) + (lag - 1)] += delta_k;
}
"""

_update_autocorr_kernel = None


def _get_update_autocorr_kernel():
    global _update_autocorr_kernel
    if _update_autocorr_kernel is None:
        _update_autocorr_kernel = cp.RawKernel(_LABS_UPDATE_AUTOCORR_KERNEL_CODE,
                                               "labs_update_autocorr_kernel")
    return _update_autocorr_kernel


# ---------------------------------------------------------------------------
# Naive (original) GPU MTS — kept as fallback for parity testing
# ---------------------------------------------------------------------------

def _run_mts_gpu_naive(
    seeds_pm1: np.ndarray,
    *,
    budget_s: float | None,
    max_iters: int | None,
    tabu_params: dict,
    rng_seed: int,
) -> tuple[np.ndarray, float, dict]:
    """Original GPU MTS using full neighbor matrix + batch energy eval."""

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


# ---------------------------------------------------------------------------
# Delta-energy GPU MTS (new default)
# ---------------------------------------------------------------------------

def run_mts_gpu(
    seeds_pm1: np.ndarray,
    *,
    budget_s: float | None,
    max_iters: int | None,
    tabu_params: dict,
    rng_seed: int,
) -> tuple[np.ndarray, float, dict]:
    """GPU MTS with incremental delta-energy CUDA kernels.

    Instead of building a K×N neighbor matrix and evaluating all sequences
    from scratch, this uses three custom CUDA kernels:
      1. autocorr_kernel    — compute initial C_k for all K sequences
      2. delta_energy_kernel — compute E_new(j) for all K×N flip positions
      3. update_autocorr_kernel — incrementally update C_k after each move

    This reduces per-iteration cost from O(K×N³) to O(K×N²).
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

    # --- Step 1: initial autocorrelations via kernel ---
    autocorrs = cp.zeros((K, N - 1), dtype=cp.int32)
    block_ac = _next_power_of_2(N - 1)
    _get_autocorr_kernel()(
        (K,), (block_ac,),
        (pop, autocorrs, np.int32(K), np.int32(N)),
    )

    # Initial energies = sum of C_k^2
    E_cur = cp.sum(autocorrs.astype(cp.int64) ** 2, axis=1)

    best_per = pop.copy()
    E_best_per = E_cur.copy()

    best_idx = int(cp.asnumpy(cp.argmin(E_best_per)))
    best_global_seq = cp.asnumpy(best_per[best_idx]).astype(np.int8, copy=False)
    best_global_E = float(cp.asnumpy(E_best_per[best_idx]))

    trace_best = [best_global_E]

    # Pre-allocate delta energies buffer
    delta_energies = cp.empty((K, N), dtype=cp.int64)
    block_de = _next_power_of_2(N)

    big = cp.int64(cp.iinfo(cp.int64).max)
    ar = cp.arange(K)

    it = 0
    while True:
        it += 1
        if max_iters is not None and it > int(max_iters):
            break
        if budget_s is not None and (time.perf_counter() - t0) >= float(budget_s):
            break

        # --- Step 2: delta-energy kernel → E_neigh (K, N) ---
        _get_delta_energy_kernel()(
            (K,), (block_de,),
            (pop, autocorrs, delta_energies, np.int32(K), np.int32(N)),
        )
        E_neigh = delta_energies

        # --- Step 3: tabu masking + argmin (unchanged logic) ---
        if tenure > 0:
            blocked = tabu > 0
            if aspiration:
                aspir = E_neigh < E_best_per[:, None]
                blocked = blocked & (~aspir)
            E_masked = E_neigh.copy()
            E_masked[blocked] = big
        else:
            E_masked = E_neigh

        move_pos = cp.argmin(E_masked, axis=1).astype(cp.int32)
        move_E = E_neigh[ar, move_pos]

        if tenure > 0:
            tabu = cp.maximum(tabu - 1, 0)
            tabu[ar, move_pos] = tenure

        # --- Step 4: update autocorrelations BEFORE flipping bits ---
        _get_update_autocorr_kernel()(
            (K,), (block_ac,),
            (pop, autocorrs, move_pos, np.int32(K), np.int32(N)),
        )

        # --- Step 5: flip the selected bits ---
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
