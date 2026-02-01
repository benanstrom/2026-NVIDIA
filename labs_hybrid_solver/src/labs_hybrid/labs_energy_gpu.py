from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None  # type: ignore


# ---------------------------------------------------------------------------
# CuPy RawKernel — single-launch LABS energy
# ---------------------------------------------------------------------------

_LABS_ENERGY_KERNEL_CODE = r"""
extern "C" __global__
void labs_energy_kernel(const signed char* seqs,
                        long long*         energies,
                        int                K,
                        int                N)
{
    // One block per sequence, one thread per lag (1 .. blockDim.x).
    int seq_idx = blockIdx.x;
    int lag     = threadIdx.x + 1;          // k = 1 .. N-1

    // --- compute C_k for this thread's lag ---
    long long c_k_sq = 0;
    if (lag < N) {
        int c_k = 0;
        const signed char* row = seqs + (long long)seq_idx * N;
        for (int i = 0; i < N - lag; ++i) {
            c_k += row[i] * row[i + lag];
        }
        c_k_sq = (long long)c_k * c_k;
    }

    // --- shared-memory tree reduction ---
    extern __shared__ long long sdata[];
    sdata[threadIdx.x] = c_k_sq;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        energies[seq_idx] = sdata[0];
    }
}
"""

_raw_kernel = None


def _get_raw_kernel():
    """Lazily compile the RawKernel (once per process)."""
    global _raw_kernel
    if _raw_kernel is None:
        _raw_kernel = cp.RawKernel(_LABS_ENERGY_KERNEL_CODE,
                                   "labs_energy_kernel")
    return _raw_kernel


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def labs_energy_batch_gpu(seqs_pm1: Any) -> Any:
    """Batched LABS energy on GPU (CuPy).

    seqs_pm1: (K,N) int8 CuPy array
    returns: (K,) int64 CuPy array
    """

    if cp is None:
        raise RuntimeError("CuPy not available")

    K, N = seqs_pm1.shape

    # Edge case: N <= 1 means no lags, energy is 0
    if N <= 1:
        return cp.zeros(K, dtype=cp.int64)

    seqs = cp.ascontiguousarray(seqs_pm1, dtype=cp.int8)
    energies = cp.zeros(K, dtype=cp.int64)

    kernel = _get_raw_kernel()
    block_size = _next_power_of_2(N - 1)     # threads >= N-1, power of 2
    shared_bytes = block_size * 8             # sizeof(long long)

    kernel((K,),                              # grid: one block per sequence
           (block_size,),                     # block: one thread per lag
           (seqs, energies, K, N),
           shared_mem=shared_bytes)

    return energies


def labs_energy_gpu_single(seq_pm1: np.ndarray) -> int:
    if cp is None:
        raise RuntimeError("CuPy not available")
    x = cp.asarray(seq_pm1, dtype=cp.int8)
    E = labs_energy_batch_gpu(x[None, :])
    return int(cp.asnumpy(E)[0])


# ---------------------------------------------------------------------------
# Fallback: original Python-loop implementation (kept for parity testing)
# ---------------------------------------------------------------------------

def _labs_energy_batch_gpu_fallback(seqs_pm1: Any) -> Any:
    """Original loop-based GPU energy (for testing parity with raw kernel)."""

    if cp is None:
        raise RuntimeError("CuPy not available")

    x = seqs_pm1.astype(cp.int16, copy=False)
    K, N = x.shape
    E = cp.zeros(K, dtype=cp.int64)
    for k in range(1, N):
        Ck = cp.sum(x[:, : N - k] * x[:, k:], axis=1, dtype=cp.int32)
        E += Ck.astype(cp.int64) ** 2
    return E
