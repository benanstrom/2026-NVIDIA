from __future__ import annotations

from typing import Any

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None  # type: ignore


def labs_energy_batch_gpu(seqs_pm1: Any) -> Any:
    """Batched LABS energy on GPU (CuPy).

    seqs_pm1: (K,N) int8 CuPy array
    returns: (K,) int64 CuPy array
    """

    if cp is None:
        raise RuntimeError("CuPy not available")

    x = seqs_pm1.astype(cp.int16, copy=False)
    K, N = x.shape
    E = cp.zeros(K, dtype=cp.int64)
    for k in range(1, N):
        Ck = cp.sum(x[:, : N - k] * x[:, k:], axis=1, dtype=cp.int32)
        E += Ck.astype(cp.int64) ** 2
    return E


def labs_energy_gpu_single(seq_pm1: np.ndarray) -> int:
    if cp is None:
        raise RuntimeError("CuPy not available")
    x = cp.asarray(seq_pm1, dtype=cp.int8)
    E = labs_energy_batch_gpu(x[None, :])
    return int(cp.asnumpy(E)[0])
