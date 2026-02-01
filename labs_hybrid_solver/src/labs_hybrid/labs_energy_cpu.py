from __future__ import annotations

import numpy as np


def aperiodic_autocorrelations_pm1(seq_pm1: np.ndarray) -> np.ndarray:
    """Return aperiodic autocorrelations C_k for k=1..N-1.

    seq_pm1: (N,) int8 in {-1,+1}
    """

    s = seq_pm1.astype(np.int16, copy=False)
    N = s.shape[0]
    C = np.empty(N - 1, dtype=np.int32)
    for k in range(1, N):
        C[k - 1] = int(np.sum(s[: N - k] * s[k:]))
    return C


def labs_energy(seq_pm1: np.ndarray) -> int:
    """LABS energy E = sum_{k=1..N-1} C_k^2 (integer)."""

    C = aperiodic_autocorrelations_pm1(seq_pm1)
    return int(np.sum(C.astype(np.int64) ** 2))


def labs_merit_factor(seq_pm1: np.ndarray) -> float:
    """Merit factor F = N^2 / (2E). Higher is better."""

    E = labs_energy(seq_pm1)
    N = int(seq_pm1.shape[0])
    if E == 0:
        return float("inf")
    return (N * N) / (2.0 * float(E))


def labs_energy_batch(seqs_pm1: np.ndarray) -> np.ndarray:
    """Batch energy on CPU.

    seqs_pm1: (K,N) int8
    returns: (K,) int64
    """

    x = seqs_pm1.astype(np.int16, copy=False)
    K, N = x.shape
    E = np.zeros(K, dtype=np.int64)
    for k in range(1, N):
        Ck = np.sum(x[:, : N - k] * x[:, k:], axis=1, dtype=np.int32)
        E += Ck.astype(np.int64) ** 2
    return E
