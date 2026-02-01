from __future__ import annotations

from typing import Any, Callable

import numpy as np


def select_topK_by_energy(
    seeds_pm1: Any,
    K_select: int,
    energy_fn: Callable[[Any], Any],
    xp=np,
) -> tuple[Any, dict]:
    """Compute energies batched and return the best K_select seeds.

    Important: this selection step extracts the left tail of the seeder's sampled distribution.
    """

    if K_select <= 0:
        raise ValueError("K_select must be > 0")

    energies = energy_fn(seeds_pm1)
    idx = xp.argsort(energies)[:K_select]
    selected = seeds_pm1[idx]

    info = {
        "K_in": int(seeds_pm1.shape[0]),
        "K_select": int(K_select),
        "energy_min": float(xp.min(energies)),
        "energy_median": float(xp.median(energies)),
        "energy_mean": float(xp.mean(energies)),
    }
    return selected, info


def mean_pairwise_hamming_topK(seqs_pm1: np.ndarray) -> float:
    """Mean pairwise Hamming distance for a small K on CPU.

    For ±1 representation, Hamming(s,t) = (N - dot(s,t))/2.
    """

    K, N = seqs_pm1.shape
    if K < 2:
        return 0.0
    x = seqs_pm1.astype(np.int16)
    dots = x @ x.T
    dists = (N - dots) / 2
    triu = np.triu_indices(K, k=1)
    return float(np.mean(dists[triu]))
