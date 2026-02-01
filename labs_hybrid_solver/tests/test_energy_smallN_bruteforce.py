import itertools

import numpy as np

from labs_hybrid.labs_energy_cpu import labs_energy, labs_energy_batch


def labs_energy_ref(seq_pm1: np.ndarray) -> int:
    """Independent reference implementation (slow but simple)."""

    s = seq_pm1.astype(int)
    N = len(s)
    E = 0
    for k in range(1, N):
        Ck = sum(int(s[i] * s[i + k]) for i in range(N - k))
        E += Ck * Ck
    return int(E)


def test_energy_matches_reference_bruteforce_tinyN():
    for N in range(2, 11):
        for bits in itertools.product([-1, 1], repeat=N):
            s = np.array(bits, dtype=np.int8)
            assert labs_energy(s) == labs_energy_ref(s)


def test_batch_matches_single_small():
    rng = np.random.default_rng(0)
    N = 10
    K = 32
    seeds = rng.integers(0, 2, size=(K, N), dtype=np.int8) * 2 - 1
    E1 = np.array([labs_energy(seeds[i]) for i in range(K)], dtype=np.int64)
    E2 = labs_energy_batch(seeds)
    assert np.all(E1 == E2)


def test_energy_symmetry_under_global_flip():
    rng = np.random.default_rng(1)
    for N in [5, 8, 10]:
        s = rng.integers(0, 2, size=N, dtype=np.int8) * 2 - 1
        assert labs_energy(s) == labs_energy((-s).astype(np.int8))
