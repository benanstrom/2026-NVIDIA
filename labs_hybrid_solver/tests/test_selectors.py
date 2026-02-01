import numpy as np
import pytest

from labs_hybrid.labs_energy_cpu import labs_energy_batch
from labs_hybrid.selectors import mean_pairwise_hamming_topK, select_topK_by_energy


def test_postselection_extracts_low_energy_tail():
    """Post-selection must return the K_select lowest-energy seeds."""
    rng = np.random.default_rng(42)
    N = 16
    K_in = 50
    K_select = 10
    seeds = rng.integers(0, 2, size=(K_in, N), dtype=np.int8) * 2 - 1

    energies_all = labs_energy_batch(seeds)
    selected, info = select_topK_by_energy(seeds, K_select, labs_energy_batch, xp=np)

    assert selected.shape == (K_select, N)

    energies_sel = labs_energy_batch(selected)

    # Selected seeds must be the actual lowest-energy subset.
    expected_idx = np.argsort(energies_all)[:K_select]
    expected_energies = np.sort(energies_all[expected_idx])
    np.testing.assert_array_equal(np.sort(energies_sel), expected_energies)

    # Info dict must have expected keys.
    assert info["K_in"] == K_in
    assert info["K_select"] == K_select
    assert "energy_min" in info
    assert "energy_median" in info
    assert "energy_mean" in info


def test_postselection_improves_population_quality():
    """Mean energy after post-selection must be <= mean energy before."""
    rng = np.random.default_rng(7)
    N = 12
    K_in = 100
    K_select = 20
    seeds = rng.integers(0, 2, size=(K_in, N), dtype=np.int8) * 2 - 1

    selected, _ = select_topK_by_energy(seeds, K_select, labs_energy_batch, xp=np)

    mean_before = float(labs_energy_batch(seeds).mean())
    mean_after = float(labs_energy_batch(selected).mean())
    assert mean_after <= mean_before


def test_postselection_preserves_sequences():
    """Selected sequences must be exact copies from the input."""
    rng = np.random.default_rng(99)
    N = 10
    K_in = 30
    K_select = 5
    seeds = rng.integers(0, 2, size=(K_in, N), dtype=np.int8) * 2 - 1

    selected, _ = select_topK_by_energy(seeds, K_select, labs_energy_batch, xp=np)

    for i in range(K_select):
        matches = np.all(seeds == selected[i], axis=1)
        assert np.any(matches), f"Selected row {i} not found in original seeds"


def test_postselection_k_select_equals_k_in():
    """When K_select == K_in, all seeds should be returned."""
    rng = np.random.default_rng(0)
    N = 8
    K = 10
    seeds = rng.integers(0, 2, size=(K, N), dtype=np.int8) * 2 - 1

    selected, info = select_topK_by_energy(seeds, K, labs_energy_batch, xp=np)
    assert selected.shape == (K, N)
    assert info["K_in"] == K
    assert info["K_select"] == K


def test_postselection_rejects_invalid_k():
    """K_select <= 0 must raise ValueError."""
    seeds = np.ones((5, 8), dtype=np.int8)
    with pytest.raises(ValueError):
        select_topK_by_energy(seeds, 0, labs_energy_batch, xp=np)


def test_mean_pairwise_hamming_identical():
    """Identical sequences -> hamming distance = 0."""
    identical = np.ones((5, 10), dtype=np.int8)
    assert mean_pairwise_hamming_topK(identical) == 0.0


def test_mean_pairwise_hamming_single():
    """Single sequence -> 0 by convention."""
    single = np.ones((1, 10), dtype=np.int8)
    assert mean_pairwise_hamming_topK(single) == 0.0


def test_mean_pairwise_hamming_opposite():
    """Opposite sequences -> hamming distance = N."""
    opp = np.array([[1] * 10, [-1] * 10], dtype=np.int8)
    assert mean_pairwise_hamming_topK(opp) == 10.0
