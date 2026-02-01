"""Kernel-level unit tests for the three delta-energy CUDA kernels."""

import numpy as np
import pytest

from labs_hybrid.labs_energy_cpu import aperiodic_autocorrelations_pm1, labs_energy
from labs_hybrid.labs_energy_gpu import labs_energy_batch_gpu

cp = pytest.importorskip("cupy")

from labs_hybrid.mts.mts_gpu import (
    _get_autocorr_kernel,
    _get_delta_energy_kernel,
    _get_update_autocorr_kernel,
    _next_power_of_2,
)


PARAMS = [(1, 4), (64, 16), (32, 40), (100, 30)]


@pytest.mark.parametrize("K,N", PARAMS)
def test_autocorr_kernel_matches_cpu(K, N):
    """autocorr_kernel output must match aperiodic_autocorrelations_pm1 per row."""
    rng = np.random.default_rng(42)
    seeds = (rng.integers(0, 2, size=(K, N)) * 2 - 1).astype(np.int8)

    pop = cp.asarray(seeds, dtype=cp.int8)
    autocorrs = cp.zeros((K, N - 1), dtype=cp.int32)
    block = _next_power_of_2(N - 1)
    _get_autocorr_kernel()(
        (K,), (block,),
        (pop, autocorrs, np.int32(K), np.int32(N)),
    )
    gpu_autocorrs = cp.asnumpy(autocorrs)

    for i in range(K):
        cpu_C = aperiodic_autocorrelations_pm1(seeds[i])
        np.testing.assert_array_equal(
            gpu_autocorrs[i], cpu_C,
            err_msg=f"Autocorrelation mismatch at row {i}",
        )


@pytest.mark.parametrize("K,N", PARAMS)
def test_delta_energy_matches_naive_full_eval(K, N):
    """delta_energy_kernel must produce the same E_new(j) as full neighbor eval."""
    rng = np.random.default_rng(99)
    seeds = (rng.integers(0, 2, size=(K, N)) * 2 - 1).astype(np.int8)

    pop = cp.asarray(seeds, dtype=cp.int8)

    # Compute autocorrelations on GPU
    autocorrs = cp.zeros((K, N - 1), dtype=cp.int32)
    block_ac = _next_power_of_2(N - 1)
    _get_autocorr_kernel()(
        (K,), (block_ac,),
        (pop, autocorrs, np.int32(K), np.int32(N)),
    )

    # Compute delta energies on GPU
    delta_energies = cp.empty((K, N), dtype=cp.int64)
    block_de = _next_power_of_2(N)
    _get_delta_energy_kernel()(
        (K,), (block_de,),
        (pop, autocorrs, delta_energies, np.int32(K), np.int32(N)),
    )
    gpu_E = cp.asnumpy(delta_energies)

    # Naive: build full neighbor matrix and evaluate
    neigh = cp.repeat(pop, N, axis=0)
    rows = cp.arange(K * N)
    cols = cp.tile(cp.arange(N), K)
    neigh[rows, cols] *= -1
    naive_E = cp.asnumpy(labs_energy_batch_gpu(neigh).reshape(K, N))

    np.testing.assert_array_equal(
        gpu_E, naive_E,
        err_msg="Delta-energy kernel does not match naive full evaluation",
    )


@pytest.mark.parametrize("K,N", PARAMS)
def test_update_autocorr_is_consistent(K, N):
    """After update_autocorr + flip, recomputing from scratch must give same C_k."""
    rng = np.random.default_rng(7)
    seeds = (rng.integers(0, 2, size=(K, N)) * 2 - 1).astype(np.int8)

    pop = cp.asarray(seeds, dtype=cp.int8)

    # Initial autocorrelations
    autocorrs = cp.zeros((K, N - 1), dtype=cp.int32)
    block_ac = _next_power_of_2(N - 1)
    _get_autocorr_kernel()(
        (K,), (block_ac,),
        (pop, autocorrs, np.int32(K), np.int32(N)),
    )

    # Pick random move positions
    move_pos = cp.array(rng.integers(0, N, size=K), dtype=cp.int32)

    # Update autocorrelations incrementally (BEFORE flip)
    _get_update_autocorr_kernel()(
        (K,), (block_ac,),
        (pop, autocorrs, move_pos, np.int32(K), np.int32(N)),
    )

    # Flip the bits
    ar = cp.arange(K)
    pop[ar, move_pos] *= -1

    # Recompute from scratch on the flipped population
    autocorrs_fresh = cp.zeros((K, N - 1), dtype=cp.int32)
    _get_autocorr_kernel()(
        (K,), (block_ac,),
        (pop, autocorrs_fresh, np.int32(K), np.int32(N)),
    )

    np.testing.assert_array_equal(
        cp.asnumpy(autocorrs),
        cp.asnumpy(autocorrs_fresh),
        err_msg="Incremental autocorr update does not match from-scratch recompute",
    )
