import numpy as np
import pytest

from labs_hybrid.labs_energy_cpu import labs_energy_batch

try:
    import cupy as cp
except Exception:
    cp = None

from labs_hybrid.labs_energy_gpu import (
    labs_energy_batch_gpu,
    _labs_energy_batch_gpu_fallback,
)


def has_cuda() -> bool:
    if cp is None:
        return False
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


# -------------------------------------------------------------------
# Original test (kept as-is)
# -------------------------------------------------------------------


@pytest.mark.skipif(not has_cuda(), reason="CUDA/CuPy not available")
def test_energy_batch_cpu_gpu_parity():
    rng = np.random.default_rng(0)
    K, N = 64, 16
    seeds = rng.integers(0, 2, size=(K, N), dtype=np.int8) * 2 - 1

    E_cpu = labs_energy_batch(seeds)
    E_gpu = cp.asnumpy(labs_energy_batch_gpu(cp.asarray(seeds, dtype=cp.int8)))

    assert np.array_equal(E_cpu, E_gpu)


# -------------------------------------------------------------------
# Parameterized CPU ↔ GPU parity across varied (K, N) sizes
# -------------------------------------------------------------------


@pytest.mark.skipif(not has_cuda(), reason="CUDA/CuPy not available")
@pytest.mark.parametrize("K,N", [(1, 4), (64, 16), (32, 40), (100, 30)])
def test_cpu_gpu_parity_parameterized(K: int, N: int):
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 2, size=(K, N), dtype=np.int8) * 2 - 1

    E_cpu = labs_energy_batch(seeds)
    E_gpu = cp.asnumpy(labs_energy_batch_gpu(cp.asarray(seeds, dtype=cp.int8)))

    assert np.array_equal(E_cpu, E_gpu), (
        f"CPU/GPU mismatch for K={K}, N={N}"
    )


# -------------------------------------------------------------------
# Raw kernel ↔ fallback loop parity
# -------------------------------------------------------------------


@pytest.mark.skipif(not has_cuda(), reason="CUDA/CuPy not available")
@pytest.mark.parametrize("K,N", [(1, 4), (64, 16), (32, 40), (100, 30)])
def test_raw_kernel_matches_fallback(K: int, N: int):
    rng = np.random.default_rng(99)
    seeds = rng.integers(0, 2, size=(K, N), dtype=np.int8) * 2 - 1
    seqs_gpu = cp.asarray(seeds, dtype=cp.int8)

    E_raw = cp.asnumpy(labs_energy_batch_gpu(seqs_gpu))
    E_loop = cp.asnumpy(_labs_energy_batch_gpu_fallback(seqs_gpu))

    assert np.array_equal(E_raw, E_loop), (
        f"Raw kernel vs fallback mismatch for K={K}, N={N}"
    )
