import numpy as np
import pytest

from labs_hybrid.labs_energy_cpu import labs_energy_batch

try:
    import cupy as cp
except Exception:
    cp = None

from labs_hybrid.labs_energy_gpu import labs_energy_batch_gpu


def has_cuda() -> bool:
    if cp is None:
        return False
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:
        return False


@pytest.mark.skipif(not has_cuda(), reason="CUDA/CuPy not available")
def test_energy_batch_cpu_gpu_parity():
    rng = np.random.default_rng(0)
    K, N = 64, 16
    seeds = rng.integers(0, 2, size=(K, N), dtype=np.int8) * 2 - 1

    E_cpu = labs_energy_batch(seeds)
    E_gpu = cp.asnumpy(labs_energy_batch_gpu(cp.asarray(seeds, dtype=cp.int8)))

    assert np.array_equal(E_cpu, E_gpu)
