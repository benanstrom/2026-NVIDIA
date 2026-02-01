import numpy as np
import pytest

from labs_hybrid.labs_energy_cpu import labs_energy_batch
from labs_hybrid.mts.mts_cpu import run_mts_cpu


def test_mts_improves_or_matches_best_seed_cpu():
    rng = np.random.default_rng(0)
    N = 16
    K = 12
    seeds = rng.integers(0, 2, size=(K, N), dtype=np.int8) * 2 - 1
    best_seed = int(labs_energy_batch(seeds).min())

    best_seq, best_E, logs = run_mts_cpu(
        seeds,
        budget_s=None,
        max_iters=50,
        tabu_params={"tenure": 5, "aspiration": True},
        rng_seed=0,
    )

    assert int(best_E) <= best_seed
    assert best_seq.shape == (N,)
    assert set(np.unique(best_seq)).issubset({-1, 1})
    assert len(logs.get("best_trace", [])) >= 1


# ---------------------------------------------------------------------------
# GPU delta-energy tests (require CuPy + GPU)
# ---------------------------------------------------------------------------

cp = None
try:
    import cupy as _cp
    cp = _cp
except Exception:
    pass

gpu_available = cp is not None


@pytest.mark.skipif(not gpu_available, reason="CuPy / GPU not available")
def test_mts_gpu_delta_improves_or_matches_best_seed():
    """Delta-energy GPU MTS must improve (or match) the best seed energy."""
    from labs_hybrid.mts.mts_gpu import run_mts_gpu

    rng = np.random.default_rng(0)
    N = 16
    K = 12
    seeds = rng.integers(0, 2, size=(K, N), dtype=np.int8) * 2 - 1
    best_seed = int(labs_energy_batch(seeds).min())

    best_seq, best_E, logs = run_mts_gpu(
        seeds,
        budget_s=None,
        max_iters=50,
        tabu_params={"tenure": 5, "aspiration": True},
        rng_seed=0,
    )

    assert int(best_E) <= best_seed
    assert best_seq.shape == (N,)
    assert set(np.unique(best_seq)).issubset({-1, 1})
    assert len(logs.get("best_trace", [])) >= 1


@pytest.mark.skipif(not gpu_available, reason="CuPy / GPU not available")
def test_mts_gpu_delta_vs_naive_parity():
    """Delta-energy and naive GPU MTS must produce the same best energy on identical inputs."""
    from labs_hybrid.mts.mts_gpu import run_mts_gpu, _run_mts_gpu_naive

    rng = np.random.default_rng(123)
    N = 16
    K = 8
    seeds = rng.integers(0, 2, size=(K, N), dtype=np.int8) * 2 - 1

    params = {"tenure": 5, "aspiration": True, "trace_stride": 1}

    _, E_delta, logs_delta = run_mts_gpu(
        seeds.copy(),
        budget_s=None,
        max_iters=30,
        tabu_params=params,
        rng_seed=0,
    )
    _, E_naive, logs_naive = _run_mts_gpu_naive(
        seeds.copy(),
        budget_s=None,
        max_iters=30,
        tabu_params=params,
        rng_seed=0,
    )

    assert int(E_delta) == int(E_naive), (
        f"Delta energy {E_delta} != naive energy {E_naive}"
    )
