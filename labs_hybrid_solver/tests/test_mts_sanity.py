import numpy as np

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
