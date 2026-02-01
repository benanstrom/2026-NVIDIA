import numpy as np
import pytest

from labs_hybrid.seeders import get_seeder, list_seeders


@pytest.mark.parametrize("name", list_seeders())
def test_seeder_outputs_valid_pm1(name: str):
    seeder = get_seeder(name)
    N = 12
    shots = 32
    K_out = 32
    params = {}
    if name == "qaoa":
        params = {"p": 1}
    if name == "dcqo":
        params = {"steps": 8}
    if name == "dcqo_plus":
        params = {
            "steps": 8,
            "trotter_config": {"time_grid": "right_endpoint", "ordering": "fixed", "K_perm": 1},
        }
    if name == "pce":
        params = {"layers": 1}

    seeds, info = seeder(N, shots, K_out, backend="cpu", rng_seed=123, params=params)

    assert isinstance(seeds, np.ndarray)
    assert seeds.shape == (K_out, N)
    assert seeds.dtype == np.int8
    assert set(np.unique(seeds)).issubset({-1, 1})
    assert isinstance(info, dict)


@pytest.mark.parametrize("name", list_seeders())
def test_seeder_determinism_when_seeded(name: str):
    seeder = get_seeder(name)
    N = 10
    shots = 16
    K_out = 16
    params = {}
    if name == "qaoa":
        params = {"p": 1}
    if name == "dcqo":
        params = {"steps": 6}
    if name == "dcqo_plus":
        params = {
            "steps": 6,
            "trotter_config": {"time_grid": "midpoint", "ordering": "forward_reverse", "K_perm": 1},
        }
    if name == "pce":
        params = {"layers": 1}

    seeds1, info1 = seeder(N, shots, K_out, backend="cpu", rng_seed=42, params=params)
    seeds2, info2 = seeder(N, shots, K_out, backend="cpu", rng_seed=42, params=params)

    if info1.get("cudaq_available") and not info1.get("cudaq_seeded", False):
        pytest.skip("CUDA-Q sampler not forced deterministic; skipping determinism assertion")

    assert np.array_equal(seeds1, seeds2)
