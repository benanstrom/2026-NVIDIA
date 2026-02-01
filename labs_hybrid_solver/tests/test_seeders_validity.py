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

    if info1.get("cudaq_available"):
        assert info1.get("cudaq_seeded", False), (
            f"Seeder {name}: CUDA-Q is available but set_random_seed() failed — "
            "quantum sampler is non-deterministic"
        )

    assert np.array_equal(seeds1, seeds2)


# ---------------------------------------------------------------------------
# CUDA-Q seeding: verify different seeds → different output
# ---------------------------------------------------------------------------

_CUDAQ_SEEDERS = ["qaoa", "dcqo", "dcqo_plus"]
_CUDAQ_PARAMS = {
    "qaoa": {"p": 1},
    "dcqo": {"steps": 6},
    "dcqo_plus": {
        "steps": 6,
        "trotter_config": {"time_grid": "midpoint", "ordering": "forward_reverse", "K_perm": 1},
    },
}


@pytest.mark.parametrize("name", _CUDAQ_SEEDERS)
def test_cudaq_seeder_different_seeds_differ(name: str):
    """When CUDA-Q is available, different rng_seeds must produce different outputs.

    This confirms cudaq.set_random_seed() is actually respected and
    not silently ignored (which would make every call return the same result).
    """
    seeder = get_seeder(name)
    N = 10
    shots = 32
    K_out = 16
    params = _CUDAQ_PARAMS[name]

    seeds_a, info_a = seeder(N, shots, K_out, backend="cpu", rng_seed=0, params=params)
    seeds_b, info_b = seeder(N, shots, K_out, backend="cpu", rng_seed=999, params=params)

    if not info_a.get("cudaq_available"):
        pytest.skip("CUDA-Q not available; classical fallbacks use numpy RNG (already tested)")

    assert not np.array_equal(seeds_a, seeds_b), (
        f"Seeder {name}: identical output for rng_seed=0 and rng_seed=999 — "
        "cudaq.set_random_seed() may not be respected"
    )
