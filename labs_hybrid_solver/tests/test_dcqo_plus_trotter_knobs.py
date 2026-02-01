import numpy as np

from labs_hybrid.seeders.dcqo_plus_seeder import generate_seeds


def test_baseline_knobs_reproduce_baseline_behavior_metadata():
    N = 12
    params = {
        "steps": 6,
        "max_lag": 4,
        "trotter_config": {"time_grid": "right_endpoint", "ordering": "fixed", "K_perm": 1},
    }
    _, info = generate_seeds(N, 8, 8, backend="cpu", rng_seed=0, params=params)
    meta = info["trotter_meta"]
    assert meta["time_grid"] == "right_endpoint"
    assert meta["ordering"] == "fixed"
    assert meta["K_perm"] == 1
    assert np.allclose(meta["t_points"], [(j + 1) / 6.0 for j in range(6)])


def test_midpoint_changes_time_grid_only():
    N = 12
    base = {
        "steps": 6,
        "max_lag": 4,
        "trotter_config": {"time_grid": "right_endpoint", "ordering": "fixed", "K_perm": 1},
    }
    mid = {
        "steps": 6,
        "max_lag": 4,
        "trotter_config": {"time_grid": "midpoint", "ordering": "fixed", "K_perm": 1},
    }
    _, info_base = generate_seeds(N, 8, 8, backend="cpu", rng_seed=0, params=base)
    _, info_mid = generate_seeds(N, 8, 8, backend="cpu", rng_seed=0, params=mid)
    m0 = info_base["trotter_meta"]
    m1 = info_mid["trotter_meta"]

    assert m0["pair_count"] == m1["pair_count"]
    assert m0["depth_proxy"] == m1["depth_proxy"]
    assert m0["order_sigs"] == m1["order_sigs"]
    assert np.allclose(m1["t_points"], [(j + 0.5) / 6.0 for j in range(6)])


def test_forward_reverse_alternates_ordering_deterministically():
    N = 12
    params = {
        "steps": 6,
        "max_lag": 4,
        "trotter_config": {"time_grid": "right_endpoint", "ordering": "forward_reverse", "K_perm": 1},
    }
    _, info = generate_seeds(N, 8, 8, backend="cpu", rng_seed=0, params=params)
    sigs = info["trotter_meta"]["order_sigs"]
    assert len(sigs) == 6
    assert sigs[0] != sigs[1]
    assert sigs[0] == sigs[2]
    assert sigs[1] == sigs[3]


def test_k_permutation_changes_ordering_without_changing_term_set_or_depth():
    N = 12
    params = {
        "steps": 6,
        "max_lag": 4,
        "trotter_config": {"time_grid": "right_endpoint", "ordering": "k_permutation", "K_perm": 3},
    }
    _, info = generate_seeds(N, 8, 8, backend="cpu", rng_seed=0, params=params)
    meta = info["trotter_meta"]
    sigs = meta["order_sigs"]
    assert len(set(sigs)) > 1
    assert meta["depth_proxy"] == 6 * (N + meta["pair_count"])

    # When pair_count is small, we include full orders for verification.
    if "orders_full" in meta:
        base_set = set(meta["orders_full"][0])
        for o in meta["orders_full"]:
            assert set(o) == base_set
