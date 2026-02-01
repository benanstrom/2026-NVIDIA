from __future__ import annotations

import numpy as np


def get_np_rng(seed: int) -> np.random.Generator:
    """Create a numpy RNG used as the determinism source of truth.

    Even when generating GPU arrays, we sample on CPU first (then transfer).
    This avoids device-dependent RNG differences.
    """

    return np.random.default_rng(int(seed))


def sample_pm1(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Sample ±1 int8 array."""

    bits = rng.integers(0, 2, size=shape, dtype=np.int8)
    return (bits * 2 - 1).astype(np.int8)


def bits01_to_pm1(bits01: np.ndarray) -> np.ndarray:
    bits01 = bits01.astype(np.int8, copy=False)
    return (bits01 * 2 - 1).astype(np.int8, copy=False)


def pm1_to_bits01(pm1: np.ndarray) -> np.ndarray:
    pm1 = pm1.astype(np.int8, copy=False)
    return ((pm1 + 1) // 2).astype(np.int8, copy=False)
