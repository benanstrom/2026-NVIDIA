# TEST_SUITE.md — NVIDIA LABS Hybrid Solver (iQuHACK 2026)

This test suite is designed to establish **Gate 0 (CPU-only correctness)** and to support **Gate 2 (CPU/GPU parity)**.

## How to run

```bash
cd labs_hybrid_solver
pytest
```

## What each test proves

### `test_energy_smallN_bruteforce.py`
- Uses an independent reference implementation of the LABS energy to validate correctness for tiny N (≤10).
- Validates batch energy matches per-sequence energy.
- Validates invariance under global sign flip.

### `test_energy_cpu_gpu_parity.py`
- Compares batched LABS energy computed on CPU (NumPy) vs GPU (CuPy).
- Skips automatically if CUDA/CuPy are unavailable.
- **Parity philosophy:** CPU is the oracle; GPU must match exactly (energy is integer).

### `test_seeders_validity.py`
- Validates each seeder returns:
  - shape `(K_out, N)`
  - dtype `int8`
  - values in `{−1, +1}`
- Determinism: if CUDA-Q is not available, or if CUDA-Q is explicitly seeded (`cudaq_seeded=True`), outputs must repeat bit-for-bit for identical inputs.

### `test_mts_sanity.py`
- Ensures the fixed CPU MTS returns a solution with final energy ≤ best seed energy under deterministic settings.

### `test_dcqo_plus_trotter_knobs.py`
Validates the **DCQO+ “zero-depth” trotter knobs** are implemented as *discretization/ordering-only* changes:
- Baseline knobs reproduce baseline time grid/ordering metadata.
- Midpoint changes schedule sampling only (depth and ordering signatures unchanged).
- Forward/reverse alternates ordering deterministically.
- K-permutation changes ordering across steps/batches without changing the term set or depth.

## Why this matters
- Gate 0 establishes that the objective, representations, and classical loop are correct.
- Gate 2 ensures the GPU acceleration path is faithful to CPU (no silent drift).
- The trotter knob tests ensure DCQO+ remains a fair “zero additional depth” upgrade.
