# TEST_SUITE.md — NVIDIA LABS Hybrid Solver (iQuHACK 2026)

This test suite establishes **Gate 0 (CPU-only correctness)** and supports **Gate 2 (CPU/GPU parity)**. 8 test files, 29 test functions, 702 lines.

## How to run

```bash
cd labs_hybrid_solver
pytest
```

GPU tests skip automatically if CUDA/CuPy are unavailable, so Gate 0 always passes on any machine.

## What each test proves

### `test_energy_smallN_bruteforce.py` (3 tests)
- Uses an independent reference implementation of the LABS energy to validate correctness for tiny N (≤ 8).
- Validates batch energy matches per-sequence energy.
- Validates invariance under global sign flip.

### `test_energy_cpu_gpu_parity.py` (3 tests)
- Compares batched LABS energy computed on CPU (NumPy) vs GPU (CuPy RawKernel).
- Confirms the raw CUDA kernel matches the Python loop fallback.
- Skips automatically if CUDA/CuPy are unavailable.
- **Parity philosophy:** CPU is the oracle; GPU must match exactly (energy is integer).

### `test_delta_energy_parity.py` (3 tests)
Validates each of the three incremental CUDA kernels against CPU reference:
- **`test_autocorr_kernel_matches_cpu`**: Kernel 1 (autocorrelation computation) matches NumPy autocorrelation.
- **`test_delta_energy_matches_naive_full_eval`**: Kernel 2 (delta-energy for all flip positions) matches naive full-eval recomputation.
- **`test_update_autocorr_is_consistent`**: Kernel 3 (in-place autocorrelation update after flip) produces consistent state — autocorr after update + flip matches fresh recomputation.

Skips if CUDA is unavailable.

### `test_seeders_validity.py` (4 tests)
- Validates each seeder returns shape `(K_out, N)`, dtype `int8`, values in `{−1, +1}`.
- **Determinism**: if CUDA-Q is not available or explicitly seeded, outputs repeat bit-for-bit for identical inputs.
- **CUDA-Q seeding assertion**: when CUDA-Q is available, verifies that the seeder produces valid output (assertion, not skip).
- **Different seeds differ**: two runs with different RNG seeds produce different output sequences.

### `test_mts_sanity.py` (4 tests)
- **`test_mts_improves_or_matches_best_seed_cpu`**: CPU MTS returns a solution with final energy ≤ best seed energy.
- **`test_mts_gpu_delta_improves_or_matches_best_seed`**: GPU delta-energy path also improves on seeds.
- **`test_mts_gpu_delta_vs_naive_parity`**: GPU delta path and naive GPU path produce identical final energies (bit-for-bit).
- **`test_mts_cpu_gpu_convergence_parity`**: CPU and GPU MTS produce identical convergence traces under deterministic settings.
- **`test_mts_cpu_stagnation_early_exit`**: stagnation detection window triggers early exit when no improvement is found.

### `test_dcqo_plus_trotter_knobs.py` (4 tests)
Validates the **DCQO+ "zero-depth" trotter knobs** are implemented as *discretization/ordering-only* changes:
- Baseline knobs reproduce baseline time grid/ordering metadata.
- Midpoint changes schedule sampling only (depth and ordering signatures unchanged).
- Forward/reverse alternates ordering deterministically.
- K-permutation changes ordering across steps/batches without changing the term set or depth.

### `test_selectors.py` (9 tests)
Validates post-selection correctness and edge cases:
- **Tail extraction**: selected subset contains exactly the K lowest-energy sequences.
- **Quality improvement**: mean energy after selection ≤ mean energy before.
- **Sequence preservation**: no sequences are mutated during selection.
- **K_select = K_in**: edge case where all sequences are selected works correctly.
- **K_select ≤ 0**: raises ValueError.
- **Info dict**: returned metadata contains correct min/median/mean energy statistics.
- **Diversity metrics**: Hamming distance computation is correct (verified against manual calculation).
- **Diversity range**: mean pairwise Hamming is within [0, N/2] bounds.
- **Identical sequences**: diversity metric returns 0 for duplicated inputs.

## Why this matters
- **Gate 0** establishes that the objective, representations, and classical loop are correct on any machine.
- **Gate 2** ensures the GPU acceleration path is faithful to CPU (no silent drift). Three layers of GPU parity: energy kernel, delta-energy kernels, full MTS convergence.
- The trotter knob tests ensure DCQO+ remains a fair "zero additional depth" upgrade.
- Post-selection tests guarantee the seed-quality filtering step is correct and doesn't introduce bugs.
