# Claude Code Review — Revisions & Suggestions

**Reviewer:** Claude (Opus 4.5)
**Date:** 2026-01-31
**Scope:** `labs_hybrid_solver/` codebase assessment against PRD for Phase 2

---

## High Priority

### 1. GPU Energy Kernel Optimization (`labs_energy_gpu.py`) — DONE
- **Issue:** Serial Python loop over autocorrelation lags — mirrors CPU implementation, negates GPU benefit.
- **Action:** Replaced Python loop with a CuPy RawKernel (`labs_energy_kernel`) that computes all N-1 lag autocorrelations in a single CUDA launch. One block per sequence, one thread per lag, shared-memory tree reduction. Old loop kept as `_labs_energy_batch_gpu_fallback` for parity testing.
- **Tests:** Parameterized CPU/GPU parity tests added for (K,N) in {(1,4), (64,16), (32,40), (100,30)}. Raw-vs-fallback parity tests added. Benchmark script: `bench_energy_kernel.py`.
- **Impact:** Eliminates N-1 kernel launches and Python loop overhead. Awaiting GPU benchmark results.

### 2. GPU MTS Optimization (`mts/mts_gpu.py`)
- **Issue:** Full K×N neighborhood evaluated naively per iteration. No delta energy updates.
- **Action:** Implement incremental (delta) energy updates for single-bit flips. Batch neighbor scoring with GPU reduction.
- **Impact:** 3–10× speedup potential per PRD acceleration strategy.

### 3. PCE Seeder Completion (`seeders/pce_seeder.py`)
- **Issue:** Skeleton with placeholder ring entanglement and explicit TODOs on lines 46, 80, 135–137.
- **Action:** Implement paper-accurate Pauli correlation encoding circuit from Sciorilli et al. (2025). Replace placeholder ring entanglement + local RZ phases.
- **Impact:** Required for stretch goal; currently would skew benchmark results if included in Gate 3 matrix.

---

## Medium Priority

### 4. GPU MTS Convergence Parity Test
- **Issue:** Only energy function CPU/GPU parity is tested. No test verifies that CPU and GPU MTS produce the same (sequence, energy) under a fixed RNG seed.
- **Action:** Add `test_mts_cpu_gpu_determinism()` in test suite.
- **File:** `tests/test_mts_sanity.py` or new `tests/test_mts_cpu_gpu_parity.py`

### 5. CUDA-Q Seeding Verification
- **Issue:** `test_seeders_validity.py` skips determinism check when CUDA-Q seeding is unverified (lines 57–58).
- **Action:** Confirm `cudaq.set_random_seed()` is respected. Add bit-for-bit reproducibility test for quantum seeders.
- **File:** `tests/test_seeders_validity.py`

### 6. Gate 3 OOM Guard
- **Issue:** N=40 in Gate 3 benchmark could exceed VRAM on L4/T4 GPUs with no safeguard.
- **Action:** Add memory estimation check before launching Gate 3 runs. Make N matrix adaptive or add early exit on OOM.
- **File:** `scripts/run_bench_matrix.py`

---

## Low Priority

### 7. Post-Selection Integration Test
- **Issue:** `selectors.py` has no dedicated test verifying that post-selection extracts the low-energy tail and improves population quality.
- **Action:** Add `test_postselection_extracts_low_energy_tail()`.
- **File:** New `tests/test_selectors.py`

### 8. DCQO/QAOA Classical Fallbacks
- **Issue:** Fallback implementations when CUDA-Q is unavailable are minimal (DCQO: 3 sweeps; QAOA: simple local search). Seeds will be poor quality.
- **Action:** Acceptable for now — these only activate without GPU. Document the limitation.
- **File:** `seeders/dcqo_seeder.py`, `seeders/qaoa_seeder.py`

### 9. MTS Stagnation Detection
- **Issue:** MTS runs for full budget even when converged.
- **Action:** Add optional early-exit when no improvement seen for N consecutive iterations.
- **File:** `mts/mts_cpu.py`, `mts/mts_gpu.py`

---

## Status Tracker

| # | Item | Priority | Status |
|---|------|----------|--------|
| 1 | GPU energy kernel optimization | High | Done (awaiting GPU benchmark) |
| 2 | GPU MTS optimization (delta energy) | High | Pending |
| 3 | PCE seeder completion | High | Pending |
| 4 | GPU MTS convergence parity test | Medium | Pending |
| 5 | CUDA-Q seeding verification | Medium | Pending |
| 6 | Gate 3 OOM guard | Medium | Pending |
| 7 | Post-selection integration test | Low | Pending |
| 8 | Document fallback limitations | Low | Pending |
| 9 | MTS stagnation detection | Low | Pending |
