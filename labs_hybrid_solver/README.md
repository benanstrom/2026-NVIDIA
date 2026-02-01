# NVIDIA LABS Hybrid Solver (iQuHACK 2026) — Phase 2 Codebase

This repository implements a **fixed classical backbone** for the Low Autocorrelation Binary Sequences (LABS) problem and exposes **interchangeable quantum seeders** as the *only experimental axis*.

**Non-negotiable architecture (do not violate):**

> **Seeder (Random/QAOA/DCQO/DCQO+/PCE)** → (optional) **energy post-selection** → **fixed Memetic Tabu Search (MTS)** (+ GPU acceleration) → metrics/logs/plots

We treat QE-MTS/DCQO as one seeding baseline. All "twist" ideas (PCE, DCQO+) are implemented as **parallel seeders feeding the same fixed MTS**.

## LABS objective
Sequences are represented internally as `±1` int8 arrays:

- Aperiodic autocorrelation: `C_k = Σ_{i=0}^{N-k-1} s_i s_{i+k}`
- Energy: `E = Σ_{k=1}^{N-1} C_k^2` (minimize)
- Merit factor: `F = N^2 / (2E)` (maximize)

## Repo layout
```
labs_hybrid_solver/
  README.md
  pyproject.toml                      # build config (Python >= 3.10)
  requirements.txt
  bench_energy_kernel.py              # GPU energy kernel microbenchmark
  bench_delta_mts.py                  # GPU delta-energy MTS microbenchmark
  01_Run_LABS_Hybrid_Solver.ipynb     # interactive notebook demo
  Claude-revisions.md                 # revision tracker (9 items, all done)

  src/labs_hybrid/
    __init__.py                       # package exports
    config.py                         # RunConfig, BenchmarkPlan, gate definitions
    types.py                          # Array, EnergyFn, Device type aliases
    rng.py                            # RNG utilities, pm1 sampling
    logging_utils.py                  # metrics I/O (CSV, JSONL, run dirs)
    labs_energy_cpu.py                # CPU energy oracle (NumPy)
    labs_energy_gpu.py                # GPU energy (CuPy RawKernel)
    selectors.py                      # post-selection by energy, Hamming diversity
    pipeline.py                       # seeder -> post-select -> MTS orchestration
    benchmarks.py                     # gate runner, OOM guard, summaries
    plots.py                          # matplotlib visualizations

    seeders/
      __init__.py                     # seeder registry (get_seeder, list_seeders)
      random_seeder.py                # IID +/-1 baseline
      qaoa_seeder.py                  # CUDA-Q QAOA with 2-local proxy cost
      dcqo_seeder.py                  # digitized counterdiabatic (Lie-Trotter)
      dcqo_plus_seeder.py             # DCQO + zero-depth Trotter mods
      pce_seeder.py                   # Pauli Correlation Encoding (Sciorilli 2025)

    mts/
      __init__.py                     # MTS dispatcher (cpu/gpu)
      mts_cpu.py                      # CPU reference MTS (correctness oracle)
      mts_gpu.py                      # GPU MTS with 3 CUDA kernels (delta-energy)

  scripts/
    run_experiment.py                 # single-run CLI
    run_bench_matrix.py               # gate orchestration (Gates 0-3)
    make_plots.py                     # generate plots from metrics.jsonl
    package_handoff.py                # zip handoff bundles

  tests/                              # 30 tests across 8 files
    conftest.py
    test_energy_smallN_bruteforce.py  # brute-force oracle (N <= 8)
    test_energy_cpu_gpu_parity.py     # CPU vs GPU energy parity
    test_delta_energy_parity.py       # 3 CUDA kernels vs CPU reference
    test_seeders_validity.py          # shape, dtype, determinism
    test_mts_sanity.py                # MTS convergence, CPU/GPU parity
    test_dcqo_plus_trotter_knobs.py   # zero-depth invariants
    test_selectors.py                 # post-selection correctness (9 tests)

  team_submissions/
    AI_REPORT.md                      # AI agent workflow + vibe log
    TEST_SUITE.md                     # test documentation
    PRESENTATION_OUTLINE.md           # slide plan with benchmark data

  results/                            # benchmark outputs (gitignored)
```

## Install / run
Editable install is optional (scripts work with `PYTHONPATH=src` too):

```bash
cd labs_hybrid_solver
pip install -e .
```

### Gate 0 — CPU-only correctness
```bash
pytest
```
What this proves: brute-force energy oracle tests (tiny N), invariants, seeder output validity, and MTS sanity.

### Gate 1 — CPU small-N integration
Runs Random/QAOA/DCQO/DCQO+ end-to-end on small N and writes metrics + artifacts.
```bash
python scripts/run_bench_matrix.py --tag gate1 --skip_gate0 --skip_gate2 --skip_gate3
```

### Gate 2 — GPU bring-up + parity philosophy
- CPU vs GPU energy parity tests (skips if no CUDA device)
- One GPU run for each seeder path
```bash
python scripts/run_bench_matrix.py --tag gate2 --skip_gate0 --skip_gate1 --skip_gate3
```

### Gate 3 — GPU benchmark matrix
Runs N ∈ {20,30,40} (adjustable in `config.default_benchmark_plan()`), all seeders, fixed budgets.
```bash
python scripts/run_bench_matrix.py --tag gate3 --skip_gate0 --skip_gate1 --skip_gate2
```

## Switching backend/device
- `--backend`: seeder execution preference (`cpu|gpu|auto`)
- `--device`: MTS + energy evaluation (`cpu|gpu|auto`)

Example (single run):
```bash
python scripts/run_experiment.py --N 20 --seeder dcqo_plus --device gpu --backend gpu --rng_seed 0 \
  --params '{"steps":24,"trotter_config":{"time_grid":"midpoint","ordering":"forward_reverse","K_perm":1}}'
```

## Outputs (judge-friendly)
Each run folder contains:
- `metrics.csv` + `metrics.jsonl` (one row per run)
- `artifacts/`
  - `*_seed_energies.npy` (for histograms / tail analysis)
  - `*_mts_trace.json` (best-so-far curve)
  - `*_best_seq_pm1.npy`
  - `plots/*.png` (generated by `scripts/make_plots.py`)
- **Handoff bundle** files per gate:
  - `results_summary.md`, `run_config.json`, `PROMPT_BACK_TO_AI.md`
  - `scripts/package_handoff.py` creates a timestamped `.zip`

## Notes on quantum seeders
Seeders use CUDA-Q if available. If CUDA-Q is not importable in the environment, seeders **fall back to deterministic classical samplers** so the pipeline remains runnable end-to-end.

- **QAOA**: lightweight baseline using a 2-local proxy cost for tractability.
- **DCQO**: digitized time-dependent evolution (Lie–Trotter, right-endpoint, fixed ordering).
- **DCQO+**: *zero-depth* upgrades (midpoint grid, ordering alternation, k-permutation ordering) — no extra gates, only discretization/ordering changes.
- **PCE**: paper-accurate implementation of Sciorilli et al. 2025 (arXiv:2506.17391). Uses numpy + scipy classical simulation (no CUDA-Q dependency): brickwork ansatz, relaxed LABS cost (Eq. 4), L-BFGS-B multi-restart optimization. Qubit count: n = ceil(log₄(N+1)), ~30 restarts.

### GPU acceleration
- **Energy kernel**: CuPy RawKernel with shared-memory tree reduction — 150–250× speedup over Python loop (31–40 µs constant time regardless of K/N).
- **MTS delta-energy**: three incremental CUDA kernels (autocorr, delta-energy, update) — 13–48× speedup, complexity reduced from O(K·N³) to O(K·N²).
- **OOM guard**: pre-check + exception handler for large-N CUDA-Q seeders to prevent GPU out-of-memory crashes.

## Fixed vs variable framing (attribution / fairness)
- **Fixed across experiments:** LABS objective, post-selection by energy, MTS (CPU & GPU structure), runtime/iteration budgets, logging format.
- **Only variable axis:** the seeding engine.

## References
- **QE-MTS / LABS hybrid architecture:** Cadavid et al., *Scaling advantage with quantum-enhanced memetic tabu search for LABS* (arXiv:2511.04553, 2025).
- **DCQO mechanism:** Hegade et al., *Digitized-Counterdiabatic Quantum Optimization* (arXiv:2201.00790, 2022).
- **PCE (twist seeder):** Sciorilli et al., *A competitive NISQ and qubit-efficient solver for the LABS problem* (arXiv:2506.17391, 2025).
- **GPU MTS design principles:** Zhang et al., *New Improvements in Solving Large LABS Instances Using Massively Parallelizable Memetic Tabu Search* (arXiv:2504.00987, 2025).
- **Time-dependent discretization (midpoint motivation):** Ikeda et al., *Minimum Trotterization Formulas for a Time-Dependent Hamiltonian* (Quantum 7:1168, 2023).
- **Commutator-scaling view of product-formula error:** Childs et al., *Theory of Trotter Error with Commutator Scaling* (Phys. Rev. X 11, 011020, 2021).
- **Randomized product formulas (k-permutation motivation):** Childs et al., *Faster quantum simulation by randomization* (Quantum 3:182, 2019).
