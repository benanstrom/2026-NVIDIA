# AI_REPORT.md — NVIDIA LABS Hybrid Solver (iQuHACK 2026)

## 0) Executive summary
- Goal: solve LABS via **fixed classical MTS** with **interchangeable seeding engines**.
- Non-negotiable: do **not** embed seeder-specific tuning inside the MTS loop.
- AI agent: **Claude Code (Opus 4.5)** via CLI, used as primary coding agent for implementation, testing, benchmarking, and documentation.

## 1) Agent workflow

**Tool:** Claude Code (Anthropic CLI) running Claude Opus 4.5.

**Development cycle:**
1. Implement locally (Claude Code writes/edits code, runs tests)
2. Push to GitHub
3. Pull on Brev GPU instance (H100/L4)
4. Run benchmarks (gate pipeline)
5. Analyze results, identify gaps
6. Iterate

**Key patterns:**
- **Plan mode** for complex tasks: used for PCE implementation (detailed spec with algorithm equations, function signatures, paper references fed to Claude before coding)
- **Gate-based verification pipeline** (Gate 0 → Gate 3) structured development from CPU correctness through GPU benchmark matrix
- **Claude-revisions.md** as living review document: 9 revision items tracked from identification through completion
- **Iterative refinement**: each benchmark run surfaced issues (e.g., PCE max_restarts too high, OOM on large N) addressed in subsequent sessions

## 2) Verification strategy

**Test suite:** 8 test files, 30 test functions (702 lines).

| Layer | What it proves | Tests |
|-------|---------------|-------|
| **Brute-force oracle** | Energy function is correct for N ≤ 8 | `test_energy_smallN_bruteforce.py` (3 tests) |
| **CPU/GPU energy parity** | CuPy RawKernel matches NumPy exactly (integer energy) | `test_energy_cpu_gpu_parity.py` (3 tests) |
| **Delta-energy kernel parity** | All 3 CUDA kernels match naive full-eval | `test_delta_energy_parity.py` (3 tests) |
| **Seeder validity** | Shape (K_out, N), dtype int8, values ±1, determinism | `test_seeders_validity.py` (3 tests) |
| **MTS sanity** | CPU improves seeds, GPU delta matches naive, CPU/GPU convergence parity, stagnation early-exit | `test_mts_sanity.py` (5 tests) |
| **DCQO+ trotter knobs** | Zero-depth invariants hold across all knob combinations | `test_dcqo_plus_trotter_knobs.py` (4 tests) |
| **Post-selection** | Correct tail extraction, quality improvement, sequence preservation, edge cases, diversity metrics | `test_selectors.py` (9 tests) |

**Philosophy:** CPU is the correctness oracle. GPU must match exactly (energy is integer-valued). All GPU tests skip gracefully if CUDA is unavailable, so Gate 0 always passes on any machine.

## 3) Metrics + logging format
Each run appends one row to `metrics.csv` and `metrics.jsonl`:
- problem: N
- seeder: name, params, shots, K_out, K_select
- runtime breakdown: compile, sampling, post-select, MTS
- quality: best seed energy, best final energy
- trace summary: length of best-so-far curve, MTS iterations
- diversity (recommended): mean pairwise Hamming distance among MTS starts

## 4) Red lines (do not cross)
- **Do not tune MTS per seeder.**
- **Do not change budgets per seeder.**
- DCQO+ must be *zero added depth*: only time discretization / ordering changes.

## 5) Vibe log

### Wins
- **GPU energy kernel**: CuPy RawKernel with shared-memory tree reduction — 150–250× speedup implemented in a single session. Kernel runs in 31–40 µs constant time regardless of batch size K or sequence length N.
- **GPU MTS delta-energy**: three incremental CUDA kernels (autocorr, delta-energy, update) — 13–48× speedup. Reduced complexity from O(K·N³) to O(K·N²).
- **PCE paper implementation**: full Sciorilli et al. 2025 implementation from spec in one plan-mode session. 519 lines, paper-accurate brickwork ansatz, relaxed LABS cost, L-BFGS-B multi-restart optimization.
- **Post-selection test suite**: 9 comprehensive tests covering tail extraction, quality, preservation, edge cases, and diversity metrics.

### Learnings
- **Detailed plans yield better results**: providing algorithm equations, function signatures, and paper references (rather than vague "implement PCE") produced correct code on first pass.
- **Gate pipeline structures development**: the Gate 0→3 progression naturally catches issues early (CPU correctness before GPU bring-up before full benchmarks).
- **Living revision doc works**: Claude-revisions.md as a shared checklist between human and AI kept work focused and trackable.

### Failures / bugs
- **scipy.optimize.minimize `kwargs` bug**: passed unsupported keyword arguments to L-BFGS-B, caused silent failures. Fixed by using only supported parameters.
- **PCE max_restarts = K_out (256)**: caused 454s seeding time for N=16. Fixed by capping max_restarts to 30 — sufficient for convergence, reduces runtime significantly.
- **OOM on large-N CUDA-Q seeders**: GPU memory exhaustion with no graceful recovery. Fixed by adding pre-check and exception handler with fallback.

### Context dump
- **Plan mode with detailed spec**: for PCE, provided full algorithm description (Pauli encoding, brickwork ansatz, relaxed cost function equations, optimization parameters) before asking Claude to implement.
- **Claude-revisions.md**: living review document with 9 items, each tracked from identification through completion with benchmarks.

## 6) Notes for judges
- All five seeders (Random, QAOA, DCQO, DCQO+, PCE) are fully implemented and benchmarked through the same fixed MTS backbone.
- **PCE is paper-accurate** (Sciorilli et al. 2025): numpy + scipy classical simulation with brickwork ansatz, relaxed LABS cost, L-BFGS-B optimization. No CUDA-Q dependency — uses n = ceil(log₄(N+1)) qubits with ~30 restarts.
- **All 9 revision items completed**: GPU energy kernel, GPU MTS delta-energy, PCE completion, convergence parity test, CUDA-Q seeding verification, OOM guard, post-selection tests, fallback documentation, MTS stagnation detection.
- **GPU acceleration benchmarks** (L4 GPU): energy kernel 150–250× speedup (31–40 µs constant), MTS delta-energy 13–48× speedup (O(K·N²) vs O(K·N³)).
- **Gate 2 results** (N=16): PCE achieves raw energy minimum of 24 (vs 32–36 for QAOA/DCQO), demonstrating superior initial seed quality. All seeders converge to final energy 24 after MTS.
- **30 tests** across 8 files ensure correctness at every layer: brute-force oracle, CPU/GPU parity, kernel-level parity, seeder validity, MTS sanity, trotter knob invariants, and post-selection correctness.
