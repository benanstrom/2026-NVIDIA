# PRESENTATION_OUTLINE.md — Slide plan (iQuHACK 2026)

1. **Problem + objective**
   - LABS: minimize energy (sum of squared aperiodic autocorrelations)
   - Why it's hard / why it's a benchmark

2. **Architecture (non-negotiable)**
   - Seeder → post-select → fixed MTS (GPU accelerated) → logs/plots
   - "Only variable axis = seeder" framing

3. **Fixed classical backbone**
   - Memetic Tabu Search (MTS) summary
   - GPU acceleration: three custom CUDA kernels (autocorr, delta-energy, update)
   - Complexity reduction: O(K·N³) → O(K·N²), 13–48× speedup

4. **Seeders overview**
   - Random baseline
   - QAOA baseline
   - DCQO baseline (Cadavid/Hegade)
   - DCQO+ upgrades (zero depth)
   - PCE twist seeder (Sciorilli)

5. **DCQO vs DCQO+**
   - Baseline trotter: Lie–Trotter, right-endpoint, fixed ordering
   - Upgrades:
     - midpoint time grid
     - forward/reverse ordering alternation
     - k-permutation ordering
   - "No extra gates" rationale
   - GPU delta-energy optimization: 13–48× speedup via incremental CUDA kernels

6. **PCE twist seeder**
   - Pauli Correlation Encoding idea (Sciorilli et al. 2025)
   - Qubit-efficiency: n = ceil(log₄(N+1)) qubits
   - Paper-accurate implementation: brickwork ansatz, relaxed LABS cost (Eq. 4), L-BFGS-B multi-restart optimization
   - Benchmark: raw energy min 24 vs 32–36 for QAOA/DCQO at N=16
   - numpy + scipy classical simulation (no CUDA-Q dependency)

7. **GPU acceleration results**
   - Energy kernel: CuPy RawKernel with shared-memory tree reduction — 150–250× speedup (31–40 µs constant time)
   - MTS delta-energy: three incremental CUDA kernels — 13–48× speedup
   - OOM guard for large-N CUDA-Q seeders
   - Benchmarks from L4 GPU

8. **Evaluation protocol**
   - Fixed budgets, fixed MTS params, fixed metrics
   - Post-selection captures left-tail advantage from seed distribution

9. **Results — Full Benchmark Summary**

   **Gate pipeline (all gates pass):**
   - Gate 0: 29 unit tests pass (7.2s)
   - Gate 1: CPU small-N (N=12,16) — all seeders converge, best energy 10/24
   - Gate 2: GPU bringup (N=16) — all 5 seeders converge to 24
   - Gate 3: GPU matrix (N=20,30,40) — full seeder comparison, 3s MTS budget end slide

   **Gate 3 — Seed quality (best raw seed energy before MTS):**

   | N | Random | QAOA | DCQO | DCQO+ | PCE |
   |---|--------|------|------|-------|-----|
   | 20 | 58 | 62 | 66 | 66 | **34** |
   | 30 | 179 | 155 | 171 | 171 | **83** |
   | 40 | 316 | — | — | — | **184** |

   → PCE produces seeds ~2× better than random across all N. end slide

   **Gate 3 — Final energy after MTS (3s GPU budget):**

   | N | Random | QAOA | DCQO | DCQO+ | PCE |
   |---|--------|------|------|-------|-----|
   | 20 | 26 | 26 | 26 | 26 | 26 |
   | 30 | 59 | 59 | 59 | 59 | 59 |
   | 40 | 116 | — | — | — | **108** |

   → At N=20,30, MTS closes the gap — all seeders reach the same final energy.
   → At N=40, **PCE's advantage survives MTS**: 108 vs 116 (7% better). MTS can't fully compensate for worse seeds within the 3s budget at larger N.

   **Seeder runtime (not counted against MTS budget):**

   | Seeder | N=20 | N=30 | N=40 |
   |--------|------|------|------|
   | Random | <1 ms | <1 ms | <1 ms |
   | QAOA | 0.11s | 0.98s | — |
   | DCQO | 1.88s | 22.1s | — |
   | DCQO+ | 1.90s | 22.3s | — |
   | PCE | 46.4s | 63.9s | 58.7s |

   → PCE seeder time is the tradeoff for superior seed quality. DCQO compile time dominates (~60% of seeder time). Random is essentially free.

   **Key insight:** Better seeds matter more as N grows. At small N, MTS is powerful enough to overcome any starting point. At N=40, the quality gap from PCE seeds translates into a measurably better final result. This trend should amplify at larger N.

   **Plots to show:**
   - Seed energy distributions by seeder (histogram/violin — PCE's left tail)
   - Final energy vs N by seeder (the N=40 divergence)
   - Seeder time vs seed quality scatterplot (cost-benefit)
   - MTS convergence traces (time-to-best curves)

10. **AI-assisted development**
    - Claude Code (Opus 4.5) as primary coding agent
    - Claude-revisions.md as living review document (9 items tracked → all completed)
    - Plan mode for complex tasks: detailed spec → correct implementation on first pass
    - Gate-based verification pipeline: CPU correctness → GPU parity → benchmarks

11. **Takeaways + next steps**
    - **PCE is the best seeder**: ~2× better raw seeds at every N, and the only seeder to beat random on final energy at N=40 (108 vs 116)
    - **Seed quality matters more at scale**: at small N, MTS equalizes all seeders; at N=40, PCE's advantage survives the 3s budget — this gap should widen at larger N
    - **GPU acceleration is critical**: 150–250× energy kernel speedup and 13–48× MTS delta-energy speedup make the 3s budget viable for N=40
    - **DCQO ≈ DCQO+**: zero-depth trotter upgrades show no measurable advantage in seed quality or final energy — the discretization changes don't shift the distribution
    - Next: scale to N=60+, amortize PCE seeder cost (cache operator sets), calibrate on CUDA-Q hardware backends
