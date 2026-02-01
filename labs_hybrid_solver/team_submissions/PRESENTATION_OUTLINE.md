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

9. **Results**
   - Gate 2 (N=16): PCE raw min 24 (vs 32–36 QAOA/DCQO), all seeders converge to 24 after MTS
   - Time-to-best curves by seeder
   - Final energy vs budget
   - Seed energy histogram (tail analysis)
   - Runtime breakdown CPU vs GPU

10. **AI-assisted development**
    - Claude Code (Opus 4.5) as primary coding agent
    - Claude-revisions.md as living review document (9 items tracked → all completed)
    - Plan mode for complex tasks: detailed spec → correct implementation on first pass
    - Gate-based verification pipeline: CPU correctness → GPU parity → benchmarks

11. **Takeaways + next steps**
    - PCE produces superior initial seeds (lower raw energy minimum)
    - GPU acceleration enables scaling to larger N within fixed time budgets
    - All seeders converge to same final energy after MTS — seed quality affects convergence speed
    - Next: scale N matrix, calibrate CUDA-Q backends on target hardware, explore PCE layer-depth tradeoffs
