# Product Requirements Document (PRD)

**Project Name:** LABS SeedLab (Quantum-Seeding + GPU-MTS)  
**Team Name:** **avoQados**  
**GitHub Repository:** https://github.com/benanstrom/2026-NVIDIA **[VERIFY / UPDATE IF DIFFERENT]**

---

> This PRD is written for Phase 1 (Due 10pm ET Sat Jan 31). It focuses on a *credible engineering plan*:
> a modular hybrid quantum–classical architecture, a concrete GPU acceleration path, and a verification strategy that
> prevents "it ran once" demos.

---

## 1. Team Roles & Responsibilities

| Role | Name | GitHub Handle | Discord Handle |
| :--- | :--- | :--- | :--- |
| **Technical Lead / QA** (Quantum implementation, verification) | Katya Mijatovic | ymijatov | katyam29 |
| **Project Lead / GPU Acceleration** (Architecture, GPU porting, presentation) | Ben Anstrom | benanstrom | WhaleParty |
| **AI Collaborator** (Code generation, debugging, analysis) | Claude — Anthropic Opus 4.5 | — | — |

---

## 2. The Architecture
**Owner:** Katya Mijatovic

### Choice of Quantum Algorithm
**Algorithm family:** *Quantum seeding engines* that generate candidate bitstrings for LABS, feeding a fixed classical Memetic Tabu Search (MTS) backend.

**Seeders we will support (plug-and-play):**
1. **Random seeding** — sanity baseline
2. **DCQO-based seeding** — challenge-endorsed template baseline (Trotterized counteradiabatic circuit)
3. **DCQO+ seeding** — engineering ablations on DCQO (midpoint discretization, Pauli ordering variants)
4. **PCE-based seeding** (Pauli Correlation Encoding) — *stretch goal*, qubit-efficient alternative encoding

**Motivation (why this architecture):**
- The NVIDIA LABS challenge explicitly frames the quantum part as a *seeding primitive* for MTS rather than a stand-alone quantum optimizer.
- Treating "quantum seeding" as the only experimental axis gives clean attribution: improvements in outcomes can be tied to the seeder, not to hidden classical tuning.
- This matches how hybrid methods are evaluated in the literature: time-to-solution, success probability, scaling with N, and seed diversity.
- Our Phase 1 experiments confirmed this framing: at N=12, MTS dominates regardless of initialization, but quantum seeding provides measurably better initial populations via post-selection, converging in 0 generations vs 2.2 for random seeding.

### Literature Review
We anchor our plan on three papers that define (1) the baseline pipeline, (2) the mechanism behind the baseline quantum circuit family, and (3) a potential upgrade path.

1. **Gómez-Cadavid et al. (2025)** — *"Scaling advantage with quantum-enhanced memetic tabu search for the Low Autocorrelation Binary Sequences problem"* (arXiv:2511.04553)  
   **Relevance:** Challenge-endorsed template: quantum sampling → seed population → MTS; defines evaluation metrics and overall structure.

2. **Hegade et al. (2022)** — *"Digitized Counterdiabatic Quantum Optimization"* (Phys. Rev. Research)  
   **Relevance:** Theoretical justification for DCQO-style circuits as a low-depth sampling primitive: counterdiabatic terms suppress diabatic transitions and improve the *distribution* of sampled solutions.

3. **Sciorilli et al. (2025)** — *"A competitive NISQ and qubit-efficient solver for the LABS problem"* (Pauli Correlation Encoding, PCE)  
   **Relevance:** Stretch-goal direction: PCE proposes a qubit-efficient alternative encoding for LABS, potentially producing higher-quality seeds at comparable depth. Will explore if time permits after core deliverables are complete.

### Phase 1 Findings That Inform Phase 2

Our Phase 1 implementation and experiments produced several findings that directly shape the Phase 2 plan:

**Finding 1: Counteradiabatic theta values are very small at N≤12.**
The annealing schedule produces peak theta values of ~0.01, insufficient to evolve the quantum state meaningfully away from |+⟩^N. This is consistent with the paper's prediction that quantum advantage emerges at N≥27 and crossover occurs at N≈47.

**Finding 2: Raw quantum distribution mean energy is worse than random at N=12.**
With n_steps=5: quantum mean=81.9 vs random mean=68.0. More Trotter steps accumulate small biases toward correlated (high-energy) states. However, the distribution's *tail* contains good solutions.

**Finding 3: Post-selected quantum seeds outperform random initialization.**
Using lowest-energy selection from 1000 quantum shots, the initial population starts at E=10.0 (the optimum) vs random's E=22.0, converging in 0 generations vs 2.2.

**Finding 4: CPU simulation is the bottleneck.**
N=20 statevector simulation (~2,280 gate blocks per Trotter step across 2^20 amplitudes) did not complete on the qBraid CPU sandbox. GPU acceleration is not optional — it is required to reach the problem sizes where quantum seeding has a chance to show advantage.

**DCQO+ ablation variants (Phase 2 engineering experiments):**
- Midpoint time discretization of the CD schedule
- Forward/reverse Pauli ordering per step (deterministic averaging)
- K-permutation averaged Pauli ordering across runs (randomized product formula "lite")

These are treated as controlled ablations — same circuit depth, potentially better theta schedules.

---

## 3. The Acceleration Strategy
**Owner:** Ben Anstrom

### Quantum Acceleration (CUDA-Q)

**Strategy:**

**GPU simulator backend:** Migrate the CUDA-Q quantum seeding stage from CPU statevector simulation to NVIDIA GPU backends (`nvidia`, with optional `nvidia-mgpu`) to accelerate statevector evolution and sampling without changing circuit logic. This is the single highest-impact change — our Phase 1 experiments showed that CPU simulation could not complete N=20 within reasonable time, while the paper's results go to N=67.

**Throughput batching:** Generate seeds by batching many independent circuit executions (shots, parameter points, or seed generations) per job to maximize GPU utilization and increase samples/sec. Target: 1000+ shots at N=20-30 in under 60 seconds.

**Parameter sweep acceleration:** The counteradiabatic circuit requires computing theta values per Trotter step. With GPU backends, we can sweep over Trotter step counts (n_steps = 5, 10, 20, 50) and total evolution times (T = 1, 3, 5, 10) to find the parameter regime where quantum seeding actually outperforms random — something we could not explore on CPU.

**Target behavior:**
- For each (N, seeder, depth) condition, generate a fixed number of seeds (e.g., 20–200) as the MTS initial population.
- Measure quantum sampling time separately from MTS time for clean attribution.

### Classical Acceleration (MTS)
The tutorial MTS spends most time in repeated energy evaluation and local search neighbor scoring. We will GPU-accelerate the most expensive inner loops first.

**Strategy (prioritized):**

1. **Batched GPU energy evaluation:** Rewrite the LABS aperiodic-autocorrelation energy function as a batched cupy kernel so hundreds of sequences can be scored per GPU launch, with all data kept resident on device.

2. **Parallel tabu neighbor scoring:** In each tabu iteration, evaluate thousands of single-bit-flip candidates simultaneously on GPU and select the best admissible move via a GPU reduction instead of sequential CPU scoring.

3. **Incremental (delta) energy updates:** When feasible, update energies using delta formulas for bit flips to avoid full autocorrelation recomputation, further reducing per-iteration cost.

4. **Parallel multi-start MTS:** Run many independent MTS trajectories concurrently in GPU batches, exploiting the embarrassingly parallel structure of multi-start tabu search to improve wall-clock time.

### Hardware Targets
* **Dev Environment (Phase 1):** qBraid CPU environment for algorithm development, tutorial completion, and correctness validation.
* **Acceleration Environment (Phase 2):** NVIDIA GPUs provisioned via Brev, starting with an available L4/T4-class GPU for kernel bring-up and profiling, and scaling to a high-memory A100-class GPU (subject to availability) for final large-N benchmarks.

---

## 4. The Verification Plan
**Owner:** Katya Mijatovic

### Phase 1 Verification (Completed)

**Hand calculations:** Verified LABS energy function by hand for N=4 (s=(+1,+1,-1,+1), E=2) and cross-referenced with code.

**Symmetry verification:** Using the NVIDIA interactive widget for N=7, confirmed negation invariance (E=91↔91), reversal invariance (E=35↔35), and combined invariance (E=11↔11).

**CUDA-Q kernel unit tests:** 7 tests on qBraid verifying R_YZ, R_ZY, R_YZZZ, R_ZYZZ, R_ZZYZ, R_ZZZY kernels — identity at θ=0, deterministic flips at θ=π, correct qubit targeting at θ=π/2, circuit structure via cudaq.draw().

**Interaction index verification:** G2/G4 index generation verified at N=4 (hand-checked against Eq. 15), N=7, N=10, N=20. Sanity checks: all indices in range, correct sizes, no duplicates.

**Quantum vs classical comparison:** Systematic parameter sweep at N=7 and N=12, documenting that quantum signal is weak at small N (consistent with paper). Post-selection analysis showed quantum distribution tail contains optimal solutions.

**MTS convergence:** Verified at N=7 (E=3), N=12 (E=10 across all 5 trials), N=20 (E=26).

### Phase 2 Verification Strategy

**Framework:** `pytest` (plus optional property-based checks)

**AI Hallucination Guardrails:**
- Any AI-generated or heavily refactored kernel must pass:
  1. **Property checks** (symmetries, invariants),
  2. **Small-N brute force** agreement,
  3. **CPU–GPU equivalence** within tolerance.

**Check 1 (Symmetry invariants):**
- Negation invariance: `E(s) == E(-s)`
- Reversal invariance: `E(s) == E(reverse(s))`
- Combined: `E(s) == E(-reverse(s))`

**Check 2 (Ground truth via brute force for tiny N):**
- For N ≤ 10, brute force all sequences and assert our energy function returns the known optimum and matches exact enumeration.

**Check 3 (CPU vs GPU parity):**
- For a random batch of sequences, assert CPU and GPU energy implementations match exactly (integer energy) or within strict tolerance.

**Check 4 (Seeding interface sanity):**
- Each seeder must output: correct shape `(num_seeds, N)`, values in `{+1, -1}`, reproducibility under fixed RNG seed.

**Check 5 (MTS behavioral regression):**
- With fixed RNG seeds, compare CPU and GPU MTS runs to ensure convergence behavior is consistent and final energies are comparable or improved.

---

## 5. Execution Strategy & Success Metrics
**Owner:** Ben Anstrom

### Agentic Workflow

**Human + AI collaboration model:**

Our team uses Claude (Anthropic Opus 4.5) as a collaborative coding partner, as explicitly encouraged by the challenge guidelines. The workflow is:

1. **Human-driven design:** Team members define the approach, select algorithms, and make architectural decisions. Claude does not drive strategy.
2. **AI-assisted implementation:** Claude generates initial code for well-specified tasks (kernel implementations, test harnesses, visualization functions). All generated code is reviewed and tested by team members before inclusion.
3. **Collaborative debugging:** When code fails, debugging is a dialogue — human provides error output, AI proposes fixes, human verifies. Example: the cudaq `SampleResult` API mismatch (`.get()` vs `.count()`) was caught by human testing and fixed collaboratively.
4. **AI as analytical partner:** Claude helped interpret experimental results (e.g., diagnosing why quantum mean energy was worse than random, connecting findings to the paper's predictions about scaling).
5. **Human quality gate:** No AI-generated code ships without human verification. The self-validation section documents every test performed.

This workflow is documented transparently because the challenge values it: "This isn't a limitation — it's a feature."

### Phase 2 Timeline (10pm Sat → 10am Sun)

| Time Block | Focus | Owner | Deliverable |
| :--- | :--- | :--- | :--- |
| 10pm–12am | GPU environment setup + MTS energy kernel port | Ben | Batched energy eval running on Brev GPU |
| 10pm–12am | Clean notebook, push Phase 1 code to repo | Katya | Complete Phase 1 notebook + self-validation |
| 12am–3am | GPU MTS inner loop (tabu neighbor scoring) | Ben | GPU-accelerated MTS end-to-end |
| 12am–3am | DCQO+ ablations (theta schedule variants) | Katya | DCQO+ comparison data |
| 3am–6am | Scale experiments: N=20, 25, 30 (stretch: N=40) | Both | Benchmark data + comparison plots |
| 6am–8am | PCE exploration (stretch goal only) | Katya | PCE seeds if feasible, skip if not |
| 8am–10am | Final plots, presentation, README | Both | Submission-ready repo + presentation |

### Success Metrics

We report results as **head-to-head comparisons under identical MTS budgets**.

**Committed Deliverables:**

**Metric 1 (GPU speedup):**
Demonstrate ≥5× speedup in energy evaluation throughput (sequences/sec) and/or end-to-end MTS wall-clock time, compared to CPU tutorial baseline, for the same N.

**Metric 2 (Scale):**
Successfully run end-to-end quantum-seeded MTS experiments for N=20–30, producing reproducible plots and test logs. (Phase 1 was limited to N=12 on CPU.)

**Metric 3 (Seeder comparison):**
Produce a clean comparison of Random vs DCQO vs DCQO+ seeding, holding MTS constant, with convergence curves and initial/final energy distributions.

**Stretch Goals:**

**Metric 4 (Large N):**
Reach N=40+ with end-to-end experiments (dependent on GPU memory and simulation time).

**Metric 5 (PCE seeds):**
Implement PCE-based seeding and include in comparison. Only attempted after committed deliverables are complete.

**Metric 6 (Quantum advantage signal):**
Show that at N≥25, quantum-seeded MTS finds better solutions (lower median energy) than random-seeded MTS under the same time budget.

### Visualization Plan
**Plot 1:** Best-so-far energy vs generation, comparing seeders under identical MTS budget (extension of our Phase 1 convergence plot to larger N).  
**Plot 2:** CPU vs GPU end-to-end runtime (or sequences/sec) vs N.  
**Plot 3:** Initial vs final energy bar chart by seeder method (extension of Phase 1 four-bar visualization).  
**Plot 4 (stretch):** Seed diversity (Hamming distance distribution) vs seeder; correlate diversity with time-to-solution.

---

## 6. Resource Management Plan
**Owner:** Ben Anstrom

**Plan:**
- Phase 1 developed and validated entirely on qBraid CPU:
  - All symmetry/invariant checks pass
  - Seeder interface stable
  - Baseline comparisons run end-to-end for N=12
  - Self-validation documentation complete
- Phase 2 uses Brev GPUs only for:
  - Porting inner-loop kernels (energy + neighbor scoring)
  - GPU-targeted CUDA-Q quantum simulation for N≥20
  - Collecting benchmark runs (pre-defined matrix of N and seeders)
  - Final plots and presentation data
- Credits conserved by: developing/debugging on smallest available GPU, running final benchmarks as a single planned sweep, avoiding open-ended exploration on GPU instances.