# Product Requirements Document (PRD)

**Project Name:** LABS SeedLab (Quantum-Seeding + GPU-MTS)  
**Team Name:** **[FILL ME]**  
**GitHub Repository:** https://github.com/benanstrom/2026-NVIDIA **[VERIFY / UPDATE IF DIFFERENT]**

---

> This PRD is written for Phase 1 (Due 10pm ET Sat Jan 31). It focuses on a *credible engineering plan*:
> a modular hybrid quantum–classical architecture, a concrete GPU acceleration path, and a verification strategy that
> prevents “it ran once” demos.

---

## 1. Team Roles & Responsibilities [You can DM the judges this information instead of including it in the repository]

| Role | Name | GitHub Handle | Discord Handle |
| :--- | :--- | :--- | :--- |
| **Project Lead** (Architect) | **[Katya Mijatovic** | **[ymijatov]** | **[katyam29]** |
| **GPU Acceleration PIC** (Builder) | **[Ben Anstrom]** | **[benanstrom]** | **[WhaleParty]** ||
| **Quality Assurance PIC** (Verifier) | **[Katya Mijatovic** | **[ymijatov]** | **[katyam29]** |
| **Technical Marketing PIC** (Storyteller) | **[Ben Anstrom]** | **[benanstrom]** | **[WhaleParty]** |

---

## 2. The Architecture
**Owner:** Katya Mijatovic

### Choice of Quantum Algorithm
**Algorithm family:** *Quantum seeding engines* that generate candidate bitstrings for LABS, feeding a fixed classical Memetic Tabu Search (MTS) backend.

**Seeders we will support (plug-and-play):**
1. **Random seeding** (sanity baseline)
2. **QAOA-based seeding** (canonical quantum baseline)
3. **DCQO-based seeding** (challenge-endorsed template baseline)
4. **PCE-based seeding** (Pauli Correlation Encoding) — *primary “beyond-template” contribution*


**Motivation (why this architecture):**
- The NVIDIA LABS challenge explicitly frames the quantum part as a *seeding primitive* for MTS rather than a stand‑alone quantum optimizer.
- Treating “quantum seeding” as the only experimental axis gives clean attribution: improvements in outcomes can be tied to the seeder, not to hidden classical tuning.
- This matches how hybrid methods are evaluated in the literature: time‑to‑solution, success probability, scaling with N, and seed diversity.

### Literature Review
We anchor our plan on three papers that define (1) the baseline pipeline, (2) the mechanism behind the baseline quantum circuit family, and (3) the main twist/upgrade path.

1. **Gómez‑Cadavid et al. (2025)** — *“Scaling advantage with quantum‑enhanced memetic tabu search for the Low Autocorrelation Binary Sequences problem”* (arXiv:2511.04553)  
   **Relevance:** This is the challenge‑endorsed template: quantum sampling → seed population → MTS; defines evaluation metrics and overall structure.

2. **Hegade et al. (2022)** — *“Digitized Counterdiabatic Quantum Optimization”* (Phys. Rev. Research)  
   **Relevance:** Provides the theoretical justification for DCQO‑style circuits as a low‑depth sampling primitive: counterdiabatic terms suppress diabatic transitions and improve the *distribution* of sampled solutions, not just single best shots.

3. **Sciorilli et al. (2025)** — *“A competitive NISQ and qubit‑efficient solver for the LABS problem”* (Pauli Correlation Encoding, PCE)  
   **Relevance:** Main “beyond-template” direction: PCE proposes a qubit‑efficient alternative encoding/circuit family for LABS, potentially producing higher-quality and/or more diverse seeds at comparable depth.

**Optional DCQO digitization upgrades (engineering ablation, no depth increase):**
- Midpoint time discretization of the CD schedule
- Forward/reverse Pauli ordering per step (deterministic averaging)
- K‑permutation averaged Pauli ordering across runs (randomized product formula “lite”)

These are treated as a **DCQO+** variant for controlled comparison.

---

## 3. The Acceleration Strategy
**Owner:** Ben Anstrom

### Quantum Acceleration (CUDA-Q)
**Strategy:**
FILLLLLLL THISSS INNNNNNN

**Target behavior:**
- For each (N, seeder, depth) condition, generate a fixed number of seeds (e.g., 20–200) as the MTS initial population.

### Classical Acceleration (MTS)
The tutorial MTS spends most time in repeated energy evaluation and local search neighbor scoring. We will GPU‑accelerate the most expensive inner loops first.

**Strategy (prioritized):**
-Batched GPU energy evaluation: Rewrite the LABS aperiodic-autocorrelation energy function as a batched cupy kernel so hundreds of sequences can be scored per GPU launch, with all data kept resident on device.

-Parallel tabu neighbor scoring: In each tabu iteration, evaluate thousands of single-bit-flip candidates simultaneously on GPU and select the best admissible move via a GPU reduction instead of sequential CPU scoring.

-Incremental (delta) energy updates: When feasible, update energies using delta formulas for bit flips to avoid full autocorrelation recomputation, further reducing per-iteration cost.

-Parallel multi-start MTS: Run many independent MTS trajectories concurrently in GPU batches, exploiting the embarrassingly parallel structure of multi-start tabu search to improve wall-clock time.

### Hardware Targets
* **Dev Environment (Phase 1):** qBraid CPU environment for algorithm development, tutorial completion, and correctness validation prior to acceleration.
* **Acceleration Environment (Phase 2):** NVIDIA GPUs provisioned via Brev, starting with an available L4/T4-class GPU for kernel bring-up and profiling, and scaling to a high-memory A100-class GPU (subject to availability) for final large-N benchmarks.


---

## 4. The Verification Plan
**Owner:** Katya Mijatovic

### Unit Testing Strategy
**Framework:** `pytest` (plus optional property-based checks)  
**AI Hallucination Guardrails:**
- Any AI-generated or heavily refactored kernel must pass:
  1. **Property checks** (symmetries, invariants),
  2. **Small-N brute force** agreement,
  3. **CPU–GPU equivalence** within tolerance (when using floats).

### Core Correctness Checks
**Check 1 (Symmetry invariants):**
- **Negation invariance:** LABS energy must satisfy `E(s) == E(-s)`.
- **Reversal invariance:** LABS energy must satisfy `E(s) == E(reverse(s))`.
- **Negation+reversal invariance:** `E(s) == E(-reverse(s))`.

These are the fastest “physics-style” correctness signals and are explicitly highlighted in the tutorial notebook.

**Check 2 (Ground truth via brute force for tiny N):**
- For **N ≤ 10**, brute force all sequences and assert our energy function returns the known optimum energy and matches exact enumeration for random samples.
- Example smoke test: for **N=3**, the minimum energy is **1** and sequence `[1, 1, -1]` achieves it.

**Check 3 (CPU vs GPU parity):**
- For a random batch of sequences, assert CPU and GPU energy implementations match exactly (integer energy) or within a strict tolerance (if using float intermediate).

**Check 4 (Seeding interface sanity):**
- Each seeder must output:
  - correct shape: `(num_seeds, N)`
  - values restricted to `{+1, -1}`
  - reproducibility under fixed RNG seed (where applicable)

**Check 5 (MTS behavioral regression):**
- With fixed RNG seeds and identical parameters, compare CPU and GPU MTS runs at moderate N to ensure best-energy-vs-iteration behavior is consistent and final energies are comparable or improved.


---

## 5. Execution Strategy & Success Metrics
**Owner:** Ben Anstrom

### Agentic Workflow
**Plan (human + AI collaboration):**


### Success Metrics
We will report results as **head-to-head comparisons under identical MTS budgets**.

**Metric 1 (Seed usefulness / optimization quality):**
- Under a fixed runtime budget (e.g., 60s per run), show that at least one quantum seeder (DCQO or PCE) improves **median best energy** and/or **median merit factor** vs random seeding for **N = 20–30**.

**Metric 2 (Speedup from GPU):**
- Demonstrate ≥ **5×** speedup in either:
  - energy evaluation throughput (seq/s), or
  - end-to-end MTS wall-clock to reach a threshold,
  compared to CPU tutorial baseline for the same N.

**Metric 3 (Scale):**
- Successfully run end-to-end experiments for **N = 40** (or highest achievable N within credits), producing reproducible plots and test logs.

**Metric 4 (Ablation clarity):**
- Produce a clean table/plot comparing:
  - Random vs QAOA vs DCQO vs PCE (and optionally DCQO+),
  holding MTS constant.

### Visualization Plan
**Plot 1:** Best-so-far energy (or merit factor) vs time, comparing seeders under identical MTS budget.  
**Plot 2:** CPU vs GPU end-to-end runtime (or seq/s) vs N.  
**Plot 3 (bonus):** Seed diversity (Hamming distance distribution) vs seeder; correlate diversity with time-to-solution.

---

## 6. Resource Management Plan
**Owner:** Ben Anstrom

**Plan:**
- Develop and validate entirely on qBraid CPU until:
  - invariants + brute-force checks pass,
  - seeder interface is stable,
  - baseline comparisons run end-to-end for N=20.
- Use Brev GPUs only for:
  - porting the inner-loop kernels (energy + neighbor scoring),
  - collecting a small number of benchmark runs (pre-defined matrix of N and seeders),
  - final plots.
