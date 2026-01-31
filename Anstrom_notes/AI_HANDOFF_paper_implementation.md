# AI Handoff Document
**Project:** NVIDIA LABS Challenge — Quantum-Seeded GPU-Accelerated LABS Solver  
**Purpose:** Preserve design intent, experimental framing, and contribution logic for downstream AI agents or collaborators.

---

## 1. Context and Intent

The NVIDIA LABS challenge explicitly references the 2025 work by Gómez-Cadavid et al. on **Quantum-Enhanced Memetic Tabu Search (QE-MTS)** for the Low Autocorrelation Binary Sequences (LABS) problem as a recommended template.

This work is treated as a **strong baseline**, not the final contribution.

The objective of this project is **not** to reproduce or slightly modify the Gómez-Cadavid QE-MTS pipeline, but to **go beyond it** by isolating and evaluating alternative quantum seeding strategies under strictly controlled classical conditions.

---

## 2. Core Design Decision (Critical)

**We intentionally decouple the quantum and classical components.**

Instead of embedding new ideas inside the Gómez-Cadavid QE-MTS loop, the system is structured so that:

- The **classical optimizer is fixed**
- The **quantum seeding mechanism is the only changing variable**

This ensures clean attribution of performance gains.

---

## 3. Fixed Components (Must NOT Change Across Experiments)

The following are held constant across all experiments:

- **Classical optimizer:** Memetic Tabu Search (MTS)
- **Implementation:** GPU-accelerated MTS
- **Runtime budgets:** identical per run
- **Stopping criteria:** identical
- **Evaluation metrics:** identical
- **Hardware:** same GPU class when comparing results

Any deviation from these invalidates comparisons.

---

## 4. Variable Component (The Experimental Axis)

**Only the quantum seeding strategy changes.**

All seeders output candidate LABS sequences that are passed *unchanged* into the same classical MTS pipeline.

---

## 5. System Architecture

Quantum Seeder  →  Fixed GPU-Accelerated MTS  →  Final Solution

The classical solver acts as a **controlled experimental environment**.

Quantum methods are interchangeable front-ends that propose initial candidate solutions.

---

## 6. Quantum Seeders Evaluated

All seeders are evaluated under identical conditions:

1. **Random seeding**  
   Sanity baseline; establishes classical floor.

2. **QAOA-based seeding**  
   Canonical quantum baseline.

3. **DCQO-based seeding (Gómez-Cadavid)**  
   Challenge-endorsed reference method.

4. **New twist seeding (this project’s contribution)**  
   Examples include:
   - Pauli Correlation Encoding (PCE)
   - Adaptive bias-field / feedback-biased quantum sampling

Each seeder produces the same number of candidate seeds.

---

## 7. Why This Framing Was Chosen

This architecture was selected to:

- Preserve **clean attribution** of performance gains
- Respect the challenge’s recommended approach while **going beyond it**
- Enable **rigorous ablation studies**
- Match evaluation standards in hybrid quantum–classical research
- Avoid confounding improvements due to classical tuning

---

## 8. Evaluation Methodology

Seeders are compared using:

- **Time-to-solution / best-so-far curves**
- **Final solution quality under fixed runtime**
- **Seed diversity metrics** (e.g., Hamming distance spread, basin coverage)

Primary question answered:

> *Given the same classical solver and compute budget, which quantum encoding produces the most useful starting states?*

---

## 9. Contribution Statement

The contribution of this project is a **drop-in alternative quantum seeding engine** evaluated **head-to-head** against:

- Standard QAOA seeding  
- The Gómez-Cadavid DCQO-based QE-MTS method  

This is **not** a minor modification of the reference pipeline, but a controlled, modular framework isolating the impact of quantum seeding strategies.

---

## 10. Guidance for Future AI Agents

- Do **not** modify the classical MTS when comparing seeders
- Add new quantum seeders as *parallel alternatives*
- Preserve identical budgets and metrics
- Prefer clean ablations over architectural entanglement
- Explain *why* seeds help, not just that they help

---

## 11. Red Lines (Do Not Cross)

- Do not fold new logic into MTS during comparisons
- Do not mix compute budgets across methods
- Do not claim scaling advantage without controlled evidence
- Do not conflate solver speedups with quantum advantage

---

**End of AI Handoff Document**
