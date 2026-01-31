# GPU Acceleration Design Reference
## Role of Zhang et al. (2025) in the NVIDIA LABS Project

**Paper:** Zhang et al., *Massively Parallel Memetic Tabu Search for the Low Autocorrelation Binary Sequences Problem* (2025)  
**Role in this project:** Classical GPU-acceleration “bones” reference

---

## 1. Purpose of This Document

This document explains **how and why** the work of Zhang et al. (2025) is used in this project.

The intent is not to reproduce their full solver, but to extract and apply the **parallelization principles** required to build a fast, GPU-accelerated classical backend suitable for hybrid quantum–classical experiments in the NVIDIA LABS challenge.

This document exists to:
- justify architectural decisions,
- clarify attribution,
- and preserve experimental rigor.

---

## 2. Why Zhang et al. Is the Correct GPU Reference

Zhang et al. directly address the problem of making **Memetic Tabu Search (MTS) for LABS** efficient on modern NVIDIA GPUs.

Key reasons this paper is foundational:

- Focuses on **engineering scalability**, not algorithmic novelty
- Demonstrates **massively parallel MTS** on NVIDIA hardware
- Shows that classical optimization must be accelerated **before** quantum advantages become observable
- Reports large speedups and improved best-known solutions for larger N

This makes it an ideal reference for the **classical backbone** of a hybrid solver.

---

## 3. How the Paper Is Used (High-Level)

In this project, Zhang et al. informs **how the classical optimizer is structured and accelerated**, not how it is modified algorithmically.

We adopt the following ideas:

- Treat MTS as **embarrassingly parallel** across many independent runs
- Batch energy and neighborhood evaluations to enable vectorization
- Represent LABS sequences in GPU-friendly formats
- Keep classical execution deterministic and uniform across experiments

The result is a **fixed, fast classical core** that allows fair comparison of quantum seeding strategies.

---

## 4. What Is Explicitly NOT Taken from Zhang et al.

To preserve clean experimental attribution:

- We do **not** replicate their full custom CUDA kernel implementation
- We do **not** incorporate their solver-specific heuristics into MTS
- We do **not** tune classical parameters differently per quantum method

Zhang et al. is used as a **design guide**, not a drop-in solver.

---

## 5. Mapping Zhang et al. Concepts to This Project

| Zhang et al. Concept | Project Implementation |
|---------------------|------------------------|
| Multi-start parallel MTS | Batched MTS runs on GPU |
| Block-level independence | Independent MTS trajectories |
| Thread-level evaluation | Vectorized / batched energy evaluation |
| Binary sequence optimization | ±1 tensor representations |
| GPU-centric design | CuPy / CUDA-Q compatible backend |

---

## 6. Why This Matters for Quantum Evaluation

Without GPU acceleration of the classical solver:

- MTS runtime dominates total cost
- Quantum seeding differences are masked
- Scaling comparisons become meaningless

By following Zhang et al., we ensure:

- Classical optimization is **not the bottleneck**
- Any improvement is attributable to **quantum seeding**
- Comparisons to Gómez-Cadavid DCQO seeding remain fair

---

## 7. Relationship to the Gómez-Cadavid QE-MTS Paper

The two papers serve distinct roles:

- **Zhang et al. (2025):** How to make MTS fast and scalable on GPUs
- **Gómez-Cadavid et al. (2025):** How quantum methods can improve seeding and scaling

This project builds **on top of both** by combining:
- a Zhang-inspired fixed GPU classical core, and
- multiple quantum seeding strategies evaluated head-to-head.

---

## 8. Summary Statement

Zhang et al. (2025) provides the **classical acceleration foundation** for this project.

Its role is to ensure that the Memetic Tabu Search component is sufficiently fast and parallelized such that differences between quantum seeding strategies can be meaningfully measured, compared, and attributed.

---

**End of Document**
