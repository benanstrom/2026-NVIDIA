# AI_REPORT.md — NVIDIA LABS Hybrid Solver (iQuHACK 2026)

## 0) Executive summary
- Goal: solve LABS via **fixed classical MTS** with **interchangeable seeding engines**.
- Non-negotiable: do **not** embed seeder-specific tuning inside the MTS loop.

## 1) Agent workflow (template)
- [ ] Implement CPU correctness first (Gate 0)
- [ ] Run CPU small-N integration (Gate 1)
- [ ] Bring up GPU path + parity (Gate 2)
- [ ] Run GPU benchmark matrix + generate plots (Gate 3)
- [ ] Package “handoff bundle” zip for judge narrative

## 2) Verification strategy
- CPU energy and MTS are the oracle.
- GPU must match CPU on energy batched parity tests.
- Seeders must output valid ±1 int8 sequences.
- Post-selection must always be based on the LABS energy function.

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

## 5) Vibe log (template)
### Wins
- 

### Learnings
- 

### Failures / bugs
- 

### Next small steps
- 

## 6) Notes for judges
- DCQO baseline and DCQO+ upgrades are implemented as independent seeders feeding the same fixed MTS backbone.
- PCE is implemented as a separate seeder module. Paper-specific details are clearly marked TODO where needed.
