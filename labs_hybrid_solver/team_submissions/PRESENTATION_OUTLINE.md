# PRESENTATION_OUTLINE.md — Slide plan (iQuHACK 2026)

1. **Problem + objective**
   - LABS: minimize energy (sum of squared aperiodic autocorrelations)
   - Why it’s hard / why it’s a benchmark

2. **Architecture (non-negotiable)**
   - Seeder → post-select → fixed MTS (GPU accelerated) → logs/plots
   - “Only variable axis = seeder” framing

3. **Fixed classical backbone**
   - Memetic Tabu Search (MTS) summary
   - GPU acceleration principles (Zhang-inspired: multi-start, batched evaluation, minimal transfers)

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
   - “No extra gates” rationale

6. **PCE twist seeder**
   - Pauli Correlation Encoding idea
   - Qubit-efficiency narrative
   - Our implementation status: runnable skeleton + TODO markers

7. **Evaluation protocol**
   - Fixed budgets, fixed MTS params, fixed metrics
   - Post-selection captures left-tail advantage from seed distribution

8. **Results plots**
   - time-to-best curves by seeder
   - final energy vs budget
   - seed energy histogram (tail)
   - runtime breakdown CPU vs GPU

9. **Takeaways + next steps**
   - Which seeder shifts the left tail best under fixed MTS
   - Where GPU acceleration buys scaling
   - Next: fill PCE paper constants; calibrate CUDA-Q backends; expand N matrix
