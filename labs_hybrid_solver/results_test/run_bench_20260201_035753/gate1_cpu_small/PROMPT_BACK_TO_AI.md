# PROMPT_BACK_TO_AI

Paste this file (or the zipped handoff bundle) back into the code-generating AI chat.

## 1) Verify correctness regressions FIRST
- Check `pytest` (Gate 0) and CPU/GPU energy parity tests (Gate 2).
- Fix correctness before optimizing performance.

## 2) Diagnose bottlenecks using runtime breakdown fields
- Use `metrics.csv` fields: t_seeder_compile_s, t_seeder_sampling_s, t_postselect_s, t_mts_total_s.
- Identify dominant components and propose improvements that do NOT change the fixed MTS logic/budgets.

## 3) Propose the SMALLEST code edits
- Preserve the fixed-vs-variable rule: do not tune/modify MTS per seeder.
- Seeders may change; selection stays energy-based.

## Key metrics (best seen per seeder)
- dcqo: best_final_energy=10.0, best_seed_energy=10, t_mts_total_s=1.000
- dcqo_plus: best_final_energy=10.0, best_seed_energy=10, t_mts_total_s=1.000
- qaoa: best_final_energy=10.0, best_seed_energy=10, t_mts_total_s=1.000
- random: best_final_energy=10.0, best_seed_energy=14, t_mts_total_s=1.000

## Run config
```json
{
  "env": {
    "platform": "Linux-6.8.0-1045-gcp-x86_64-with-glibc2.35",
    "python": "3.12.12"
  },
  "gate": {
    "N_list": [
      12,
      16
    ],
    "budget_s": 1.0,
    "device": "cpu",
    "max_iters": null,
    "name": "gate1_cpu_small",
    "seeders": [
      {
        "K_out": 256,
        "K_select": 32,
        "name": "random",
        "params": {},
        "shots": 256
      },
      {
        "K_out": 256,
        "K_select": 32,
        "name": "qaoa",
        "params": {
          "p": 1
        },
        "shots": 256
      },
      {
        "K_out": 256,
        "K_select": 32,
        "name": "dcqo",
        "params": {
          "steps": 24
        },
        "shots": 256
      },
      {
        "K_out": 256,
        "K_select": 32,
        "name": "dcqo_plus",
        "params": {
          "steps": 24,
          "trotter_config": {
            "K_perm": 1,
            "ordering": "fixed",
            "time_grid": "right_endpoint"
          }
        },
        "shots": 256
      }
    ]
  },
  "rng_seed": 0,
  "run_tag": "bench"
}
```
