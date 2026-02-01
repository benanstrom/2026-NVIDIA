#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from labs_hybrid.config import RunConfig, SeederConfig, TabuParams
from labs_hybrid.logging_utils import new_run_dir, write_json
from labs_hybrid.pipeline import run_experiment


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one NVIDIA LABS hybrid experiment")
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--seeder", type=str, required=True, choices=["random", "qaoa", "dcqo", "dcqo_plus", "pce"])
    ap.add_argument("--shots", type=int, default=256)
    ap.add_argument("--K_out", type=int, default=256)
    ap.add_argument("--K_select", type=int, default=32)
    ap.add_argument("--device", type=str, default="auto", choices=["cpu", "gpu", "auto"])
    ap.add_argument("--backend", type=str, default="auto", choices=["cpu", "gpu", "auto"])
    ap.add_argument("--budget_s", type=float, default=2.0)
    ap.add_argument("--max_iters", type=int, default=None)
    ap.add_argument("--rng_seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--tag", type=str, default="single")
    ap.add_argument("--params", type=str, default="{}", help='JSON dict for seeder params')

    args = ap.parse_args()

    params = json.loads(args.params)
    seeder_cfg = SeederConfig(name=args.seeder, shots=args.shots, K_out=args.K_out, K_select=args.K_select, params=params)

    run_dir = new_run_dir(args.out, args.tag)
    cfg = RunConfig(
        N=args.N,
        seeder=seeder_cfg,
        backend=args.backend,
        device=args.device,
        budget_s=args.budget_s,
        max_iters=args.max_iters,
        rng_seed=args.rng_seed,
        out_dir=str(run_dir),
        run_tag=args.tag,
        save_traces=True,
    )
    write_json(Path(run_dir) / "run_config.json", cfg.to_dict())

    row = run_experiment(cfg)
    print(json.dumps(row, indent=2, sort_keys=True))
    print(f"\nWrote results to: {run_dir}")


if __name__ == "__main__":
    main()
