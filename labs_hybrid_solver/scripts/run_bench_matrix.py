#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from labs_hybrid.config import default_benchmark_plan
from labs_hybrid.benchmarks import run_gate, run_pytest
from labs_hybrid.logging_utils import ensure_dir, new_run_dir, write_json, write_text


def write_gate0_bundle(root_dir: Path, pytest_row: dict) -> Path:
    g0 = ensure_dir(root_dir / "gate0_tests")
    (g0 / "artifacts").mkdir(exist_ok=True)

    # Copy pytest log into gate folder if it exists
    log_src = Path(pytest_row.get("pytest_log", ""))
    if log_src.exists():
        (g0 / "artifacts" / log_src.name).write_text(log_src.read_text())

    # Minimal metrics for Gate0
    metrics_csv = g0 / "metrics.csv"
    metrics_jsonl = g0 / "metrics.jsonl"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pytest_row.keys()))
        w.writeheader()
        w.writerow(pytest_row)
    metrics_jsonl.write_text(json.dumps(pytest_row) + "\n")

    write_json(g0 / "run_config.json", {"gate": "gate0_tests"})

    summary = "# Results summary: gate0_tests\n\n" + (
        "✅ pytest passed\n" if pytest_row.get("pytest_passed") else "❌ pytest failed\n"
    )
    summary += f"\nReturn code: {pytest_row.get('pytest_returncode')}\n"
    summary += f"Time (s): {pytest_row.get('pytest_time_s')}\n"
    write_text(g0 / "results_summary.md", summary)

    prompt = "\n".join(
        [
            "# PROMPT_BACK_TO_AI",
            "",
            "Gate 0 (pytest) was executed.",
            "If pytest failed, fix correctness regressions before running benchmark gates.",
            "Do NOT change MTS heuristics per seeder; only fix bugs / parity issues.",
            "",
            f"pytest_passed={pytest_row.get('pytest_passed')} returncode={pytest_row.get('pytest_returncode')}",
        ]
    )
    write_text(g0 / "PROMPT_BACK_TO_AI.md", prompt)
    return g0


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Gate 0→3 benchmark plan (CPU tests -> CPU small -> GPU bring-up -> GPU matrix)")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--tag", type=str, default="bench")
    ap.add_argument("--rng_seed", type=int, default=0)
    ap.add_argument("--skip_gate0", action="store_true")
    ap.add_argument("--skip_gate1", action="store_true")
    ap.add_argument("--skip_gate2", action="store_true")
    ap.add_argument("--skip_gate3", action="store_true")
    ap.add_argument("--no_zip", action="store_true", help="Do not package handoff zips")

    args = ap.parse_args()

    plan = default_benchmark_plan()
    run_root = new_run_dir(args.out, args.tag)
    run_root = Path(run_root)

    # Gate 0
    if not args.skip_gate0:
        pytest_row = run_pytest(run_root)
        gate0_dir = write_gate0_bundle(run_root, pytest_row)
        if not args.no_zip:
            subprocess.run([sys.executable, str(Path(__file__).parent / "package_handoff.py"), "--run_dir", str(gate0_dir)], check=False)

        if not pytest_row.get("pytest_passed"):
            print("Gate 0 pytest failed; refusing to proceed to later gates unless --skip_gate0 is used.")
            return

    # Gates 1..3
    gpu_elapsed = 0.0
    for gate in plan.gates:
        if gate.name.startswith("gate1") and args.skip_gate1:
            continue
        if gate.name.startswith("gate2") and args.skip_gate2:
            continue
        if gate.name.startswith("gate3") and args.skip_gate3:
            continue

        if gate.device == "gpu" and gpu_elapsed >= plan.gpu_budget_cap_s:
            print(f"GPU budget cap reached ({gpu_elapsed:.1f}s >= {plan.gpu_budget_cap_s:.1f}s); stopping.")
            break

        t0 = time.perf_counter()
        result = run_gate(gate, run_root, run_tag=args.tag, rng_seed=args.rng_seed)
        dt = time.perf_counter() - t0
        if gate.device == "gpu":
            gpu_elapsed += dt

        gate_dir = Path(result["gate_dir"])

        # Generate plots
        subprocess.run([sys.executable, str(Path(__file__).parent / "make_plots.py"), "--run_dir", str(gate_dir)], check=False)

        # Zip handoff bundle
        if not args.no_zip:
            subprocess.run([sys.executable, str(Path(__file__).parent / "package_handoff.py"), "--run_dir", str(gate_dir)], check=False)

    print(f"\nAll done. Root run directory: {run_root}")


if __name__ == "__main__":
    main()
