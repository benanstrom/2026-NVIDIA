#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from labs_hybrid.logging_utils import ensure_dir
from labs_hybrid.plots import plot_final_energy_bar, plot_runtime_breakdown, plot_seed_energy_hist, plot_time_to_best


def load_rows_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate judge-friendly plots for a run folder")
    ap.add_argument("--run_dir", type=str, required=True, help="Path to a results run directory (contains metrics.jsonl)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    rows = load_rows_jsonl(run_dir / "metrics.jsonl")
    if not rows:
        raise SystemExit(f"No metrics.jsonl found in {run_dir}")

    artifacts_dir = run_dir / "artifacts"
    fig_dir = ensure_dir(artifacts_dir / "plots")

    plot_time_to_best(rows, artifacts_dir, fig_dir)
    plot_final_energy_bar(rows, fig_dir)
    plot_seed_energy_hist(rows, artifacts_dir, fig_dir)
    plot_runtime_breakdown(rows, fig_dir)

    print(f"Wrote plots to: {fig_dir}")


if __name__ == "__main__":
    main()
