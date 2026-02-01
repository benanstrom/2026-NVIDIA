from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import GateConfig, RunConfig
from .logging_utils import ensure_dir, write_json, write_text
from .pipeline import run_experiment


def run_pytest(out_dir: str | Path) -> dict[str, Any]:
    """Run Gate 0: CPU-only correctness via pytest."""

    out_dir = ensure_dir(out_dir)
    log_path = out_dir / 'artifacts' / 'pytest_gate0.txt'
    log_path.parent.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-m", "pytest"], capture_output=True, text=True)
    dt = time.perf_counter() - t0

    log_path.write_text(proc.stdout + "\n\n" + proc.stderr)
    return {
        'gate': 'gate0_tests',
        'pytest_returncode': int(proc.returncode),
        'pytest_time_s': float(dt),
        'pytest_log': str(log_path),
        'pytest_passed': proc.returncode == 0,
    }


def run_gate(gate: GateConfig, root_out_dir: str | Path, run_tag: str, rng_seed: int) -> dict[str, Any]:
    """Run a gate (1..3) and write required handoff bundle files."""

    gate_dir = ensure_dir(Path(root_out_dir) / gate.name)
    (gate_dir / 'artifacts').mkdir(exist_ok=True)

    rows: list[dict] = []
    for N in gate.N_list:
        for seeder_cfg in gate.seeders:
            cfg = RunConfig(
                N=int(N),
                seeder=seeder_cfg,
                backend='auto',
                device=gate.device,
                budget_s=gate.budget_s,
                max_iters=gate.max_iters,
                rng_seed=int(rng_seed),
                out_dir=str(gate_dir),
                run_tag=run_tag,
                save_traces=True,
            )
            rows.append(run_experiment(cfg))

    run_cfg = {
        'gate': asdict(gate),
        'run_tag': run_tag,
        'rng_seed': int(rng_seed),
        'env': {'platform': platform.platform(), 'python': platform.python_version()},
    }
    write_json(gate_dir / 'run_config.json', run_cfg)
    write_text(gate_dir / 'results_summary.md', summarize_rows(rows, title=f'Results summary: {gate.name}'))
    write_text(gate_dir / 'PROMPT_BACK_TO_AI.md', prompt_back_to_ai(rows, run_cfg))

    return {'gate': gate.name, 'gate_dir': str(gate_dir), 'num_runs': len(rows)}


def summarize_rows(rows: list[dict], title: str = "Results summary") -> str:
    if not rows:
        return f"# {title}\n\n(no runs)\n"

    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(str(r.get('seeder')), []).append(r)

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("| seeder | best_final_energy (min) | median_final_energy | median_time_s |")
    lines.append("|---|---:|---:|---:|")
    for seeder, rs in sorted(by.items()):
        finals = sorted(float(x.get("best_final_energy", 1e30)) for x in rs)
        times = sorted(float(x.get("t_mts_total_s", 0.0)) for x in rs)
        best = finals[0]
        med = finals[len(finals)//2]
        med_t = times[len(times)//2]
        lines.append(f"| {seeder} | {best:.1f} | {med:.1f} | {med_t:.3f} |")

    lines.append("Artifacts: see `artifacts/` (seed energy arrays, best sequences, MTS traces).")
    return "\n".join(lines) + "\n"


def prompt_back_to_ai(rows: list[dict], run_cfg: dict[str, Any]) -> str:
    """Create the required PROMPT_BACK_TO_AI.md content."""

    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(str(r.get('seeder')), []).append(r)

    top_lines: list[str] = []
    for seeder, rs in sorted(by.items()):
        best = min(rs, key=lambda x: float(x.get("best_final_energy", 1e30)))
        top_lines.append(
            f"- {seeder}: best_final_energy={float(best.get('best_final_energy')):.1f}, "
            f"best_seed_energy={int(best.get('best_seed_energy'))}, "
            f"t_mts_total_s={float(best.get('t_mts_total_s')):.3f}"
        )

    parts = [
        "# PROMPT_BACK_TO_AI",
        "",
        "Paste this file (or the zipped handoff bundle) back into the code-generating AI chat.",
        "",
        "## 1) Verify correctness regressions FIRST",
        "- Check `pytest` (Gate 0) and CPU/GPU energy parity tests (Gate 2).",
        "- Fix correctness before optimizing performance.",
        "",
        "## 2) Diagnose bottlenecks using runtime breakdown fields",
        "- Use `metrics.csv` fields: t_seeder_compile_s, t_seeder_sampling_s, t_postselect_s, t_mts_total_s.",
        "- Identify dominant components and propose improvements that do NOT change the fixed MTS logic/budgets.",
        "",
        "## 3) Propose the SMALLEST code edits",
        "- Preserve the fixed-vs-variable rule: do not tune/modify MTS per seeder.",
        "- Seeders may change; selection stays energy-based.",
        "",
        "## Key metrics (best seen per seeder)",
        *top_lines,
        "",
        "## Run config",
        "```json",
        json.dumps(run_cfg, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(parts)
