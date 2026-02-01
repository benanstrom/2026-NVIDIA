from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .logging_utils import ensure_dir


def _group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(str(r.get(key)), []).append(r)
    return out


def plot_time_to_best(rows: list[dict], artifacts_dir: str | Path, out_dir: str | Path) -> Path:
    """Plot a proxy time-to-best curve by seeder.

    We use MTS best_trace length and time_total_s; times are linearly interpolated.
    """

    artifacts_dir = Path(artifacts_dir)
    out_dir = ensure_dir(out_dir)
    p = out_dir / "time_to_best.png"

    groups = _group_by(rows, "seeder")

    plt.figure()
    for seeder, rs in groups.items():
        # Plot the best run (lowest final energy)
        best = min(rs, key=lambda r: float(r.get("best_final_energy", 1e30)))
        prefix = best.get("artifact_prefix")
        if not prefix:
            continue
        trace_path = artifacts_dir / f"{prefix}_mts_trace.json"

        try:
            import json

            d = json.loads(trace_path.read_text())
            trace = d.get("best_trace", [])
            t_total = float(d.get("time_total_s", best.get("t_mts_total_s", 0.0)))
        except Exception:
            trace = [float(best.get("best_final_energy", 0.0))]
            t_total = float(best.get("t_mts_total_s", 0.0))

        if len(trace) <= 1:
            xs = [0.0]
            ys = [float(trace[0])]
        else:
            xs = np.linspace(0.0, max(1e-9, t_total), num=len(trace))
            ys = [float(x) for x in trace]
        plt.plot(xs, ys, label=seeder)

    plt.xlabel("time (s)")
    plt.ylabel("best energy so far")
    plt.title("Time-to-best (proxy from MTS trace)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def plot_final_energy_bar(rows: list[dict], out_dir: str | Path) -> Path:
    out_dir = ensure_dir(out_dir)
    p = out_dir / "final_energy_bar.png"

    groups = _group_by(rows, "seeder")
    labels = []
    vals = []
    for seeder, rs in groups.items():
        labels.append(seeder)
        vals.append(np.median([float(r.get("best_final_energy", np.nan)) for r in rs]))

    plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, vals)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("final best energy (median)")
    plt.title("Final energy by seeder")
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def plot_seed_energy_hist(rows: list[dict], artifacts_dir: str | Path, out_dir: str | Path) -> Path:
    """Histogram of seed energies, using per-run saved arrays."""

    artifacts_dir = Path(artifacts_dir)
    out_dir = ensure_dir(out_dir)
    p = out_dir / "seed_energy_hist.png"

    groups = _group_by(rows, "seeder")

    plt.figure()
    for seeder, rs in groups.items():
        all_E = []
        for r in rs:
            prefix = r.get("artifact_prefix")
            if not prefix:
                continue
            f = artifacts_dir / f"{prefix}_seed_energies.npy"
            if f.exists():
                all_E.append(np.load(f))
        if not all_E:
            continue
        data = np.concatenate(all_E)
        plt.hist(data, bins=30, alpha=0.5, label=seeder)

    plt.xlabel("seed energy")
    plt.ylabel("count")
    plt.title("Seed energy histogram (tail matters)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def plot_runtime_breakdown(rows: list[dict], out_dir: str | Path) -> Path:
    out_dir = ensure_dir(out_dir)
    p = out_dir / "runtime_breakdown.png"

    groups = _group_by(rows, "seeder")
    labels = []
    comp = []
    samp = []
    sel = []
    mts = []
    for seeder, rs in groups.items():
        labels.append(seeder)
        comp.append(np.median([float(r.get("t_seeder_compile_s", 0.0)) for r in rs]))
        samp.append(np.median([float(r.get("t_seeder_sampling_s", 0.0)) for r in rs]))
        sel.append(np.median([float(r.get("t_postselect_s", 0.0)) for r in rs]))
        mts.append(np.median([float(r.get("t_mts_total_s", 0.0)) for r in rs]))

    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x, comp, label="compile")
    plt.bar(x, samp, bottom=comp, label="sample")
    plt.bar(x, sel, bottom=np.array(comp) + np.array(samp), label="post-select")
    plt.bar(x, mts, bottom=np.array(comp) + np.array(samp) + np.array(sel), label="MTS")

    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("seconds (median)")
    plt.title("Runtime breakdown by seeder")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()
    return p
