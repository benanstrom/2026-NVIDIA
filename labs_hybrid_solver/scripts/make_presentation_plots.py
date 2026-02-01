"""Generate presentation-quality plots from existing benchmark results.

Uses the NVIDIA tutorial notebook visual style (#76b900 green, bold titles,
clean grids). Reads metrics.jsonl from results/ — no GPU or re-run required.

Usage:
    python scripts/make_presentation_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# NVIDIA tutorial theme
# ---------------------------------------------------------------------------
NVIDIA_GREEN = "#76b900"
NVIDIA_DARK = "#2d2d2d"
COLORS = {
    "random": "lightgray",
    "qaoa": "dimgray",
    "dcqo": "#c4e680",
    "dcqo_plus": NVIDIA_GREEN,
    "pce": "#1a6b00",
}
SEEDER_ORDER = ["random", "qaoa", "dcqo", "dcqo_plus", "pce"]
SEEDER_LABELS = {
    "random": "Random",
    "qaoa": "QAOA",
    "dcqo": "DCQO",
    "dcqo_plus": "DCQO+",
    "pce": "PCE",
}


def _apply_theme():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cccccc",
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def _load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().strip().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _group_by_seeder(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(str(r["seeder"]), []).append(r)
    return out


# ---------------------------------------------------------------------------
# Plot 1: Seed quality by N (grouped bar)
# ---------------------------------------------------------------------------
def plot_seed_quality(rows: list[dict], out_dir: Path):
    by_n: dict[int, dict[str, int]] = {}
    for r in rows:
        n = int(r["N"])
        s = str(r["seeder"])
        by_n.setdefault(n, {})[s] = int(r["best_seed_energy"])

    ns = sorted(by_n.keys())
    seeders = [s for s in SEEDER_ORDER if any(s in by_n[n] for n in ns)]
    n_seeders = len(seeders)
    x = np.arange(len(ns))
    width = 0.8 / n_seeders

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, s in enumerate(seeders):
        vals = [by_n[n].get(s, 0) for n in ns]
        mask = [by_n[n].get(s) is not None for n in ns]
        positions = x[mask] + i * width - (n_seeders - 1) * width / 2
        bar_vals = [v for v, m in zip(vals, mask) if m]
        bars = ax.bar(positions, bar_vals, width * 0.9, label=SEEDER_LABELS[s],
                      color=COLORS[s], edgecolor="black", linewidth=0.8)
        for bar, val in zip(bars, bar_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                    str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in ns], fontsize=14)
    ax.set_ylabel("Best Seed Energy (lower = better)", fontsize=14)
    ax.set_title("Seed Quality Before MTS", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "seed_quality_by_N.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Final energy by N (grouped bar)
# ---------------------------------------------------------------------------
def plot_final_energy(rows: list[dict], out_dir: Path):
    by_n: dict[int, dict[str, float]] = {}
    for r in rows:
        n = int(r["N"])
        s = str(r["seeder"])
        by_n.setdefault(n, {})[s] = float(r["best_final_energy"])

    ns = sorted(by_n.keys())
    seeders = [s for s in SEEDER_ORDER if any(s in by_n[n] for n in ns)]
    n_seeders = len(seeders)
    x = np.arange(len(ns))
    width = 0.8 / n_seeders

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, s in enumerate(seeders):
        vals = [by_n[n].get(s, 0) for n in ns]
        mask = [by_n[n].get(s) is not None for n in ns]
        positions = x[mask] + i * width - (n_seeders - 1) * width / 2
        bar_vals = [v for v, m in zip(vals, mask) if m]
        bars = ax.bar(positions, bar_vals, width * 0.9, label=SEEDER_LABELS[s],
                      color=COLORS[s], edgecolor="black", linewidth=0.8)
        for bar, val in zip(bars, bar_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in ns], fontsize=14)
    ax.set_ylabel("Best Final Energy (lower = better)", fontsize=14)
    ax.set_title("Final Energy After MTS (3s GPU budget)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "final_energy_by_N.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Initial vs Final energy comparison (tutorial-style)
# ---------------------------------------------------------------------------
def plot_initial_vs_final(rows: list[dict], out_dir: Path):
    by_seeder = _group_by_seeder([r for r in rows if int(r["N"]) == 20])
    seeders = [s for s in SEEDER_ORDER if s in by_seeder]

    labels = []
    init_vals = []
    final_vals = []
    colors_init = []
    colors_final = []
    for s in seeders:
        r = by_seeder[s][0]
        labels.extend([f"{SEEDER_LABELS[s]}\n(seed)", f"{SEEDER_LABELS[s]}\n(final)"])
        init_vals.append(int(r["best_seed_energy"]))
        final_vals.append(float(r["best_final_energy"]))
        colors_init.append(COLORS[s])
        colors_final.append(COLORS[s])

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(seeders))
    w = 0.35
    bars_init = ax.bar(x_pos - w / 2, init_vals, w, label="Best Seed Energy",
                       color=[COLORS[s] for s in seeders], edgecolor="black",
                       linewidth=0.8, alpha=0.5)
    bars_final = ax.bar(x_pos + w / 2, final_vals, w, label="Best Final Energy",
                        color=[COLORS[s] for s in seeders], edgecolor="black",
                        linewidth=0.8)

    for bar, val in zip(bars_init, init_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontsize=14, fontweight="bold")
    for bar, val in zip(bars_final, final_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}", ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([SEEDER_LABELS[s] for s in seeders], fontsize=12)
    ax.set_ylabel("Energy", fontsize=14)
    ax.set_title("Initial vs Final Energy (N=20)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, loc="upper right")
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "initial_vs_final_N20.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Seeder time vs seed quality (scatter)
# ---------------------------------------------------------------------------
def plot_cost_benefit(rows: list[dict], out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in rows:
        s = str(r["seeder"])
        n = int(r["N"])
        t = float(r["t_seeder_total_s"])
        e = int(r["best_seed_energy"])
        color = COLORS.get(s, "gray")
        marker = {20: "o", 30: "s", 40: "D"}.get(n, "^")
        ax.scatter(t + 0.001, e, c=color, marker=marker, s=120,
                   edgecolors="black", linewidth=0.8, zorder=5)

    # Manual legend for seeders
    for s in SEEDER_ORDER:
        if any(str(r["seeder"]) == s for r in rows):
            ax.scatter([], [], c=COLORS[s], marker="o", s=80,
                       edgecolors="black", linewidth=0.8, label=SEEDER_LABELS[s])
    # Manual legend for N values
    for n, m in [(20, "o"), (30, "s"), (40, "D")]:
        ax.scatter([], [], c="white", marker=m, s=80,
                   edgecolors="black", linewidth=0.8, label=f"N={n}")

    ax.set_xscale("log")
    ax.set_xlabel("Seeder Time (s, log scale)", fontsize=14)
    ax.set_ylabel("Best Seed Energy (lower = better)", fontsize=14)
    ax.set_title("Cost vs Benefit: Seeder Time vs Seed Quality", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cost_vs_benefit.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: Runtime breakdown (stacked bar, tutorial-style)
# ---------------------------------------------------------------------------
def plot_runtime_breakdown(rows: list[dict], out_dir: Path):
    rows_n20 = [r for r in rows if int(r["N"]) == 20]
    by_seeder = _group_by_seeder(rows_n20)
    seeders = [s for s in SEEDER_ORDER if s in by_seeder]

    compile_t = []
    sample_t = []
    select_t = []
    mts_t = []
    for s in seeders:
        r = by_seeder[s][0]
        compile_t.append(float(r.get("t_seeder_compile_s", 0)))
        sample_t.append(float(r.get("t_seeder_sampling_s", 0)))
        select_t.append(float(r.get("t_postselect_s", 0)))
        mts_t.append(float(r.get("t_mts_total_s", 0)))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(seeders))
    comp = np.array(compile_t)
    samp = np.array(sample_t)
    sel = np.array(select_t)
    mts = np.array(mts_t)

    ax.bar(x, comp, color="#4a90d9", edgecolor="black", linewidth=0.8, label="Compile")
    ax.bar(x, samp, bottom=comp, color="#c4e680", edgecolor="black", linewidth=0.8, label="Sample")
    ax.bar(x, sel, bottom=comp + samp, color="#f5a623", edgecolor="black", linewidth=0.8, label="Post-select")
    ax.bar(x, mts, bottom=comp + samp + sel, color=NVIDIA_GREEN, edgecolor="black", linewidth=0.8, label="MTS")

    ax.set_xticks(x)
    ax.set_xticklabels([SEEDER_LABELS[s] for s in seeders], fontsize=12)
    ax.set_ylabel("Time (s)", fontsize=14)
    ax.set_title("Runtime Breakdown (N=20, GPU)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # Note: PCE seeder time is off-chart, annotate
    for i, s in enumerate(seeders):
        total = comp[i] + samp[i] + sel[i] + mts[i]
        if total > 10:
            ax.annotate(f"Total: {total:.1f}s",
                        xy=(i, min(total, ax.get_ylim()[1] * 0.95)),
                        ha="center", fontsize=10, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="gray", alpha=0.8))

    fig.tight_layout()
    fig.savefig(out_dir / "runtime_breakdown_N20.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 7: MTS iterations per second (grouped bar by N)
# ---------------------------------------------------------------------------
def plot_mts_iters_per_sec(rows: list[dict], out_dir: Path):
    by_n: dict[int, dict[str, float]] = {}
    for r in rows:
        n = int(r["N"])
        s = str(r["seeder"])
        iters = r.get("mts_summary", {}).get("iters", 0)
        t = float(r.get("t_mts_total_s", 1.0))
        if t > 0 and iters > 0:
            by_n.setdefault(n, {})[s] = iters / t

    ns = sorted(by_n.keys())
    seeders = [s for s in SEEDER_ORDER if any(s in by_n[n] for n in ns)]

    fig, ax = plt.subplots(figsize=(10, 6))
    for s in seeders:
        s_ns = [n for n in ns if s in by_n[n]]
        s_vals = [by_n[n][s] for n in s_ns]
        ax.plot(s_ns, s_vals, marker="o", markersize=8, linewidth=2.5,
                color=COLORS[s], label=SEEDER_LABELS[s],
                markeredgecolor="black", markeredgewidth=0.8)

    ax.set_xlabel("Sequence Length N", fontsize=14)
    ax.set_ylabel("MTS Iterations / Second", fontsize=14)
    ax.set_title("MTS Throughput (3s GPU Budget)", fontsize=15, fontweight="bold")
    ax.set_xticks(ns)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "mts_iters_per_sec.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 6: Summary results table (tutorial-style matplotlib table)
# ---------------------------------------------------------------------------
def plot_results_table(rows: list[dict], out_dir: Path):
    ns = sorted(set(int(r["N"]) for r in rows))
    seeders = [s for s in SEEDER_ORDER if any(str(r["seeder"]) == s for r in rows)]

    # Build table data
    headers = ["Seeder"] + [f"N={n}\n(seed)" for n in ns] + [f"N={n}\n(final)" for n in ns]
    table_data = []
    for s in seeders:
        row_data = [SEEDER_LABELS[s]]
        for n in ns:
            match = [r for r in rows if str(r["seeder"]) == s and int(r["N"]) == n]
            row_data.append(str(int(match[0]["best_seed_energy"])) if match else "-")
        for n in ns:
            match = [r for r in rows if str(r["seeder"]) == s and int(r["N"]) == n]
            row_data.append(f"{float(match[0]['best_final_energy']):.0f}" if match else "-")
        table_data.append(row_data)

    fig, ax = plt.subplots(figsize=(14, 3 + len(seeders) * 0.6))
    ax.axis("off")

    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor(NVIDIA_DARK)
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=13)

    # Style rows
    for i in range(1, len(table_data) + 1):
        bg = "#f5f5f5" if i % 2 == 0 else "white"
        for j in range(len(headers)):
            table[i, j].set_facecolor(bg)
            table[i, j].set_edgecolor("#cccccc")
        # Highlight PCE cells
        if table_data[i - 1][0] == "PCE":
            for j in range(len(headers)):
                cell = table[i, j]
                cell.set_text_props(color="#1a6b00", fontweight="bold")

    ax.set_title("Gate 3 Results Summary", fontsize=15, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(out_dir / "results_table.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _apply_theme()

    base = Path(__file__).resolve().parent.parent
    results_dir = base / "results" / "run_bench_20260201_052047"
    gate3_metrics = results_dir / "gate3_gpu_matrix" / "metrics.jsonl"
    gate2_metrics = results_dir / "gate2_gpu_bringup" / "metrics.jsonl"

    if not gate3_metrics.exists():
        print(f"Gate 3 metrics not found at {gate3_metrics}")
        return

    rows_g3 = _load_rows(gate3_metrics)
    rows_g2 = _load_rows(gate2_metrics) if gate2_metrics.exists() else []

    out_dir = base / "results" / "presentation_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating plots from {len(rows_g3)} Gate 3 rows...")
    plot_seed_quality(rows_g3, out_dir)
    print("  -> seed_quality_by_N.png")

    plot_final_energy(rows_g3, out_dir)
    print("  -> final_energy_by_N.png")

    plot_initial_vs_final(rows_g3, out_dir)
    print("  -> initial_vs_final_N20.png")

    plot_cost_benefit(rows_g3, out_dir)
    print("  -> cost_vs_benefit.png")

    plot_runtime_breakdown(rows_g3, out_dir)
    print("  -> runtime_breakdown_N20.png")

    plot_mts_iters_per_sec(rows_g3, out_dir)
    print("  -> mts_iters_per_sec.png")

    plot_results_table(rows_g3, out_dir)
    print("  -> results_table.png")

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
