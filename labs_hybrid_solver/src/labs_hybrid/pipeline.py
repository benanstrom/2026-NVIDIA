from __future__ import annotations

import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import RunConfig
from .labs_energy_cpu import labs_energy_batch
from .logging_utils import append_csv_row, ensure_dir, write_json, write_jsonl_row
from .mts import run_mts
from .selectors import mean_pairwise_hamming_topK, select_topK_by_energy
from .seeders import get_seeder

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None  # type: ignore

from .labs_energy_gpu import labs_energy_batch_gpu


def _resolve_device(request: str) -> str:
    if request in ("cpu", "gpu"):
        return request
    # auto
    if cp is None:
        return "cpu"
    try:
        n = int(cp.cuda.runtime.getDeviceCount())
        return "gpu" if n > 0 else "cpu"
    except Exception:
        return "cpu"


def _energy_fns(device: str) -> tuple[Callable[[Any], Any], Any]:
    if device == "gpu" and cp is not None:
        return labs_energy_batch_gpu, cp
    return labs_energy_batch, np


def run_experiment(cfg: RunConfig) -> dict:
    """End-to-end experiment: seeder -> optional post-select -> MTS -> logging row -> artifacts.

    Returns a JSON-serializable dict suitable for CSV/JSONL.
    """

    out_dir = ensure_dir(cfg.out_dir)
    (out_dir / "artifacts").mkdir(exist_ok=True)

    device = _resolve_device(cfg.device)
    backend = cfg.backend if cfg.backend != "auto" else device

    seeder_fn = get_seeder(cfg.seeder.name)

    # 1) Seed generation
    t_seed0 = time.perf_counter()
    seeds, seeder_info = seeder_fn(
        int(cfg.N),
        int(cfg.seeder.shots),
        int(cfg.seeder.K_out),
        backend=backend,
        rng_seed=int(cfg.rng_seed),
        params=dict(cfg.seeder.params),
    )
    t_seed = time.perf_counter() - t_seed0

    if seeds.dtype != np.int8:
        seeds = seeds.astype(np.int8, copy=False)

    # 2) Optional post-selection
    energy_fn, xp = _energy_fns(device)
    sel_info = None
    t_sel = 0.0
    seeds_for_mts = seeds
    best_seed_energy = int(labs_energy_batch(seeds).min())

    if cfg.seeder.K_select is not None:
        K_select = int(cfg.seeder.K_select)
        if K_select < seeds.shape[0]:
            t_sel0 = time.perf_counter()
            if device == "gpu" and cp is not None:
                seeds_xp = cp.asarray(seeds, dtype=cp.int8)
                selected_xp, sel_info = select_topK_by_energy(seeds_xp, K_select, energy_fn, xp=cp)
                seeds_for_mts = cp.asnumpy(selected_xp).astype(np.int8, copy=False)
            else:
                selected_np, sel_info = select_topK_by_energy(seeds, K_select, energy_fn, xp=np)
                seeds_for_mts = selected_np.astype(np.int8, copy=False)
            t_sel = time.perf_counter() - t_sel0

    # Diversity on CPU among MTS starts
    diversity = mean_pairwise_hamming_topK(np.array(seeds_for_mts, copy=False))

    # 3) MTS
    t_mts0 = time.perf_counter()
    best_seq, best_energy, mts_logs = run_mts(
        seeds_for_mts,
        budget_s=cfg.budget_s,
        max_iters=cfg.max_iters,
        tabu_params=asdict(cfg.tabu),
        device=device,
        rng_seed=int(cfg.rng_seed),
    )
    t_mts = time.perf_counter() - t_mts0

    # 4) Persist minimal artifacts for plotting/debugging
    run_id = f"N{cfg.N}_{cfg.seeder.name}_seed{cfg.rng_seed}"
    if cfg.save_traces:
        write_json(out_dir / "artifacts" / f"{run_id}_mts_trace.json", mts_logs)

    # Save seed energy histogram data (CPU) for judge-friendly plots.
    seed_energies = labs_energy_batch(seeds).astype(np.int64)
    np.save(out_dir / "artifacts" / f"{run_id}_seed_energies.npy", seed_energies)

    # Save best sequence
    np.save(out_dir / "artifacts" / f"{run_id}_best_seq_pm1.npy", best_seq.astype(np.int8))

    # 5) Assemble one row
    row = {
        "timestamp": time.time(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "N": int(cfg.N),
        "seeder": cfg.seeder.name,
        "shots": int(cfg.seeder.shots),
        "K_out": int(cfg.seeder.K_out),
        "K_select": int(cfg.seeder.K_select) if cfg.seeder.K_select is not None else None,
        "backend": backend,
        "device": device,
        "budget_s": cfg.budget_s,
        "max_iters": cfg.max_iters,
        "rng_seed": int(cfg.rng_seed),
        "t_seeder_total_s": float(t_seed),
        "t_seeder_compile_s": float(seeder_info.get("compile_s", 0.0)),
        "t_seeder_sampling_s": float(seeder_info.get("sampling_s", 0.0)),
        "t_postselect_s": float(t_sel),
        "t_mts_total_s": float(t_mts),
        "best_seed_energy": int(best_seed_energy),
        "best_final_energy": float(best_energy),
        "diversity_mean_hamming": float(diversity),
        "seeder_info": seeder_info,
        "postselect_info": sel_info,
        "mts_summary": {
            "iters": int(mts_logs.get("iters", 0)),
            "best_energy": float(mts_logs.get("best_energy", best_energy)),
            "trace_len": int(len(mts_logs.get("best_trace", []))),
        },
        "artifact_prefix": run_id,
    }

    # 6) Append to metrics logs
    append_csv_row(out_dir / "metrics.csv", row)
    write_jsonl_row(out_dir / "metrics.jsonl", row)

    return row
