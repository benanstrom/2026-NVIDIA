#!/usr/bin/env python
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from labs_hybrid.logging_utils import now_timestamp


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            if p.is_dir():
                continue
            # store paths relative to src_dir
            zf.write(p, arcname=str(p.relative_to(src_dir)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Package a gate run folder into a timestamped handoff zip")
    ap.add_argument("--run_dir", type=str, required=True, help="Gate directory containing results_summary.md, run_config.json, metrics.csv, metrics.jsonl, PROMPT_BACK_TO_AI.md, artifacts/")
    args = ap.parse_args()

    src = Path(args.run_dir).resolve()
    required = [
        src / "results_summary.md",
        src / "run_config.json",
        src / "metrics.csv",
        src / "metrics.jsonl",
        src / "PROMPT_BACK_TO_AI.md",
        src / "artifacts",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("Missing required handoff bundle paths:\n" + "\n".join(missing))

    zip_name = f"handoff_{src.name}_{now_timestamp()}.zip"
    zip_path = src.parent / zip_name
    zip_dir(src, zip_path)
    print(f"Wrote: {zip_path}")


if __name__ == "__main__":
    main()
