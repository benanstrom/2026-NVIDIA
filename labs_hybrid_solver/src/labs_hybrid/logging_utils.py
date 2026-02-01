from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def now_timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def jsonable(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable objects."""

    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    return str(obj)


def write_jsonl_row(path: str | Path, row: dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(jsonable(row), sort_keys=True) + "\n")


def append_csv_row(path: str | Path, row: dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    file_exists = Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: jsonable(v) for k, v in row.items()})


def new_run_dir(base: str | Path, tag: str) -> Path:
    base = ensure_dir(base)
    run_dir = base / f"run_{tag}_{now_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    return run_dir


def write_text(path: str | Path, text: str) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_json(path: str | Path, data: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jsonable(data), f, indent=2, sort_keys=True)
