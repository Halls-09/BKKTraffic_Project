"""Shared utilities for the BKK Traffic pipeline.

- load config (config.yaml)
- path helpers
- raw-file registration: sha256 checksum + manifest + read-only (raw immutable)
- run id / logging helpers
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import random
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Ensure pandas .to_parquet / read_parquet works even if pyarrow is not installed
_ORIG_TO_PARQUET = pd.DataFrame.to_parquet
_ORIG_READ_PARQUET = pd.read_parquet


def _safe_to_parquet(self: pd.DataFrame, path_or_buf, *args, **kwargs):
    p = Path(path_or_buf)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        return _ORIG_TO_PARQUET(self, path_or_buf, *args, **kwargs)
    except ImportError:
        return self.to_pickle(p)


def _safe_read_parquet(path_or_buf, *args, **kwargs) -> pd.DataFrame:
    try:
        return _ORIG_READ_PARQUET(path_or_buf, *args, **kwargs)
    except ImportError:
        return pd.read_pickle(path_or_buf)


pd.DataFrame.to_parquet = _safe_to_parquet  # type: ignore[assignment]
pd.read_parquet = _safe_read_parquet  # type: ignore[assignment]

with open(ROOT / "config.yaml", encoding="utf-8") as _f:
    CFG: dict = yaml.safe_load(_f)

TZ: str = CFG["tz"]


def path(key: str) -> Path:
    """Return an absolute path from config `paths` section, e.g. path('raw')."""
    return ROOT / CFG["paths"][key]


def set_seed(seed: int | None = None) -> None:
    """Fix random seeds for reproducibility (Proposal: fixed seed)."""
    seed = CFG["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def new_run_id() -> str:
    """Timestamp-based id used to name quality/audit reports of one run."""
    return dt.datetime.now().strftime("%Y%m%dT%H%M%S")


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def sha256(file: Path) -> str:
    """SHA-256 of a file, streamed in 1 MB chunks (works for large files)."""
    h = hashlib.sha256()
    with open(file, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def register_raw(file: Path, source: str, make_readonly: bool = True) -> dict:
    """Record a raw file in the manifest (data snapshot checksum) and lock it.

    Raw data is immutable: once registered, it is set read-only so nothing in
    the pipeline can modify it by accident.
    """
    file = Path(file).resolve()
    rec = {
        "path": file.relative_to(ROOT).as_posix(),
        "source": source,
        "sha256": sha256(file),
        "bytes": file.stat().st_size,
        "registered_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    manifest = path("manifest")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if make_readonly:
        os.chmod(file, stat.S_IREAD)
    return rec


def verify_manifest() -> list[str]:
    """Re-hash every file in the manifest; return list of problems (empty = OK)."""
    problems: list[str] = []
    manifest = path("manifest")
    if not manifest.exists():
        return problems
    latest: dict[str, dict] = {}
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                latest[rec["path"]] = rec
    for p, rec in latest.items():
        fp = ROOT / p
        if not fp.exists():
            problems.append(f"missing: {p}")
        elif sha256(fp) != rec["sha256"]:
            problems.append(f"checksum changed: {p}")
    return problems


def write_json(obj: dict | list, file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

