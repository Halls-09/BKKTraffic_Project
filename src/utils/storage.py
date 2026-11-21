"""Minimal storage, checksum & config helpers for the Bangkok Traffic MLOps workspace."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config(config_rel_path: str = "configs/pipeline_config.yaml") -> dict[str, Any]:
    path = ROOT_DIR / config_rel_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_sha256(file_path: str | Path) -> str:
    path = Path(file_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def has_parquet_engine() -> bool:
    for engine in ("pyarrow", "fastparquet"):
        try:
            __import__(engine)
            return True
        except ImportError:
            continue
    return False


def load_table(rel_or_abs_path: str | Path) -> pd.DataFrame:
    path = Path(rel_or_abs_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Data table not found: {path}")

    if has_parquet_engine():
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_pickle(path)
    try:
        return pd.read_pickle(path)
    except Exception:
        return pd.read_csv(path)


def save_table(df: pd.DataFrame, rel_or_abs_path: str | Path) -> Path:
    path = Path(rel_or_abs_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if has_parquet_engine():
        df.to_parquet(path, index=False)
    else:
        df.to_pickle(path)
    return path
