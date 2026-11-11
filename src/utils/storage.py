"""
Unified Medallion Lakehouse Storage & Query Engine.
Supports Apache Parquet (via PyArrow/DuckDB) with automatic fallback to
Pickle/SQLite3 when running in minimal environments without PyArrow wheels.
Also manages SHA-256 data checksums for reproducibility (Requirement 0).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_config(config_rel_path: str = "configs/pipeline_config.yaml") -> dict[str, Any]:
    path = ROOT_DIR / config_rel_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_contract(contract_rel_path: str = "configs/data_contract.yaml") -> dict[str, Any]:
    path = ROOT_DIR / contract_rel_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def update_checksum_manifest(file_path: Path, layer: str, row_count: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest_path = ROOT_DIR / "data" / "checksums_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {"files": {}}
    else:
        manifest = {"files": {}}

    rel_key = str(file_path.resolve().relative_to(ROOT_DIR)).replace("\\", "/")
    entry = {
        "sha256": compute_sha256(file_path),
        "layer": layer,
        "row_count": int(row_count),
        "size_bytes": file_path.stat().st_size,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        entry.update(extra)

    manifest["files"][rel_key] = entry
    manifest["last_updated_utc"] = datetime.now(timezone.utc).isoformat()

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return entry


def has_parquet_engine() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            return True
        except ImportError:
            return False


def save_table(df: pd.DataFrame, rel_or_abs_path: str | Path, layer: str = "silver") -> Path:
    """
    Saves a DataFrame to the Lakehouse.
    Uses native Parquet if pyarrow/fastparquet is installed; otherwise stores
    a high-speed pickle payload at the target path so all downstream paths
    (`silver_traffic_15m.parquet`, etc.) remain 100% consistent.
    """
    path = Path(rel_or_abs_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)

    engine_used = "parquet"
    if has_parquet_engine():
        df.to_parquet(path, index=False)
    else:
        engine_used = "pickle_compat"
        df.to_pickle(path)

    update_checksum_manifest(
        path,
        layer=layer,
        row_count=len(df),
        extra={"storage_format": engine_used, "columns": list(df.columns)},
    )
    return path


def load_table(rel_or_abs_path: str | Path) -> pd.DataFrame:
    """Loads a Lakehouse table transparently regardless of underlying binary engine."""
    path = Path(rel_or_abs_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Lakehouse table not found: {path}")

    if has_parquet_engine():
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_pickle(path)
    else:
        try:
            return pd.read_pickle(path)
        except Exception:
            return pd.read_csv(path)


def execute_sql_query(query: str, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Executes analytical SQL over registered DataFrames.
    Uses DuckDB if installed; falls back to in-memory SQLite3.
    """
    try:
        import duckdb
        con = duckdb.connect(database=":memory:")
        for name, df in tables.items():
            con.register(name, df)
        return con.execute(query).df()
    except ImportError:
        con = sqlite3.connect(":memory:")
        for name, df in tables.items():
            df_copy = df.copy()
            for col in df_copy.columns:
                if isinstance(df_copy[col].dtype, pd.DatetimeTZDtype):
                    df_copy[col] = df_copy[col].astype(str)
                elif df_copy[col].apply(lambda x: isinstance(x, list)).any():
                    df_copy[col] = df_copy[col].apply(json.dumps)
            df_copy.to_sql(name, con, index=False, if_exists="replace")
        return pd.read_sql_query(query, con)
