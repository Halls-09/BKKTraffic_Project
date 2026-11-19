"""Helper script to inspect any .parquet, .csv, or .json file in the project.

Usage examples:
    python peek.py data/processed/weather_hourly.parquet
    python peek.py data/processed/calendar_daily.parquet --rows 20
    python peek.py data/raw/calendar/calendar_2022.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_VENV_PY = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
if _VENV_PY.exists() and Path(sys.executable).resolve() != _VENV_PY.resolve():
    raise SystemExit(subprocess.call([str(_VENV_PY), *sys.argv]))

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 40)


def inspect_file(filepath: str, n: int = 10, export_csv: bool = False) -> None:
    p = Path(filepath)
    if not p.exists():
        print(f"File not found: {p}")
        return

    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    elif p.suffix == ".json":
        df = pd.read_json(p)
    else:
        print(f"Unsupported extension: {p.suffix}")
        return

    print("=" * 80)
    print(f"FILE : {p.as_posix()}  ({p.stat().st_size:,} bytes)")
    print(f"SHAPE: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("=" * 80)
    print("\n--- COLUMNS & DATA TYPES ---")
    dtype_df = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "nulls": df.isna().sum(),
        "sample": df.iloc[0].values if len(df) > 0 else None,
    })
    print(dtype_df)

    print(f"\n--- FIRST {n} ROWS ---")
    print(df.head(n))

    if export_csv:
        out_csv = p.with_suffix(".preview.csv")
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n-> Exported readable CSV to: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="Path to .parquet / .csv / .json file")
    ap.add_argument("--rows", type=int, default=10, help="Number of rows to preview")
    ap.add_argument("--csv", action="store_true", help="Also export a .preview.csv file for Excel")
    args = ap.parse_args()
    inspect_file(args.file, args.rows, args.csv)

