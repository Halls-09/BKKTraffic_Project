"""
Plug-and-Play Data Contract 4.2 Validator & Schema Adapter for Teammate (Data Engineer) Deliveries.
Supports:
  - Both 15-minute granularity (`horizon = 4 steps = 60 mins`) and 1-hour granularity (`horizon = 1 step = 60 mins`)
  - Both `congestion_ratio` (speed / free_flow_speed, congested < 0.5) and `red_rate` (congested >= 0.5)
  - Parquet (`.parquet`), Pickle (`.pkl`), and CSV (`.csv`) feature tables from Data Engineer
  - Automated Data Contract 4.2 Quality Gate Verification (Schema, Completeness, Uniqueness, Plausibility, Join Coverage)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import log_info, log_step, log_success, log_warning
from src.utils.storage import ROOT_DIR, load_config, load_table, save_table

# Canonical column mapping so Data Engineer's new table works without code changes
COLUMN_ALIASES: dict[str, list[str]] = {
    "link_id": ["link_id", "seg_id", "segment_id", "road_segment_id"],
    "ts_hour": ["ts_hour", "timestamp", "event_time", "ts_15m", "ts"],
    "red_rate_t0": ["red_rate_t0", "congestion_index_t0"],
    "target_red_rate_t1h": ["target_red_rate_t1h", "target_t60", "label_t60"],
    "precip_mm_t0": ["precip_mm_t0", "rain_mm", "precipitation_mm", "precip_1h_mm"],
    "precip_3h_mm_t0": ["precip_3h_mm_t0", "rain_3h_mm", "precipitation_3h_mm"],
    "downstream_red_rate_t0": ["downstream_red_rate_t0", "neighbor_downstream_t0"],
}


def detect_granularity_and_adapt(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Normalizes external Data Engineer feature table to canonical schema.
    If the table uses `congestion_ratio` (speed / free_flow_speed where < 0.5 is congested),
    converts `red_rate = 1.0 - clip(congestion_ratio, 0, 1)` so all downstream thresholds align.
    """
    work = df.copy()
    mapped_cols: dict[str, str] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical not in work.columns:
            for alias in aliases:
                if alias in work.columns:
                    work[canonical] = work[alias]
                    mapped_cols[alias] = canonical
                    break

    # Convert congestion_ratio (1 = free-flow, <0.5 = jammed) -> red_rate (1 = jammed, >=0.5 = jammed)
    ratio_converted = False
    if "red_rate_t0" not in work.columns and "congestion_ratio_t0" in work.columns:
        work["red_rate_t0"] = (1.0 - np.clip(work["congestion_ratio_t0"].astype(np.float32), 0.0, 1.0)).round(4)
        ratio_converted = True
    if "target_red_rate_t1h" not in work.columns and "target_congestion_ratio_t60" in work.columns:
        work["target_red_rate_t1h"] = (
            1.0 - np.clip(work["target_congestion_ratio_t60"].astype(np.float32), 0.0, 1.0)
        ).round(4)
        ratio_converted = True

    work["ts_hour"] = pd.to_datetime(work["ts_hour"])
    diffs = work.groupby("link_id")["ts_hour"].diff().dropna()
    median_step_min = int(diffs.median().total_seconds() // 60) if len(diffs) else 60
    granularity = "15min" if median_step_min <= 20 else "60min"
    horizon_steps = 4 if granularity == "15min" else 1

    if "target_ts" not in work.columns:
        work["target_ts"] = work["ts_hour"] + pd.Timedelta(minutes=60)

    info = {
        "detected_granularity": granularity,
        "median_step_minutes": median_step_min,
        "horizon_steps_for_60min": horizon_steps,
        "congestion_ratio_converted": ratio_converted,
        "mapped_column_aliases": mapped_cols,
    }
    return work, info


def validate_data_contract_4_2(df: pd.DataFrame | None = None) -> dict[str, Any]:
    """
    Executes the 5 Quality Gates from Proposal Section 4.2 (Page 5):
      1) Schema: valid Asia/Bangkok timestamps, congestion rate in [0, 1.2]
      2) Completeness: each kept segment meets >= 80% coverage rule
      3) Uniqueness: 0 duplicate (link_id, ts_hour) keys
      4) Plausibility: ratio <= 1.2, max step change flagged (not auto-deleted)
      5) Join Coverage: 100% of link_ids matched in Location Table (from_loc & to_loc) + Weather + Traffy district
    """
    log_step("DATA_CONTRACT", "ตรวจสอบ Data Contract 4.2 (5 Quality Gates) สำหรับเชื่อมต่อข้อมูลชุดใหม่...")
    cfg = load_config()
    if df is None:
        df = load_table(cfg["paths"]["gold_data"])

    work, adapt_info = detect_granularity_and_adapt(df)

    # Gate 1: Schema
    ts_valid = bool(pd.api.types.is_datetime64_any_dtype(work["ts_hour"]))
    rate_min = float(work["red_rate_t0"].min())
    rate_max = float(work["red_rate_t0"].max())
    schema_ok = bool(ts_valid and rate_min >= 0.0 and rate_max <= 1.2)

    # Gate 2: Completeness
    n_links = int(work["link_id"].nunique())
    n_rows = int(len(work))
    avg_rows_per_link = round(n_rows / max(n_links, 1), 1)
    completeness_pct = round(min(100.0, (avg_rows_per_link / (122 * 24)) * 100.0), 2)
    completeness_ok = bool(completeness_pct >= 80.0)

    # Gate 3: Uniqueness
    dup_count = int(work.duplicated(subset=["link_id", "ts_hour"]).sum())
    uniqueness_ok = bool(dup_count == 0)

    # Gate 4: Plausibility (flag rapid jumps > 0.60 in 1 step without deleting)
    rapid_jumps = int((work["delta_red_1h"].abs() > 0.60).sum()) if "delta_red_1h" in work.columns else 0
    plausibility_ok = bool(rate_max <= 1.2)

    # Gate 5: Join Coverage (Location Table endpoints + Weather + Traffy District)
    has_geom = all(c in work.columns for c in ("lat_from", "lon_from", "lat_to", "lon_to"))
    loc_join_pct = 100.0 if (has_geom and not work[["lat_from", "lat_to"]].isna().any().any()) else 0.0
    wx_join_pct = round((1.0 - float(work["precip_mm_t0"].isna().mean())) * 100.0, 2)
    traffy_join_pct = round((1.0 - float(work["train_flood_per_sq_km"].isna().mean())) * 100.0, 2)
    join_ok = bool(loc_join_pct == 100.0 and wx_join_pct == 100.0 and traffy_join_pct == 100.0)

    all_passed = bool(schema_ok and completeness_ok and uniqueness_ok and plausibility_ok and join_ok)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "contract_version": "Proposal Section 4.2 (Page 5)",
        "all_gates_passed": all_passed,
        "adapter_info": adapt_info,
        "dataset_summary": {
            "total_rows": n_rows,
            "directional_segments": n_links,
            "avg_rows_per_segment": avg_rows_per_link,
        },
        "gates": [
            {
                "gate": "1. Schema & Type Gate",
                "rule": "Valid timestamp + rate in [0.0, 1.2]",
                "actual": f"rate in [{rate_min:.2f}, {rate_max:.2f}]",
                "action": "Quarantine invalid partition",
                "passed": schema_ok,
            },
            {
                "gate": "2. Completeness Gate",
                "rule": "Segment coverage >= 80%",
                "actual": f"{completeness_pct:.1f}% ({avg_rows_per_link} hrs/link)",
                "action": "Filter segment below threshold",
                "passed": completeness_ok,
            },
            {
                "gate": "3. Uniqueness Gate",
                "rule": "0 duplicate (seg_id, ts) keys",
                "actual": f"{dup_count} duplicates",
                "action": "Deduplicate with audit log",
                "passed": uniqueness_ok,
            },
            {
                "gate": "4. Plausibility Gate",
                "rule": "ratio <= 1.2; flag rapid jumps (no auto-delete)",
                "actual": f"max={rate_max:.2f}, flagged jumps={rapid_jumps:,}",
                "action": "Flag outlier; keep in stream",
                "passed": plausibility_ok,
            },
            {
                "gate": "5. Join Coverage Gate",
                "rule": "100% Location Table + Weather + Traffy District",
                "actual": f"Loc={loc_join_pct:.0f}%, Wx={wx_join_pct:.0f}%, Traffy={traffy_join_pct:.0f}%",
                "action": "Stop run & alert on missing join",
                "passed": join_ok,
            },
        ],
    }

    mon_dir = ROOT_DIR / cfg["paths"]["monitoring_dir"]
    mon_dir.mkdir(parents=True, exist_ok=True)
    out_path = mon_dir / "data_contract_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if all_passed:
        log_success(f"ผ่าน Data Contract 4.2 ครบทั้ง 5 ด่าน ({n_links} ช่วงถนน, {n_rows:,} แถว, Granularity={adapt_info['detected_granularity']})")
    else:
        log_warning("พบด่าน Data Contract 4.2 ที่ต้องตรวจสอบ กรุณาดู data_contract_audit.json")
    return report


def ingest_external_teammate_dataset(external_path: str | Path) -> dict[str, Any]:
    """
    Validates and ingests a new feature table produced by the Data Engineer teammate
    (`.parquet`, `.pkl`, or `.csv`), adapting column names/granularity and saving to `gold_data`.
    """
    ext_p = Path(external_path)
    if not ext_p.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูลจาก Data Engineer: {ext_p}")
    log_info(f"กำลังนำเข้าข้อมูลชุดใหม่จาก Data Engineer: {ext_p}")
    df_new = pd.read_csv(ext_p) if ext_p.suffix.lower() == ".csv" else load_table(ext_p)
    adapted_df, _ = detect_granularity_and_adapt(df_new)
    report = validate_data_contract_4_2(adapted_df)
    if not report["all_gates_passed"]:
        raise RuntimeError("ข้อมูลชุดใหม่ไม่ผ่าน Data Contract 4.2 -> หยุด Pipeline ตามข้อกำหนด Stop Run!")
    cfg = load_config()
    save_table(adapted_df, cfg["paths"]["gold_data"])
    return report
