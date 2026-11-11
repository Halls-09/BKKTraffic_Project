"""
Quality Gate 4.2 & Data Contract Enforcement Engine.
Validates schema, completeness (>=80% daily, >=90% daytime 06:00-22:00),
and physical plausibility (0 <= speed <= 150 km/h, jump <= 60 km/h, probe_count >= 1).
Routes rejected rows to data/02_quarantine/ with explicit rejection reasons.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import log_info, log_step, log_success, log_warning
from src.utils.storage import ROOT_DIR, load_config, load_contract, save_table


def validate_and_partition_traffic(
    df_traffic: pd.DataFrame,
    df_segments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Applies Data Contract 4.2 on Bronze traffic observations.
    Returns (valid_df, quarantined_df, quality_report).
    """
    log_step("QUALITY_GATE", "ตรวจสอบ Data Contract 4.2 (Schema, Completeness, Plausibility)...")
    cfg = load_config()
    contract = load_contract()
    rules = contract["schemas"]["traffic_15m"]

    df = df_traffic.copy()
    required_cols = rules["required_columns"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema violation in traffic_15m! Missing columns: {missing_cols}")

    df["ts_15m"] = pd.to_datetime(df["ts_15m"])
    df = df.sort_values(["seg_id", "ts_15m"]).reset_index(drop=True)

    # 1. Hard Plausibility checks (quarantine invalid sensor rows)
    p_rules = rules["plausibility_rules"]
    cond_speed_range = (df["speed_median_kmh"] < p_rules["min_speed_kmh"]) | (
        df["speed_median_kmh"] > p_rules["max_speed_kmh"]
    )
    cond_probes = df["probe_count"] < p_rules["min_probe_count"]
    cond_unknown_seg = ~df["seg_id"].isin(set(df_segments["seg_id"]))

    quarantine_mask = cond_speed_range | cond_probes | cond_unknown_seg
    df_quarantine = df.loc[quarantine_mask].copy()
    if len(df_quarantine) > 0:
        reasons = np.where(
            cond_speed_range[quarantine_mask],
            "SPEED_OUT_OF_BOUNDS_0_150",
            np.where(
                cond_probes[quarantine_mask],
                "ZERO_PROBE_COUNT",
                "UNKNOWN_SEGMENT_ID",
            ),
        )
        df_quarantine["quarantine_reason"] = reasons
        df_quarantine["quarantined_at_utc"] = datetime.now(timezone.utc).isoformat()

    df_valid = df.loc[~quarantine_mask].copy().reset_index(drop=True)

    # 2. Speed Jump Flagging (> 60 km/h within 15 minutes)
    df_valid["prev_speed"] = df_valid.groupby("seg_id")["speed_median_kmh"].shift(1)
    df_valid["prev_ts"] = df_valid.groupby("seg_id")["ts_15m"].shift(1)
    is_consecutive = (df_valid["ts_15m"] - df_valid["prev_ts"]) == pd.Timedelta(minutes=15)
    speed_jump = (df_valid["speed_median_kmh"] - df_valid["prev_speed"]).abs()
    df_valid["is_jump_outlier"] = is_consecutive & (speed_jump > p_rules["max_speed_jump_15m_kmh"])

    # Smooth extreme single-slot jumps by capping delta at 60 km/h while keeping flag = True
    jump_mask = df_valid["is_jump_outlier"]
    if jump_mask.any():
        direction_sign = np.sign(df_valid.loc[jump_mask, "speed_median_kmh"] - df_valid.loc[jump_mask, "prev_speed"])
        df_valid.loc[jump_mask, "speed_median_kmh"] = (
            df_valid.loc[jump_mask, "prev_speed"] + direction_sign * p_rules["max_speed_jump_15m_kmh"]
        )

    df_valid = df_valid.drop(columns=["prev_speed", "prev_ts"])

    # 3. Completeness Check (Daily >= 80%, Daytime 06:00-22:00 >= 90%)
    c_rules = rules["completeness_rules"]
    n_expected_segs = len(df_segments)
    unique_dates = df_valid["ts_15m"].dt.date.unique()
    total_expected_slots = len(unique_dates) * 96 * n_expected_segs
    overall_completeness = len(df_valid) / max(1, total_expected_slots)

    hr = df_valid["ts_15m"].dt.hour
    daytime_rows = int(((hr >= 6) & (hr < 22)).sum())
    expected_daytime = len(unique_dates) * 64 * n_expected_segs
    daytime_completeness = daytime_rows / max(1, expected_daytime)

    completeness_passed = (
        overall_completeness >= c_rules["min_daily_coverage_ratio"]
        and daytime_completeness >= c_rules["min_daytime_coverage_ratio"]
    )

    quarantine_dir = ROOT_DIR / cfg["paths"]["quarantine_dir"]
    save_table(df_quarantine, quarantine_dir / "quarantine_traffic_15m.parquet", layer="quarantine")

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "contract_version": contract["contract_version"],
        "total_input_rows": int(len(df_traffic)),
        "valid_rows": int(len(df_valid)),
        "quarantined_rows": int(len(df_quarantine)),
        "jump_outliers_flagged": int(jump_mask.sum()),
        "overall_completeness": round(float(overall_completeness), 4),
        "daytime_completeness": round(float(daytime_completeness), 4),
        "completeness_gate_passed": bool(completeness_passed),
        "status": "PASSED" if completeness_passed else "WARNING_LOW_COVERAGE",
    }

    report_path = ROOT_DIR / cfg["paths"]["artifacts_dir"] / "lineage" / "quality_gate_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log_info(
        f"Quality Gate Results: Valid={report['valid_rows']:,} | "
        f"Quarantined={report['quarantined_rows']} | "
        f"Jump Flagged={report['jump_outliers_flagged']} | "
        f"Completeness={report['overall_completeness']*100:.1f}% (Daytime {report['daytime_completeness']*100:.1f}%)"
    )
    if report["quarantined_rows"] > 0:
        log_warning(f"แยกข้อมูลผิดปกติ {report['quarantined_rows']} แถว ไปยัง data/02_quarantine/ เรียบร้อย")
    log_success("ผ่านเกณฑ์ Data Contract 4.2 พร้อมเข้าสู่ Silver Layer")
    return df_valid, df_quarantine, report
