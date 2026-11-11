"""
Ingests and standardizes iTIC Probe Data into Silver Layer:
1. Computes P90 Free-Flow Speed strictly from Train Split ONLY (Zero Leakage)
2. Performs bounded short-term Forward-Fill (max 2 slots = 30 mins) with `is_imputed` flag
3. Computes clipped congestion_ratio in [0.0, 1.2] and `is_outlier_flagged`
4. Saves `silver_road_segments.parquet` and `silver_traffic_15m.parquet`
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_engineering.quality_gate import validate_and_partition_traffic
from src.utils.logger import log_info, log_step, log_success
from src.utils.storage import ROOT_DIR, load_config, load_table, save_table


def run_ingest_itic() -> dict[str, int]:
    log_step("INGEST_ITIC", "ประมวลผลข้อมูลจราจร iTIC และสร้างกราฟโครงข่ายถนน (Bronze -> Silver)...")
    cfg = load_config()
    bronze_dir = ROOT_DIR / cfg["paths"]["bronze_dir"]
    silver_dir = ROOT_DIR / cfg["paths"]["silver_dir"]

    df_segments = load_table(bronze_dir / "bronze_itic_segments.parquet")
    df_traffic_raw = load_table(bronze_dir / "bronze_itic_traffic_15m.parquet")

    # 1. Run Quality Gate 4.2
    df_valid, _, q_report = validate_and_partition_traffic(df_traffic_raw, df_segments)

    # 2. Compute Free-Flow Speed (P90) strictly on TRAIN split only (Strict Leakage Control)
    train_end = pd.Timestamp(cfg["temporal_split"]["train_end"])
    df_train_sub = df_valid[df_valid["ts_15m"] <= train_end]
    p90_by_seg = (
        df_train_sub.groupby("seg_id")["speed_median_kmh"]
        .quantile(0.90)
        .rename("free_flow_speed_kmh")
        .round(2)
        .reset_index()
    )

    df_silver_segments = df_segments.drop(columns=["true_free_flow_kmh"], errors="ignore").merge(
        p90_by_seg, on="seg_id", how="left"
    )
    df_silver_segments["free_flow_speed_kmh"] = df_silver_segments["free_flow_speed_kmh"].fillna(50.0)

    # 3. Short-term bounded Forward-Fill (max 2 slots = 30 mins) per segment within each continuous period
    max_ffill = int(cfg["feature_engineering"]["max_forward_fill_slots"])
    all_observed_ts = pd.Index(sorted(df_valid["ts_15m"].unique()), name="ts_15m")
    full_grid = pd.MultiIndex.from_product(
        [df_silver_segments["seg_id"].unique(), all_observed_ts],
        names=["seg_id", "ts_15m"],
    ).to_frame().reset_index(drop=True)

    df_merged = full_grid.merge(df_valid, on=["seg_id", "ts_15m"], how="left")
    was_missing = df_merged["speed_median_kmh"].isna()

    df_merged["speed_median_kmh"] = df_merged.groupby("seg_id")["speed_median_kmh"].ffill(limit=max_ffill)
    df_merged["probe_count"] = df_merged["probe_count"].fillna(0).astype(int)
    df_merged["is_jump_outlier"] = df_merged["is_jump_outlier"].where(df_merged["is_jump_outlier"].notna(), False).astype(bool)
    df_merged["is_imputed"] = (was_missing & df_merged["speed_median_kmh"].notna()).astype(bool)

    # Drop any remaining rows that exceeded max forward-fill window
    df_merged = df_merged.dropna(subset=["speed_median_kmh"]).reset_index(drop=True)

    # 4. Compute congestion_ratio and clip to [0.0, 1.2]
    df_merged = df_merged.merge(
        df_silver_segments[["seg_id", "free_flow_speed_kmh"]],
        on="seg_id",
        how="left",
    )
    raw_ratio = df_merged["speed_median_kmh"] / df_merged["free_flow_speed_kmh"]
    df_merged["is_outlier_flagged"] = (df_merged["is_jump_outlier"] | (raw_ratio > 1.2)).astype(bool)
    df_merged["congestion_ratio"] = np.clip(raw_ratio, 0.0, 1.2).round(4)
    df_merged["year_month"] = df_merged["ts_15m"].dt.strftime("%Y-%m")

    silver_cols = [
        "seg_id",
        "ts_15m",
        "speed_median_kmh",
        "probe_count",
        "congestion_ratio",
        "is_outlier_flagged",
        "is_imputed",
        "year_month",
    ]
    df_silver_traffic = df_merged[silver_cols].sort_values(["seg_id", "ts_15m"]).reset_index(drop=True)

    save_table(df_silver_segments, silver_dir / "silver_road_segments.parquet", layer="silver")
    save_table(df_silver_traffic, silver_dir / "silver_traffic_15m.parquet", layer="silver")

    n_imputed = int(df_silver_traffic["is_imputed"].sum())
    log_info(
        f"คำนวณ P90 Free-flow speed เฉพาะจากชุด Train (<= {train_end.strftime('%Y-%m-%d')}) สำเร็จ | "
        f"Imputed (<=30m): {n_imputed:,} แถว"
    )
    log_success(f"บันทึก silver_road_segments ({len(df_silver_segments)} ถนน) และ silver_traffic_15m ({len(df_silver_traffic):,} แถว)")
    return {
        "silver_segments": len(df_silver_segments),
        "silver_traffic_rows": len(df_silver_traffic),
        "imputed_rows": n_imputed,
        "quarantined_rows": q_report["quarantined_rows"],
    }
