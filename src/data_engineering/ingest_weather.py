"""
Ingests Open-Meteo Historical Weather observations and Incidents into Silver Layer.
Enforces plausibility bounds and `rain_mode = "observed_lag_only"` contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import log_info, log_step, log_success
from src.utils.storage import ROOT_DIR, load_config, load_table, save_table


def run_ingest_weather_and_incidents() -> dict[str, int]:
    log_step("INGEST_WEATHER", "ประมวลผลข้อมูลฝน Open-Meteo และข้อมูลอุบัติเหตุ (Bronze -> Silver)...")
    cfg = load_config()
    bronze_dir = ROOT_DIR / cfg["paths"]["bronze_dir"]
    silver_dir = ROOT_DIR / cfg["paths"]["silver_dir"]

    df_weather = load_table(bronze_dir / "bronze_weather_15m.parquet").copy()
    df_weather["ts_15m"] = pd.to_datetime(df_weather["ts_15m"])
    df_weather["rain_mm"] = np.clip(df_weather["rain_mm"], 0.0, 100.0).round(2)
    df_weather["temp_c"] = np.clip(df_weather["temp_c"], 15.0, 45.0).round(2)
    df_weather = df_weather.sort_values(["weather_zone_id", "ts_15m"]).reset_index(drop=True)

    # Compute rolling 3h (12 slots) and 6h (24 slots) cumulative rainfall using ONLY past observations (<= t)
    df_weather["rain_cum_3h"] = (
        df_weather.groupby("weather_zone_id")["rain_mm"]
        .transform(lambda s: s.rolling(12, min_periods=1).sum())
        .round(2)
    )
    df_weather["rain_cum_6h"] = (
        df_weather.groupby("weather_zone_id")["rain_mm"]
        .transform(lambda s: s.rolling(24, min_periods=1).sum())
        .round(2)
    )

    save_table(df_weather, silver_dir / "silver_weather_15m.parquet", layer="silver")

    df_incidents = load_table(bronze_dir / "bronze_incidents.parquet").copy()
    df_incidents["event_time"] = pd.to_datetime(df_incidents["event_time"])
    df_incidents["reported_time"] = pd.to_datetime(df_incidents["reported_time"])
    df_incidents["cleared_time"] = pd.to_datetime(df_incidents["cleared_time"])
    # Enforce strict causality: reported_time >= event_time
    df_incidents = df_incidents[df_incidents["reported_time"] >= df_incidents["event_time"]].reset_index(drop=True)
    save_table(df_incidents, silver_dir / "silver_incidents.parquet", layer="silver")

    log_info(f"Weather rows: {len(df_weather):,} | Validated incidents: {len(df_incidents):,}")
    log_success("บันทึก silver_weather_15m.parquet และ silver_incidents.parquet เรียบร้อย")
    return {"silver_weather_rows": len(df_weather), "silver_incidents_rows": len(df_incidents)}
