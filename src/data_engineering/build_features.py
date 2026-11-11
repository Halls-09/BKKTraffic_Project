"""
Gold Layer Feature Engineering Engine (8 Feature Groups + Zero-Leakage Cutoff).
Grain: (seg_id, ts_t)
Enforces:
- Strict As-Of Cutoff (available_time <= t)
- Historical Profile (Group 2) fit strictly on Train split
- Spatial Graph Neighbors (Group 6) computed at time t for downstream spillback (H3)
- Rain x Traffy Flood Vulnerability interaction (Group 8) for H4
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from src.utils.logger import log_info, log_step, log_success
from src.utils.storage import ROOT_DIR, load_config, load_table, save_table


FEATURE_GROUPS: dict[str, list[str]] = {
    "group_1_current_lags": [
        "ratio_t0",
        "ratio_lag_15m",
        "ratio_lag_30m",
        "ratio_lag_45m",
        "ratio_lag_60m",
        "delta_ratio_15m",
        "delta_ratio_30m",
        "rolling_std_60m",
    ],
    "group_2_historical_profile": [
        "hist_avg_ratio_target_slot",
        "hist_p25_ratio_target_slot",
        "current_diff_from_hist",
    ],
    "group_3_temporal_cyclical": [
        "target_hour_sin",
        "target_hour_cos",
        "target_slot_idx",
        "is_morning_rush",
        "is_evening_rush",
    ],
    "group_4_calendar": [
        "day_of_week",
        "is_weekend",
        "is_public_holiday",
        "is_friday_evening",
        "is_payday_window",
        "is_school_term",
    ],
    "group_5_weather": [
        "rain_mm_t0",
        "rain_cum_3h",
        "rain_cum_6h",
        "is_heavy_rain_t0",
        "temp_c_t0",
    ],
    "group_6_spatial_graph": [
        "downstream_min_ratio_t0",
        "downstream_mean_ratio_t0",
        "downstream_delta_15m",
        "upstream_mean_ratio_t0",
        "spatial_gradient_t0",
    ],
    "group_7_incidents": [
        "active_incident_on_seg",
        "incident_count_within_2km_1h",
        "mins_since_last_incident",
    ],
    "group_8_road_traffy_static": [
        "free_flow_speed_kmh",
        "length_km",
        "direction_code",
        "traffy_flood_density_district",
        "traffy_flood_buffer500m",
        "rain_x_traffy_flood_interaction",
    ],
}

ALL_FEATURE_COLS: list[str] = [col for grp in FEATURE_GROUPS.values() for col in grp]

THAI_HOLIDAYS_2022 = {
    "2022-06-03",
    "2022-07-13",
    "2022-07-14",
    "2022-07-15",
    "2022-07-28",
    "2022-07-29",
    "2022-08-12",
}


def _parse_list_col(val: object) -> list[str]:
    if isinstance(val, list):
        return val
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return []
    return []


def run_build_features() -> dict[str, int]:
    log_step("BUILD_FEATURES", "สร้าง Gold Feature Store ทั้ง 8 กลุ่ม (Zero-Leakage As-Of Join ณ เวลา t)...")
    cfg = load_config()
    silver_dir = ROOT_DIR / cfg["paths"]["silver_dir"]
    gold_dir = ROOT_DIR / cfg["paths"]["gold_dir"]

    df_segs = load_table(silver_dir / "silver_road_segments.parquet")
    df_traffic = load_table(silver_dir / "silver_traffic_15m.parquet")
    df_weather = load_table(silver_dir / "silver_weather_15m.parquet")
    df_traffy = load_table(silver_dir / "silver_traffy_spatial_features.parquet")
    df_incidents = load_table(silver_dir / "silver_incidents.parquet")

    df_traffic["ts_15m"] = pd.to_datetime(df_traffic["ts_15m"])
    df_weather["ts_15m"] = pd.to_datetime(df_weather["ts_15m"])
    df_traffic = df_traffic.sort_values(["seg_id", "ts_15m"]).reset_index(drop=True)

    df = df_traffic.rename(columns={"ts_15m": "ts_t", "congestion_ratio": "ratio_t0"}).copy()
    df["target_ts_t60"] = df["ts_t"] + pd.Timedelta(minutes=60)

    # ------------------------------------------------------------------
    # Group 1: Current & Lag Features (strictly <= t) & Targets (t + 60m)
    # ------------------------------------------------------------------
    g = df.groupby("seg_id", sort=False)
    df["ratio_lag_15m"] = g["ratio_t0"].shift(1)
    df["ratio_lag_30m"] = g["ratio_t0"].shift(2)
    df["ratio_lag_45m"] = g["ratio_t0"].shift(3)
    df["ratio_lag_60m"] = g["ratio_t0"].shift(4)

    df["delta_ratio_15m"] = (df["ratio_t0"] - df["ratio_lag_15m"]).round(4)
    df["delta_ratio_30m"] = (df["ratio_t0"] - df["ratio_lag_30m"]).round(4)
    df["rolling_std_60m"] = (
        g["ratio_t0"]
        .transform(lambda s: s.rolling(4, min_periods=2).std())
        .fillna(0.0)
        .round(4)
    )

    # Target at exactly t + 60m (4 slots ahead, verifying timestamp continuity)
    lead_ratio_4 = g["ratio_t0"].shift(-4)
    lead_ts_4 = g["ts_t"].shift(-4)
    exact_60m_mask = (lead_ts_4 - df["ts_t"]) == pd.Timedelta(minutes=60)
    df["target_ratio_t60"] = np.where(exact_60m_mask, lead_ratio_4, np.nan)

    # ------------------------------------------------------------------
    # Assign Temporal Split (Train / Val / Test)
    # ------------------------------------------------------------------
    train_end = pd.Timestamp(cfg["temporal_split"]["train_end"])
    val_start = pd.Timestamp(cfg["temporal_split"]["val_start"])
    val_end = pd.Timestamp(cfg["temporal_split"]["val_end"])
    test_start = pd.Timestamp(cfg["temporal_split"]["test_start"])

    # Note: A training row's target_ts_t60 must also be <= train_end so no target leaks across boundary!
    df["split"] = "ignore"
    df.loc[df["target_ts_t60"] <= train_end, "split"] = "train"
    df.loc[(df["ts_t"] >= val_start) & (df["target_ts_t60"] <= val_end), "split"] = "val"
    df.loc[df["ts_t"] >= test_start, "split"] = "test"

    # ------------------------------------------------------------------
    # Group 3 & Group 4: Temporal Cyclical & Calendar Features
    # ------------------------------------------------------------------
    current_slot = df["ts_t"].dt.hour * 4 + (df["ts_t"].dt.minute // 15)
    target_slot = df["target_ts_t60"].dt.hour * 4 + (df["target_ts_t60"].dt.minute // 15)
    df["current_slot_idx"] = current_slot
    df["target_slot_idx"] = target_slot
    df["target_hour_sin"] = np.sin(2.0 * np.pi * target_slot / 96.0).round(4)
    df["target_hour_cos"] = np.cos(2.0 * np.pi * target_slot / 96.0).round(4)

    target_hr_float = df["target_ts_t60"].dt.hour + df["target_ts_t60"].dt.minute / 60.0
    df["is_morning_rush"] = ((target_hr_float >= 6.5) & (target_hr_float <= 9.5)).astype(int)
    df["is_evening_rush"] = ((target_hr_float >= 16.0) & (target_hr_float <= 20.0)).astype(int)

    df["day_of_week"] = df["ts_t"].dt.dayofweek.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    date_str = df["ts_t"].dt.strftime("%Y-%m-%d")
    df["is_public_holiday"] = date_str.isin(THAI_HOLIDAYS_2022).astype(int)
    df["is_day_off"] = ((df["is_weekend"] == 1) | (df["is_public_holiday"] == 1)).astype(int)

    cur_hr_float = df["ts_t"].dt.hour + df["ts_t"].dt.minute / 60.0
    df["is_friday_evening"] = ((df["day_of_week"] == 4) & (cur_hr_float >= 16.0) & (cur_hr_float <= 21.0)).astype(int)
    day_of_month = df["ts_t"].dt.day
    df["is_payday_window"] = ((day_of_month >= 25) | (day_of_month <= 2)).astype(int)
    df["is_school_term"] = 1  # June-Sept is Bangkok school term

    # ------------------------------------------------------------------
    # Group 2: Historical Profile (Fit strictly on Train split!)
    # ------------------------------------------------------------------
    train_subset = df[df["split"] == "train"]
    hist_profile = (
        train_subset.groupby(["seg_id", "current_slot_idx", "is_day_off"])["ratio_t0"]
        .agg(
            hist_avg_slot="mean",
            hist_p25_slot=lambda s: np.percentile(s.dropna(), 25) if len(s.dropna()) > 0 else 0.5,
        )
        .reset_index()
    )

    # Merge for current slot t0 (to compute current_diff_from_hist)
    df = df.merge(
        hist_profile.rename(columns={"hist_avg_slot": "hist_avg_t0", "hist_p25_slot": "hist_p25_t0"}),
        on=["seg_id", "current_slot_idx", "is_day_off"],
        how="left",
    )
    # Merge for target slot t+60 (to provide historical expectation at target time)
    df = df.merge(
        hist_profile.rename(
            columns={
                "current_slot_idx": "target_slot_idx",
                "hist_avg_slot": "hist_avg_ratio_target_slot",
                "hist_p25_slot": "hist_p25_ratio_target_slot",
            }
        ),
        on=["seg_id", "target_slot_idx", "is_day_off"],
        how="left",
    )
    df["hist_avg_ratio_target_slot"] = df["hist_avg_ratio_target_slot"].fillna(0.65).round(4)
    df["hist_p25_ratio_target_slot"] = df["hist_p25_ratio_target_slot"].fillna(0.50).round(4)
    df["current_diff_from_hist"] = (df["ratio_t0"] - df["hist_avg_t0"].fillna(0.65)).round(4)

    # ------------------------------------------------------------------
    # Join Segment Metadata & Group 5: Weather Features (As-Of t0)
    # ------------------------------------------------------------------
    df = df.merge(
        df_segs[["seg_id", "road_name", "direction", "length_km", "district_name", "weather_zone_id", "free_flow_speed_kmh"]],
        on="seg_id",
        how="left",
    )
    df = df.merge(
        df_weather.rename(
            columns={
                "ts_15m": "ts_t",
                "rain_mm": "rain_mm_t0",
                "temp_c": "temp_c_t0",
            }
        ),
        on=["weather_zone_id", "ts_t"],
        how="left",
    )
    df["rain_mm_t0"] = df["rain_mm_t0"].fillna(0.0)
    df["rain_cum_3h"] = df["rain_cum_3h"].fillna(0.0)
    df["rain_cum_6h"] = df["rain_cum_6h"].fillna(0.0)
    df["is_heavy_rain_t0"] = (df["rain_mm_t0"] >= 3.0).astype(int)
    df["temp_c_t0"] = df["temp_c_t0"].fillna(30.0)

    # ------------------------------------------------------------------
    # Group 6: Spatial Graph Neighbors (Downstream Spillback & Upstream Flow at time t)
    # ------------------------------------------------------------------
    pivot_ratio = df.pivot(index="ts_t", columns="seg_id", values="ratio_t0")
    pivot_delta15 = df.pivot(index="ts_t", columns="seg_id", values="delta_ratio_15m")

    down_min_map = {}
    down_mean_map = {}
    down_delta_map = {}
    up_mean_map = {}

    for _, s_row in df_segs.iterrows():
        s_id = s_row["seg_id"]
        down_ids = [d for d in _parse_list_col(s_row["downstream_seg_ids"]) if d in pivot_ratio.columns]
        up_ids = [u for u in _parse_list_col(s_row["upstream_seg_ids"]) if u in pivot_ratio.columns]

        if down_ids:
            down_min_map[s_id] = pivot_ratio[down_ids].min(axis=1)
            down_mean_map[s_id] = pivot_ratio[down_ids].mean(axis=1)
            down_delta_map[s_id] = pivot_delta15[down_ids].mean(axis=1)
        else:
            down_min_map[s_id] = pivot_ratio[s_id]
            down_mean_map[s_id] = pivot_ratio[s_id]
            down_delta_map[s_id] = pivot_delta15[s_id]

        if up_ids:
            up_mean_map[s_id] = pivot_ratio[up_ids].mean(axis=1)
        else:
            up_mean_map[s_id] = pivot_ratio[s_id]

    df_neighbor = pd.DataFrame({
        "downstream_min_ratio_t0": pd.concat(down_min_map, names=["seg_id", "ts_t"]),
        "downstream_mean_ratio_t0": pd.concat(down_mean_map, names=["seg_id", "ts_t"]),
        "downstream_delta_15m": pd.concat(down_delta_map, names=["seg_id", "ts_t"]),
        "upstream_mean_ratio_t0": pd.concat(up_mean_map, names=["seg_id", "ts_t"]),
    }).reset_index()

    df = df.merge(df_neighbor, on=["seg_id", "ts_t"], how="left")
    df["spatial_gradient_t0"] = (df["downstream_mean_ratio_t0"] - df["ratio_t0"]).round(4)

    # ------------------------------------------------------------------
    # Group 7: Traffic Incidents (reported_time <= t < cleared_time)
    # ------------------------------------------------------------------
    df["active_incident_on_seg"] = 0
    df["incident_count_within_2km_1h"] = 0
    df["mins_since_last_incident"] = 180.0

    if cfg["feature_engineering"].get("enable_incidents", True) and len(df_incidents) > 0:
        df_incidents["reported_time"] = pd.to_datetime(df_incidents["reported_time"])
        df_incidents["cleared_time"] = pd.to_datetime(df_incidents["cleared_time"])
        for _, inc in df_incidents.iterrows():
            s_id = inc["seg_id"]
            rep_t = inc["reported_time"]
            clr_t = inc["cleared_time"]
            mask_active = (df["seg_id"] == s_id) & (df["ts_t"] >= rep_t) & (df["ts_t"] <= clr_t)
            df.loc[mask_active, "active_incident_on_seg"] = 1
            df.loc[mask_active, "incident_count_within_2km_1h"] += 1
            mins_elapsed = (df.loc[mask_active, "ts_t"] - rep_t).dt.total_seconds() / 60.0
            df.loc[mask_active, "mins_since_last_incident"] = np.minimum(
                df.loc[mask_active, "mins_since_last_incident"], mins_elapsed
            )

    # ------------------------------------------------------------------
    # Group 8: Static Road & Traffy Spatial Vulnerability + Rain Interaction
    # ------------------------------------------------------------------
    df = df.merge(
        df_traffy[[
            "seg_id",
            "district_flood_density_per_km2",
            "seg_buffer500m_flood_count",
        ]].rename(
            columns={
                "district_flood_density_per_km2": "traffy_flood_density_district",
                "seg_buffer500m_flood_count": "traffy_flood_buffer500m",
            }
        ),
        on="seg_id",
        how="left",
    )
    dir_map = {"inbound": 1, "outbound": 0, "other": 2}
    df["direction_code"] = df["direction"].map(dir_map).fillna(2).astype(int)
    df["rain_x_traffy_flood_interaction"] = (
        df["rain_cum_3h"] * np.log1p(df["traffy_flood_buffer500m"].fillna(0))
    ).round(4)

    # ------------------------------------------------------------------
    # Target Variables & Onset Event Definition (with 0.55 Buffer)
    # ------------------------------------------------------------------
    cong_th = float(cfg["feature_engineering"]["congestion_threshold_default"])
    onset_buf = float(cfg["feature_engineering"]["onset_freeflow_buffer"])

    df = df.dropna(subset=["target_ratio_t60", "ratio_lag_60m"]).copy()
    df = df[df["split"].isin(["train", "val", "test"])].reset_index(drop=True)

    df["target_is_congested_t60"] = (df["target_ratio_t60"] < cong_th).astype(int)
    df["is_onset_event"] = (
        (df["ratio_t0"] >= onset_buf) & (df["target_is_congested_t60"] == 1)
    ).astype(int)

    # Ensure no NaNs remain in feature columns
    for col in ALL_FEATURE_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(0.0)

    save_table(df, gold_dir / "gold_features.parquet", layer="gold")

    split_counts = df["split"].value_counts().to_dict()
    onset_count = int(df["is_onset_event"].sum())
    log_info(
        f"Gold Feature Store: รวม {len(df):,} แถว | {len(ALL_FEATURE_COLS)} ฟีเจอร์ (8 กลุ่ม) | "
        f"Train={split_counts.get('train', 0):,}, Val={split_counts.get('val', 0):,}, Test={split_counts.get('test', 0):,} | "
        f"Onset Events={onset_count:,}"
    )
    log_success("บันทึก data/04_gold/gold_features.parquet สำเร็จ")
    return {
        "gold_total_rows": len(df),
        "num_features": len(ALL_FEATURE_COLS),
        "train_rows": int(split_counts.get("train", 0)),
        "val_rows": int(split_counts.get("val", 0)),
        "test_rows": int(split_counts.get("test", 0)),
        "onset_events": onset_count,
    }
