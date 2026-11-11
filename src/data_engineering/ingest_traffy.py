"""
Ingests Traffy Fondue Citizen Reports into Silver Layer:
1. Deduplicates identical complaints (addressing the 109k duplicate finding in EDA)
2. Filters strictly to Train Period (<= train_end) to prevent temporal leakage
3. Converts raw reports into Static Spatial Vulnerability Features (500m Buffer + District Density)
   to immune the model against Platform Adoption Bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import log_info, log_step, log_success
from src.utils.storage import ROOT_DIR, load_config, load_contract, load_table, save_table


DISTRICT_AREA_KM2 = {
    "วัฒนา": 12.56,
    "ปทุมวัน": 8.37,
    "ห้วยขวาง": 15.03,
    "ลาดพร้าว": 21.50,
    "จตุจักร": 32.91,
    "ดินแดง": 8.35,
    "ราชเทวี": 7.13,
    "สาทร": 9.33,
    "คลองเตย": 13.00,
    "บางนา": 18.79,
    "บางกะปิ": 28.52,
    "บางรัก": 5.54,
}


def run_ingest_traffy() -> dict[str, int]:
    log_step("INGEST_TRAFFY", "ทำความสะอาด Traffy Fondue (Deduplication + Spatial Buffer 500m)...")
    cfg = load_config()
    contract = load_contract()
    bronze_dir = ROOT_DIR / cfg["paths"]["bronze_dir"]
    silver_dir = ROOT_DIR / cfg["paths"]["silver_dir"]

    df_raw = load_table(bronze_dir / "bronze_traffy_reports.parquet").copy()
    df_segments = load_table(silver_dir / "silver_road_segments.parquet")

    raw_count = len(df_raw)
    dedup_cols = contract["schemas"]["traffy_reports"]["deduplication"]["subset"]
    df_dedup = df_raw.drop_duplicates(subset=dedup_cols, keep="first").copy()
    dedup_removed = raw_count - len(df_dedup)

    # Filter strictly to Train cutoff to prevent future leakage
    train_end = pd.Timestamp(cfg["temporal_split"]["train_end"])
    df_dedup["timestamp"] = pd.to_datetime(df_dedup["timestamp"])
    df_train_traffy = df_dedup[df_dedup["timestamp"] <= train_end].copy()

    # District-level flood aggregates
    df_flood = df_train_traffy[df_train_traffy["type"] == "น้ำท่วม"]
    dist_counts = (
        df_flood.groupby("district_name")
        .size()
        .rename("district_flood_reports_train")
        .reset_index()
    )

    # Segment 500m buffer spatial join using coordinate distance (~0.0045 deg ~ 500m in Bangkok)
    spatial_rows = []
    t_lats = df_train_traffy["lat"].to_numpy()
    t_lons = df_train_traffy["lon"].to_numpy()
    t_types = df_train_traffy["type"].to_numpy()

    for _, seg in df_segments.iterrows():
        mid_lat = 0.5 * (seg["lat_start"] + seg["lat_end"])
        mid_lon = 0.5 * (seg["lon_start"] + seg["lon_end"])
        dist_deg = np.sqrt((t_lats - mid_lat) ** 2 + (t_lons - mid_lon) ** 2)
        in_buffer = dist_deg <= 0.0050  # ~500m radius

        flood_500m = int(np.sum(in_buffer & (t_types == "น้ำท่วม")))
        defect_500m = int(np.sum(in_buffer & (t_types == "ถนน/ผิวจราจร")))

        spatial_rows.append({
            "seg_id": seg["seg_id"],
            "district_name": seg["district_name"],
            "seg_buffer500m_flood_count": flood_500m,
            "seg_buffer500m_road_defect_count": defect_500m,
        })

    df_spatial = pd.DataFrame(spatial_rows).merge(dist_counts, on="district_name", how="left")
    df_spatial["district_flood_reports_train"] = df_spatial["district_flood_reports_train"].fillna(0).astype(int)
    df_spatial["district_area_km2"] = df_spatial["district_name"].map(DISTRICT_AREA_KM2).fillna(15.0)
    df_spatial["district_flood_density_per_km2"] = (
        df_spatial["district_flood_reports_train"] / df_spatial["district_area_km2"]
    ).round(3)
    df_spatial = df_spatial.drop(columns=["district_area_km2"])

    save_table(df_spatial, silver_dir / "silver_traffy_spatial_features.parquet", layer="silver")

    log_info(
        f"Traffy Deduplication: ลบข้อมูลซ้ำ {dedup_removed:,} รายการ (จาก {raw_count:,} เหลือ {len(df_dedup):,}) | "
        f"ใช้เฉพาะช่วง Train ({len(df_train_traffy):,} รายการ)"
    )
    log_success("สร้าง silver_traffy_spatial_features.parquet สำเร็จ")
    return {
        "traffy_raw": raw_count,
        "traffy_dedup_removed": dedup_removed,
        "traffy_train_used": len(df_train_traffy),
    }
