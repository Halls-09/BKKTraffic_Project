"""
Real-Data Hourly Road-Link Feature Builder (Streaming directly from CleanData_BKK-*.zip).

Enforces Proposal Data Contract 4.2 (Page 5):
  - 100% Location Table Join Coverage (both `from_loc` and `to_loc` must exist in `itic_location_table_bkk_clean.csv`)
  - No self-loops (`from_loc != to_loc`)
  - Exact directional road segment geometry: `(lat_from, lon_from) -> (lat_to, lon_to)`,
    lane-offset midpoint `(lat, lon)` (zero duplicate stacked coordinates!), `length_km`,
    and human-readable directional label `segment_label` (e.g. "ถนนพหลโยธิน (แยกสะพานควาย → แยกอารีย์)").
"""
from __future__ import annotations

import glob
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logger import log_info, log_step, log_success
from src.utils.storage import ROOT_DIR, load_config, load_table, save_table


def _process_zip_file(zip_path: str) -> pd.DataFrame:
    """Worker: stream-read all daily iTIC CSVs inside one zip and aggregate to hourly per link."""
    parts: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as zf:
        itic_members = [n for n in zf.namelist() if "/iTIC/" in n and n.endswith(".csv")]
        for member in itic_members:
            with zf.open(member) as f:
                df = pd.read_csv(
                    f,
                    usecols=["snapshot_time", "from_loc", "to_loc", "link_speed_kmh", "adm2_pcode"],
                    dtype={
                        "from_loc": "int32",
                        "to_loc": "int32",
                        "link_speed_kmh": "int8",
                        "adm2_pcode": "str",
                    },
                )
            if df.empty:
                continue
            df = df[df["from_loc"] != df["to_loc"]]
            df["ts_hour"] = df["snapshot_time"].str.slice(0, 13) + ":00:00"
            df["is_red"] = (df["link_speed_kmh"] == 3).astype("float32")
            df["is_yellow"] = (df["link_speed_kmh"] == 2).astype("float32")
            df["severity"] = ((df["link_speed_kmh"] - 1.0) / 2.0).astype("float32")
            df["dcode"] = df["adm2_pcode"].str.slice(2).astype("int16")

            agg = (
                df.groupby(["from_loc", "to_loc", "ts_hour"], sort=False)
                .agg(
                    dcode=("dcode", "first"),
                    n_obs=("link_speed_kmh", "count"),
                    red_rate=("is_red", "mean"),
                    yellow_rate=("is_yellow", "mean"),
                    severity=("severity", "mean"),
                )
                .reset_index()
            )
            agg["n_obs"] = agg["n_obs"].astype("int8")
            agg["red_rate"] = agg["red_rate"].astype("float32")
            agg["yellow_rate"] = agg["yellow_rate"].astype("float32")
            agg["severity"] = agg["severity"].astype("float32")
            parts.append(agg)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _load_context_tables(zip_paths: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load district_hourly_panel.csv and itic_location_table_bkk_clean.csv directly from zips."""
    panel_df = None
    loc_df = None
    for zp in zip_paths:
        with zipfile.ZipFile(zp) as zf:
            for name in zf.namelist():
                if name.endswith("district_hourly_panel.csv") and panel_df is None:
                    with zf.open(name) as f:
                        panel_df = pd.read_csv(f)
                elif name.endswith("itic_location_table_bkk_clean.csv") and loc_df is None:
                    with zf.open(name) as f:
                        loc_df = pd.read_csv(f)
        if panel_df is not None and loc_df is not None:
            break

    if panel_df is None or loc_df is None:
        raise FileNotFoundError("ไม่พบไฟล์ district_hourly_panel.csv หรือ itic_location_table_bkk_clean.csv ใน zip")
    return panel_df, loc_df


def _build_link_geometry_table(links_df: pd.DataFrame, loc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes accurate directional road segment geometry for each (from_loc, to_loc) pair:
      - lat_from, lon_from, lat_to, lon_to
      - lat, lon: segment midpoint with left-hand traffic directional offset (zero overlapping pins!)
      - length_km: Haversine distance in km
      - segment_label: "ถนน... (แยกต้นทาง → แยกปลายทาง)"
    """
    loc_clean = (
        loc_df[["loc_code", "lat", "lon", "road_name_th", "first_name_th", "second_name_th"]]
        .dropna(subset=["loc_code", "lat", "lon"])
        .drop_duplicates("loc_code")
        .copy()
    )
    loc_clean["loc_code"] = loc_clean["loc_code"].astype("int32")

    m = links_df[["link_id", "from_loc", "to_loc"]].drop_duplicates("link_id").copy()
    m = m.merge(
        loc_clean.rename(
            columns={
                "loc_code": "from_loc",
                "lat": "lat_from",
                "lon": "lon_from",
                "road_name_th": "road_from",
                "first_name_th": "name_from",
                "second_name_th": "sec_from",
            }
        ),
        on="from_loc",
        how="inner",
    )
    m = m.merge(
        loc_clean.rename(
            columns={
                "loc_code": "to_loc",
                "lat": "lat_to",
                "lon": "lon_to",
                "road_name_th": "road_to",
                "first_name_th": "name_to",
                "second_name_th": "sec_to",
            }
        ),
        on="to_loc",
        how="inner",
    )

    # Midpoint + directional perpendicular offset (approx 22m) so opposite directions don't stack
    dlat = m["lat_to"].to_numpy(dtype=np.float64) - m["lat_from"].to_numpy(dtype=np.float64)
    dlon = m["lon_to"].to_numpy(dtype=np.float64) - m["lon_from"].to_numpy(dtype=np.float64)
    norm = np.hypot(dlat, dlon)
    norm_safe = np.where(norm < 1e-7, 1.0, norm)
    # Normal vector (-dlon, dlat) / norm
    offset_deg = 0.00022
    perp_lat = (-dlon / norm_safe) * offset_deg
    perp_lon = (dlat / norm_safe) * offset_deg

    m["lat"] = ((m["lat_from"] + m["lat_to"]) / 2.0 + perp_lat).astype("float32")
    m["lon"] = ((m["lon_from"] + m["lon_to"]) / 2.0 + perp_lon).astype("float32")
    m["lat_from"] = m["lat_from"].astype("float32")
    m["lon_from"] = m["lon_from"].astype("float32")
    m["lat_to"] = m["lat_to"].astype("float32")
    m["lon_to"] = m["lon_to"].astype("float32")
    m["length_km"] = np.clip(np.hypot(dlat * 110.6, dlon * 107.8), 0.05, 25.0).astype("float32")

    def _fmt_label(r: pd.Series) -> str:
        road = str(r["road_from"]) if pd.notna(r["road_from"]) else (str(r["road_to"]) if pd.notna(r["road_to"]) else "")
        n_from = str(r["sec_from"]) if pd.notna(r["sec_from"]) else (str(r["name_from"]) if pd.notna(r["name_from"]) else f"#{r['from_loc']}")
        n_to = str(r["sec_to"]) if pd.notna(r["sec_to"]) else (str(r["name_to"]) if pd.notna(r["name_to"]) else f"#{r['to_loc']}")
        for prefix in ("ถนน", "แยก", "ซอย"):
            if n_from == road:
                n_from = f"จุด #{r['from_loc']}"
        if road and road != "nan":
            if n_from != n_to:
                return f"{road} ({n_from} → {n_to})"
            return f"{road} ({n_from})"
        return f"{n_from} → {n_to}"

    m["road_name_th"] = m["road_from"].combine_first(m["road_to"]).combine_first(m["name_from"])
    m["first_name_th"] = m["name_from"]
    m["segment_label"] = m.apply(_fmt_label, axis=1)

    return m[
        [
            "link_id",
            "lat",
            "lon",
            "lat_from",
            "lon_from",
            "lat_to",
            "lon_to",
            "length_km",
            "road_name_th",
            "first_name_th",
            "segment_label",
        ]
    ]


def upgrade_existing_gold_geometry() -> Path:
    """
    Fast in-place upgrade of `real_link_hourly_features.pkl`:
    Enforces Data Contract 4.2 (100% location join coverage, no self-loops) and adds
    accurate segment midpoint/endpoint geometry (`lat_from`, `lon_from`, `lat_to`, `lon_to`, `segment_label`).
    """
    t0 = time.time()
    cfg = load_config()
    gold_path = ROOT_DIR / cfg["paths"]["gold_data"]
    zip_paths = sorted(glob.glob(str(ROOT_DIR / cfg["paths"]["raw_zips_glob"])))
    _, loc_df = _load_context_tables(zip_paths)

    df = load_table(gold_path)
    before_links = int(df["link_id"].nunique())

    # Remove self-loops
    df = df[df["from_loc"] != df["to_loc"]].copy()

    geom = _build_link_geometry_table(df[["link_id", "from_loc", "to_loc"]].drop_duplicates(), loc_df)
    valid_ids = set(geom["link_id"])
    df = df[df["link_id"].isin(valid_ids)].copy()

    drop_cols = [
        c
        for c in ("lat", "lon", "lat_from", "lon_from", "lat_to", "lon_to", "length_km", "road_name_th", "first_name_th", "segment_label")
        if c in df.columns
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df.merge(geom, on="link_id", how="inner")
    save_table(df, gold_path)
    after_links = int(df["link_id"].nunique())
    dup_coords = int(geom.duplicated(["lat", "lon"]).sum())
    log_success(
        f"อัปเกรดพิกัดช่วงถนนตาม Data Contract 4.2 สำเร็จ ({before_links} -> {after_links} ช่วงถนนที่มีพิกัดต้นทาง-ปลายทางครบ 100%, "
        f"พิกัดซ้ำ={dup_coords}, {len(df):,} แถว, {time.time()-t0:.1f}s)"
    )
    return gold_path


def build_real_dataset() -> Path:
    t0 = time.time()
    cfg = load_config()
    zip_paths = sorted(glob.glob(str(ROOT_DIR / cfg["paths"]["raw_zips_glob"])))
    if not zip_paths:
        raise FileNotFoundError(f"ไม่พบไฟล์ zip ตาม pattern: {cfg['paths']['raw_zips_glob']}")

    log_step("PREPARE_DATA", f"เริ่มสตรีมข้อมูล iTIC จาก {len(zip_paths)} ไฟล์ zip (ไม่แตกไฟล์ 20GB ลงดิสก์)...")

    panel_df, loc_df = _load_context_tables(zip_paths)
    panel_df["ts_hour"] = pd.to_datetime(panel_df["ts"]).dt.tz_localize(None)
    panel_df["dcode"] = panel_df["dcode"].astype("int16")
    panel_df["split"] = panel_df["split"].replace({"valid": "val"})

    valid_loc_codes = set(loc_df["loc_code"].dropna().astype("int32"))

    hourly_parts: list[pd.DataFrame] = []
    max_workers = min(4, len(zip_paths))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_zip_file, zp): Path(zp).name for zp in zip_paths}
        for fut in as_completed(futures):
            zname = futures[fut]
            part = fut.result()
            if not part.empty:
                hourly_parts.append(part)
                log_info(f"อ่าน {zname} เสร็จแล้ว ({len(part):,} link-hours)")

    raw_hourly = pd.concat(hourly_parts, ignore_index=True)
    # Enforce Data Contract 4.2: both from_loc and to_loc in Location Table, and from_loc != to_loc
    raw_hourly = raw_hourly[
        (raw_hourly["from_loc"] != raw_hourly["to_loc"])
        & (raw_hourly["from_loc"].isin(valid_loc_codes))
        & (raw_hourly["to_loc"].isin(valid_loc_codes))
    ].copy()
    raw_hourly["ts_hour"] = pd.to_datetime(raw_hourly["ts_hour"])
    raw_hourly = (
        raw_hourly.groupby(["from_loc", "to_loc", "ts_hour"], as_index=False)
        .agg(
            dcode=("dcode", "first"),
            n_obs=("n_obs", "sum"),
            red_rate=("red_rate", "mean"),
            yellow_rate=("yellow_rate", "mean"),
            severity=("severity", "mean"),
        )
    )
    raw_hourly["link_id"] = raw_hourly["from_loc"].astype(str) + "_" + raw_hourly["to_loc"].astype(str)

    total_hours = int(panel_df["ts_hour"].nunique())
    min_comp = float(cfg["dataset"]["min_completeness"])
    max_links = int(cfg["dataset"]["max_links"])

    link_stats = (
        raw_hourly.groupby(["link_id", "from_loc", "to_loc", "dcode"], as_index=False)
        .agg(n_hours=("ts_hour", "count"), mean_red=("red_rate", "mean"), std_red=("red_rate", "std"))
    )
    link_stats["completeness"] = link_stats["n_hours"] / total_hours
    eligible = link_stats[
        (link_stats["completeness"] >= min_comp)
        & (link_stats["mean_red"] >= 0.03)
        & (link_stats["std_red"] >= 0.05)
    ].copy()
    eligible = eligible.sort_values(["completeness", "std_red"], ascending=[False, False]).head(max_links)
    selected_links = set(eligible["link_id"])

    df = raw_hourly[raw_hourly["link_id"].isin(selected_links)].copy()
    all_hours = pd.date_range(panel_df["ts_hour"].min(), panel_df["ts_hour"].max(), freq="1h")
    meta_links = eligible[["link_id", "from_loc", "to_loc", "dcode"]].drop_duplicates("link_id")
    grid = meta_links.merge(pd.DataFrame({"ts_hour": all_hours}), how="cross")
    df = grid.merge(
        df[["link_id", "ts_hour", "n_obs", "red_rate", "yellow_rate", "severity"]],
        on=["link_id", "ts_hour"],
        how="left",
    )
    df = df.sort_values(["link_id", "ts_hour"]).reset_index(drop=True)

    for col in ("red_rate", "yellow_rate", "severity"):
        df[col] = df.groupby("link_id")[col].ffill(limit=1)
    df["n_obs"] = df["n_obs"].fillna(0).astype("int8")

    # Spatial Graph Neighbors at time t
    from_node_red = (
        df.groupby(["from_loc", "ts_hour"], as_index=False)["red_rate"]
        .mean()
        .rename(columns={"from_loc": "to_loc", "red_rate": "downstream_red_rate_t0"})
    )
    to_node_red = (
        df.groupby(["to_loc", "ts_hour"], as_index=False)["red_rate"]
        .mean()
        .rename(columns={"to_loc": "from_loc", "red_rate": "upstream_red_rate_t0"})
    )
    dist_red = (
        df.groupby(["dcode", "ts_hour"], as_index=False)["red_rate"]
        .mean()
        .rename(columns={"red_rate": "district_red_rate_t0"})
    )

    df = df.merge(from_node_red, on=["to_loc", "ts_hour"], how="left")
    df = df.merge(to_node_red, on=["from_loc", "ts_hour"], how="left")
    df = df.merge(dist_red, on=["dcode", "ts_hour"], how="left")
    df["downstream_red_rate_t0"] = df["downstream_red_rate_t0"].fillna(df["district_red_rate_t0"]).astype("float32")
    df["upstream_red_rate_t0"] = df["upstream_red_rate_t0"].fillna(df["district_red_rate_t0"]).astype("float32")
    df["district_red_rate_t0"] = df["district_red_rate_t0"].astype("float32")

    df = df.sort_values(["link_id", "ts_hour"]).reset_index(drop=True)
    grp = df.groupby("link_id")["red_rate"]

    df["red_rate_t0"] = df["red_rate"].astype("float32")
    df["yellow_rate_t0"] = df["yellow_rate"].astype("float32")
    df["severity_t0"] = df["severity"].astype("float32")
    df["n_obs_t0"] = df["n_obs"].astype("int8")
    df["red_rate_lag_1h"] = grp.shift(1).astype("float32")
    df["red_rate_lag_2h"] = grp.shift(2).astype("float32")
    df["red_rate_lag_3h"] = grp.shift(3).astype("float32")
    df["red_rate_lag_24h"] = grp.shift(24).astype("float32")
    df["delta_red_1h"] = (df["red_rate_t0"] - df["red_rate_lag_1h"]).astype("float32")
    df["delta_red_2h"] = (df["red_rate_t0"] - df["red_rate_lag_2h"]).astype("float32")
    df["rolling_mean_3h"] = grp.transform(lambda s: s.rolling(3, min_periods=1).mean()).astype("float32")
    df["rolling_std_3h"] = grp.transform(lambda s: s.rolling(3, min_periods=2).std()).fillna(0.0).astype("float32")
    df["spatial_gradient_t0"] = (df["downstream_red_rate_t0"] - df["red_rate_t0"]).astype("float32")

    df["target_ts"] = df["ts_hour"] + pd.Timedelta(hours=1)
    df["target_red_rate_t1h"] = grp.shift(-1).astype("float32")
    df = df.dropna(subset=["red_rate_t0", "red_rate_lag_1h", "red_rate_lag_24h", "target_red_rate_t1h"]).copy()

    wx_cols = [
        "dcode", "ts_hour", "district", "district_en", "area_sq_km",
        "precip_mm", "precip_3h_mm", "precip_6h_mm", "temp_c", "is_rainy_gt_1mm",
        "train_flood_per_sq_km", "train_traffic_per_sq_km", "train_flood_per_day",
    ]
    wx_t0 = panel_df[wx_cols].rename(
        columns={
            "precip_mm": "precip_mm_t0",
            "precip_3h_mm": "precip_3h_mm_t0",
            "precip_6h_mm": "precip_6h_mm_t0",
            "temp_c": "temp_c_t0",
            "is_rainy_gt_1mm": "is_rainy_gt_1mm_t0",
        }
    )
    df = df.merge(wx_t0, on=["dcode", "ts_hour"], how="inner")

    cal_cols = [
        "dcode", "ts_hour", "dow", "is_weekend", "is_friday", "is_holiday",
        "is_working_day", "is_payday", "is_school_term", "split",
    ]
    cal_t1 = panel_df[cal_cols].rename(columns={"ts_hour": "target_ts"})
    df = df.merge(cal_t1, on=["dcode", "target_ts"], how="inner")

    for bcol in ("is_rainy_gt_1mm_t0", "is_weekend", "is_friday", "is_holiday", "is_working_day", "is_payday", "is_school_term"):
        df[bcol] = df[bcol].astype("int8")

    df["target_hour"] = df["target_ts"].dt.hour.astype("int8")
    df["current_hour"] = df["ts_hour"].dt.hour.astype("int8")
    df["target_hour_sin"] = np.sin(2 * np.pi * df["target_hour"] / 24.0).astype("float32")
    df["target_hour_cos"] = np.cos(2 * np.pi * df["target_hour"] / 24.0).astype("float32")
    df["is_morning_rush_t1h"] = df["target_hour"].between(7, 9).astype("int8")
    df["is_evening_rush_t1h"] = df["target_hour"].between(16, 19).astype("int8")
    df["is_friday_evening"] = ((df["is_friday"] == 1) & (df["target_hour"].between(16, 21))).astype("int8")

    geom = _build_link_geometry_table(meta_links, loc_df)
    df = df.merge(geom, on="link_id", how="inner")

    df["rain_x_flood_interaction"] = (
        df["precip_3h_mm_t0"] * np.log1p(df["train_flood_per_sq_km"].clip(lower=0))
    ).astype("float32")

    train_only = df[df["split"] == "train"]
    hist_profile = (
        train_only.groupby(["link_id", "target_hour", "is_weekend"], as_index=False)["target_red_rate_t1h"]
        .agg(
            hist_avg_red_target_slot="mean",
            hist_p75_red_target_slot=lambda s: float(np.percentile(s, 75)),
        )
    )
    hist_curr = (
        train_only.groupby(["link_id", "current_hour", "is_weekend"], as_index=False)["red_rate_t0"]
        .mean()
        .rename(columns={"red_rate_t0": "_hist_curr"})
    )

    df = df.merge(hist_profile, on=["link_id", "target_hour", "is_weekend"], how="left")
    df = df.merge(hist_curr, on=["link_id", "current_hour", "is_weekend"], how="left")

    global_train_mean = float(train_only["target_red_rate_t1h"].mean())
    df["hist_avg_red_target_slot"] = df["hist_avg_red_target_slot"].fillna(global_train_mean).astype("float32")
    df["hist_p75_red_target_slot"] = df["hist_p75_red_target_slot"].fillna(global_train_mean).astype("float32")
    df["current_diff_from_hist"] = (df["red_rate_t0"] - df["_hist_curr"].fillna(global_train_mean)).astype("float32")
    df = df.drop(columns=["_hist_curr", "current_hour"])

    cong_th = float(cfg["dataset"]["congestion_threshold"])
    onset_max = float(cfg["dataset"]["onset_freeflow_max"])
    df["target_is_congested_t1h"] = (df["target_red_rate_t1h"] >= cong_th).astype("int8")
    df["is_onset_event"] = ((df["red_rate_t0"] < onset_max) & (df["target_is_congested_t1h"] == 1)).astype("int8")

    out_path = save_table(df, cfg["paths"]["gold_data"])
    log_success(
        f"สร้างชุดข้อมูลจริงสำเร็จ -> {out_path.relative_to(ROOT_DIR)} "
        f"({len(df):,} แถว, {df['link_id'].nunique()} ช่วงถนน, 100% Location Join Coverage, {time.time()-t0:.1f}s)"
    )
    return out_path
