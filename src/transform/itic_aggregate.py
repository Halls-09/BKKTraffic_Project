"""Aggregate cleaned iTIC Bangkok daily CSVs into 15-minute road segment tables (Handoff 1).

Implements Proposal Sections 0, 4.2, 5 (H1-H5), 6.1, 6.2, 7, 8:
1. Unit of analysis: 1 row per directed road segment (`seg_id = "{from_loc}_{to_loc}"`) x 15-minute slot (`ts`).
2. Aggregation: median `speed_kmh`, max `congestion_level`, count `n_obs` within each 15-minute slot.
3. Dimension table `dim_segment`:
   - Maps `from_loc` and `to_loc` against `itic_location_table_bkk_clean.csv`
   - Computes midpoint (`lat`, `lon`), Haversine distance (`length_m`), direction (`inbound_outbound` toward central BKK),
     Bangkok `district`, nearest `weather_point`, and `downstream_seg_ids` (connected segments where `next.from_loc == this.to_loc`
     for Hypothesis H3: "สภาพถนนข้างหน้าช่วยจับ 'เริ่มติดใหม่'").
4. Quality Gate 4.2:
   - Completeness per segment-day: >= 80% of 96 slots (>= 77 slots) and >= 90% in 06:00-22:00 (>= 58 of 64 slots)
     -> saves `quality_completeness_seg_day.parquet` & `.csv` for Apichet's Missing Heatmap (EDA 6.1).
   - Plausibility flags: `flag_ratio_gt_1_2` (`ratio > 1.2`) and `flag_jump_gt_60` (`|speed_t - speed_{t-1}| > 60 km/h`).
5. Leakage Controls 6.2 & Target Formulation 8:
   - `free_flow_kmh` = 90th percentile of `speed_kmh` per segment computed STRICTLY on the Train split (`2022-06-01..2022-08-23`).
   - `ratio = speed_kmh / free_flow_kmh`, `is_congested = ratio < 0.5`.
   - Computes `target_ratio` at exact timestamp `ts + 60 min` via self-join on `(seg_id, ts + 60 min)` so missing slots
     never shift timestamps across gaps!
   - `target_congested = target_ratio < 0.5`, `new_onset = (~is_congested) & target_congested`.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import CFG, TZ, get_logger, path, write_json

log = get_logger("transform.itic_aggregate")

BKK_CENTER_LAT = 13.7563
BKK_CENTER_LON = 100.5018


def haversine_m(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Compute Haversine distance in meters between two (lat, lon) arrays."""
    r = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def aggregate_daily_file_to_15min(csv_file: Path) -> pd.DataFrame:
    """Read one daily cleaned CSV and aggregate to (seg_id, from_loc, to_loc, ts_15m)."""
    usecols = ["snapshot_time", "from_loc", "to_loc", "link_speed_kmh", "congestion_level"]
    df = pd.read_csv(
        csv_file,
        usecols=usecols,
        dtype={
            "from_loc": "int32",
            "to_loc": "int32",
            "link_speed_kmh": "float32",
            "congestion_level": "int8",
        },
    )
    df = df[df["link_speed_kmh"].between(0.001, 150.0)].copy()
    ts = pd.to_datetime(df["snapshot_time"], errors="coerce")
    df["ts"] = ts.dt.floor("15min").dt.tz_localize(TZ)

    agg = (
        df.groupby(["from_loc", "to_loc", "ts"], as_index=False)
        .agg(
            speed_kmh=("link_speed_kmh", "median"),
            congestion_level=("congestion_level", "max"),
            n_obs=("link_speed_kmh", "size"),
        )
    )
    agg["seg_id"] = agg["from_loc"].astype(str) + "_" + agg["to_loc"].astype(str)
    agg["speed_kmh"] = agg["speed_kmh"].astype("float32")
    agg["n_obs"] = agg["n_obs"].astype("int16")
    return agg[["seg_id", "from_loc", "to_loc", "ts", "speed_kmh", "congestion_level", "n_obs"]]


def build_dim_segment(seg_pairs: pd.DataFrame) -> pd.DataFrame:
    """Build dim_segment reference table with coordinates, district, inbound/outbound, and downstream graph."""
    loc_csv = path("processed") / "csv" / "itic_location_table" / "itic_location_table_bkk_clean.csv"
    loc = pd.read_csv(loc_csv)
    dim_dist = pd.read_parquet(path("processed") / "dim_district.parquet")
    pcode_to_th = dict(zip("TH" + dim_dist["dcode"].astype(str), dim_dist["district"]))
    pcode_to_wpt = dict(zip("TH" + dim_dist["dcode"].astype(str), dim_dist["weather_point"]))

    loc_sub = loc[
        [
            "loc_code",
            "road_name_en",
            "road_name_th",
            "first_name_en",
            "first_name_th",
            "lat",
            "lon",
            "neg_offset_loc",
            "pos_offset_loc",
            "urban",
            "adm2_pcode",
            "adm2_name_en",
            "adm2_name_th",
        ]
    ].drop_duplicates("loc_code")

    dim = seg_pairs[["seg_id", "from_loc", "to_loc"]].drop_duplicates("seg_id").copy()
    dim = dim.merge(
        loc_sub.add_prefix("from_"),
        left_on="from_loc",
        right_on="from_loc_code",
        how="left",
    ).merge(
        loc_sub.add_prefix("to_"),
        left_on="to_loc",
        right_on="to_loc_code",
        how="left",
    )

    dim["lat"] = dim[["from_lat", "to_lat"]].mean(axis=1).round(6)
    dim["lon"] = dim[["from_lon", "to_lon"]].mean(axis=1).round(6)

    both_valid = dim["from_lat"].notna() & dim["to_lat"].notna()
    dim["length_m"] = np.nan
    dim.loc[both_valid, "length_m"] = haversine_m(
        dim.loc[both_valid, "from_lat"].to_numpy(),
        dim.loc[both_valid, "from_lon"].to_numpy(),
        dim.loc[both_valid, "to_lat"].to_numpy(),
        dim.loc[both_valid, "to_lon"].to_numpy(),
    ).round(1)

    d_from = haversine_m(
        dim["from_lat"].fillna(dim["lat"]).to_numpy(),
        dim["from_lon"].fillna(dim["lon"]).to_numpy(),
        BKK_CENTER_LAT,
        BKK_CENTER_LON,
    )
    d_to = haversine_m(
        dim["to_lat"].fillna(dim["lat"]).to_numpy(),
        dim["to_lon"].fillna(dim["lon"]).to_numpy(),
        BKK_CENTER_LAT,
        BKK_CENTER_LON,
    )
    dim["inbound_outbound"] = np.where(d_to < d_from, "inbound", "outbound")

    dim["road_name_th"] = dim["from_road_name_th"].fillna(dim["to_road_name_th"]).fillna("")
    dim["road_name_en"] = dim["from_road_name_en"].fillna(dim["to_road_name_en"]).fillna("")
    dim["from_place_th"] = dim["from_first_name_th"].fillna("")
    dim["to_place_th"] = dim["to_first_name_th"].fillna("")
    dim["adm2_pcode"] = dim["from_adm2_pcode"].fillna(dim["to_adm2_pcode"]).fillna("")
    dim["district_en"] = dim["from_adm2_name_en"].fillna(dim["to_adm2_name_en"]).fillna("")
    dim["district"] = (
        dim["adm2_pcode"]
        .map(pcode_to_th)
        .fillna(dim["from_adm2_name_th"])
        .fillna(dim["to_adm2_name_th"])
        .fillna("")
    )
    dim["weather_point"] = dim["adm2_pcode"].map(pcode_to_wpt).fillna("central")

    # 3-Tier Downstream Adjacency Graph (Hypothesis H3):
    # Tier 1 (exact_node): B.from_loc == A.to_loc (excluding immediate U-turn B.to_loc == A.from_loc)
    # Tier 2 (offset_hop): Walk 1-2 hops along TMC pos_offset_loc / neg_offset_loc chain on the same road
    # Tier 3 (junction_400m): Cross-road intersection transfer within 400m continuing in forward heading (cos_angle > -0.1)
    pos_map = {
        int(r.loc_code): int(r.pos_offset_loc)
        for r in loc_sub.dropna(subset=["pos_offset_loc"]).itertuples()
    }
    neg_map = {
        int(r.loc_code): int(r.neg_offset_loc)
        for r in loc_sub.dropna(subset=["neg_offset_loc"]).itertuples()
    }
    coord_map = {
        int(r.loc_code): (float(r.lat), float(r.lon))
        for r in loc_sub.dropna(subset=["lat", "lon"]).itertuples()
    }

    from_map: dict[int, list[tuple[int, str]]] = {}
    seg_records = list(dim[["seg_id", "from_loc", "to_loc"]].itertuples(index=False))
    for r in seg_records:
        from_map.setdefault(int(r.from_loc), []).append((int(r.to_loc), str(r.seg_id)))

    downstream_list = []
    downstream_count = []
    match_types = []

    for r in seg_records:
        fl, tl = int(r.from_loc), int(r.to_loc)
        # Tier 1: Exact node match
        c1 = [sid for t_lc, sid in from_map.get(tl, []) if t_lc != fl]
        if c1:
            downstream_list.append(",".join(c1))
            downstream_count.append(len(c1))
            match_types.append("tier1_exact_node")
            continue

        # Tier 2: 1-2 hops along TMC offset chain (same road)
        c2: list[str] = []
        for nxt_map in (pos_map, neg_map):
            cur = tl
            for _ in range(2):
                cur = nxt_map.get(cur)
                if not cur or cur == fl:
                    break
                c2.extend([sid for t_lc, sid in from_map.get(cur, []) if t_lc != fl and t_lc != tl])
                if c2:
                    break
        if c2:
            c2_uniq = list(dict.fromkeys(c2))
            downstream_list.append(",".join(c2_uniq))
            downstream_count.append(len(c2_uniq))
            match_types.append("tier2_offset_hop")
            continue

        # Tier 3: Cross-road junction transfer within 400m in forward direction
        c3: list[tuple[float, str]] = []
        if fl in coord_map and tl in coord_map:
            lat1, lon1 = coord_map[fl]
            lat2, lon2 = coord_map[tl]
            vx, vy = lon2 - lon1, lat2 - lat1
            vnorm = float(np.hypot(vx, vy))
            for b in seg_records:
                bfl, btl = int(b.from_loc), int(b.to_loc)
                if bfl == fl or btl == fl or (bfl not in coord_map) or (btl not in coord_map):
                    continue
                blat1, blon1 = coord_map[bfl]
                dist_m = float(np.hypot((blat1 - lat2) * 111000.0, (blon1 - lon2) * 108000.0))
                if dist_m <= 400.0:
                    blat2, blon2 = coord_map[btl]
                    wx, wy = blon2 - blon1, blat2 - blat1
                    wnorm = float(np.hypot(wx, wy))
                    cos_ang = (vx * wx + vy * wy) / (vnorm * wnorm + 1e-12)
                    if cos_ang > -0.1:
                        c3.append((dist_m, str(b.seg_id)))
        if c3:
            c3.sort(key=lambda x: x[0])
            top_c3 = [sid for _, sid in c3[:3]]
            downstream_list.append(",".join(top_c3))
            downstream_count.append(len(top_c3))
            match_types.append("tier3_junction_400m")
            continue

        downstream_list.append("NONE_TERMINUS_OR_BORDER")
        downstream_count.append(0)
        match_types.append("terminus_or_bkk_border")

    dim["downstream_seg_ids"] = downstream_list
    dim["n_downstream"] = downstream_count
    dim["downstream_match_type"] = match_types

    keep = [
        "seg_id",
        "from_loc",
        "to_loc",
        "road_name_th",
        "road_name_en",
        "from_place_th",
        "to_place_th",
        "lat",
        "lon",
        "length_m",
        "inbound_outbound",
        "adm2_pcode",
        "district",
        "district_en",
        "weather_point",
        "n_downstream",
        "downstream_match_type",
        "downstream_seg_ids",
    ]
    return dim[keep].sort_values("seg_id").reset_index(drop=True)


def assign_split_by_date(dates: pd.Series) -> pd.Series:
    sp = CFG["split"]
    out = pd.Series("unknown", index=dates.index, dtype="object")
    out.loc[dates.between(sp["train"][0], sp["train"][1])] = "train"
    out.loc[dates.between(sp["valid"][0], sp["valid"][1])] = "valid"
    out.loc[dates.between(sp["test"][0], sp["test"][1])] = "test"
    return out


def run_itic_aggregation(run_id: str = "manual") -> dict:
    """Aggregate all available daily iTIC CSVs into 15-minute segment tables and compute targets & quality flags."""
    traffic_root = path("processed") / "csv" / "iTic Traffic"
    daily_files = sorted(traffic_root.rglob("itic_bkk_clean_*.csv"))
    if not daily_files:
        raise FileNotFoundError(f"No daily iTIC CSV files found in {traffic_root}")

    log.info("Aggregating %s daily iTIC files to 15-minute segments using multiprocessing...", len(daily_files))
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 4)) as ex:
        frames = list(ex.map(aggregate_daily_file_to_15min, daily_files))

    df = pd.concat(frames, ignore_index=True)
    # Deduplicate (seg_id, ts) in case midnight snapshot appeared across two day files
    df = (
        df.groupby(["seg_id", "from_loc", "to_loc", "ts"], as_index=False)
        .agg(
            speed_kmh=("speed_kmh", "median"),
            congestion_level=("congestion_level", "max"),
            n_obs=("n_obs", "sum"),
        )
        .sort_values(["seg_id", "ts"])
        .reset_index(drop=True)
    )

    # 1) Build dim_segment and merge free_flow_kmh (90th percentile strictly on Train split)
    dim_seg = build_dim_segment(df[["seg_id", "from_loc", "to_loc"]])

    df["date"] = df["ts"].dt.strftime("%Y-%m-%d")
    df["split"] = assign_split_by_date(df["date"])

    train_mask = df["split"] == "train"
    ff_series = (
        df[train_mask]
        .groupby("seg_id")["speed_kmh"]
        .quantile(CFG["free_flow_percentile"] / 100.0)
        .round(2)
        .rename("free_flow_kmh")
    )
    dim_seg = dim_seg.merge(ff_series, on="seg_id", how="left")

    # 2) Compute congestion ratio and current congested status
    df = df.merge(dim_seg[["seg_id", "free_flow_kmh", "district", "weather_point", "inbound_outbound"]], on="seg_id", how="left")
    df["ratio"] = (df["speed_kmh"] / df["free_flow_kmh"]).round(4).astype("float32")
    df["is_congested"] = df["ratio"] < CFG["congested_threshold"]

    # 3) Exact t + 60 min target alignment via timestamp join (immune to missing slots!)
    df["target_ts"] = df["ts"] + pd.Timedelta(minutes=15 * CFG["horizon_steps"])
    future_lookup = df[["seg_id", "ts", "speed_kmh", "ratio"]].rename(
        columns={
            "ts": "target_ts",
            "speed_kmh": "target_speed_kmh",
            "ratio": "target_ratio",
        }
    )
    df = df.merge(future_lookup, on=["seg_id", "target_ts"], how="left")
    df["target_congested"] = df["target_ratio"] < CFG["congested_threshold"]
    df["new_onset"] = (~df["is_congested"]) & (df["target_ratio"] < CFG["congested_threshold"])

    # 4) Plausibility flags (Proposal 4.2: ratio <= 1.2; speed jump <= 60 km/h in 15 min)
    prev_lookup = df[["seg_id", "ts", "speed_kmh"]].copy()
    prev_lookup["ts"] = prev_lookup["ts"] + pd.Timedelta(minutes=15)
    prev_lookup = prev_lookup.rename(columns={"speed_kmh": "prev_15m_speed_kmh"})
    df = df.merge(prev_lookup, on=["seg_id", "ts"], how="left")

    df["flag_ratio_gt_1_2"] = df["ratio"] > CFG["quality"]["ratio_max"]
    df["flag_jump_gt_60"] = (df["speed_kmh"] - df["prev_15m_speed_kmh"]).abs() > CFG["quality"]["max_jump_kmh"]

    # 5) Completeness per segment-day (Proposal 4.2: >= 80% of 96 slots; 06:00-22:00 >= 90% of 64 slots)
    df["is_peak_window"] = df["ts"].dt.hour.between(6, 21)
    comp = (
        df.groupby(["seg_id", "date"], as_index=False)
        .agg(
            n_slots=("ts", "count"),
            n_peak_slots=("is_peak_window", "sum"),
        )
    )
    comp["completeness_all"] = (comp["n_slots"] / 96.0).round(4)
    comp["completeness_peak"] = (comp["n_peak_slots"] / 64.0).round(4)
    comp["flag_incomplete_day"] = (comp["n_slots"] < 0.80 * 96) | (comp["n_peak_slots"] < 0.90 * 64)

    df = df.merge(comp[["seg_id", "date", "flag_incomplete_day"]], on=["seg_id", "date"], how="left")
    df["event_time"] = df["ts"]
    df["available_time"] = df["ts"] + pd.Timedelta(minutes=15)

    # Save outputs
    proc = path("processed")
    csv_dir = proc / "csv"

    dim_seg.to_parquet(proc / "dim_segment.parquet", index=False)
    dim_seg.to_csv(csv_dir / "itic_location_table" / "dim_segment.csv", index=False, encoding="utf-8-sig")

    comp.to_parquet(proc / "quality_completeness_seg_day.parquet", index=False)
    comp.to_csv(csv_dir / "itic_location_table" / "quality_completeness_seg_day.csv", index=False, encoding="utf-8-sig")

    seg15_dir = proc / "itic_seg15"
    seg15_dir.mkdir(parents=True, exist_ok=True)
    df["month"] = df["date"].str.slice(0, 7)
    for mo, g in df.groupby("month"):
        g.drop(columns=["month"]).to_parquet(seg15_dir / f"itic_seg15_{mo}.parquet", index=False)

    report = {
        "daily_files_aggregated": len(daily_files),
        "total_15m_rows": len(df),
        "unique_segments": len(dim_seg),
        "duplicate_seg_ts": int(df.duplicated(["seg_id", "ts"]).sum()),
        "speed_min": float(df["speed_kmh"].min()),
        "speed_max": float(df["speed_kmh"].max()),
        "flag_ratio_gt_1_2_count": int(df["flag_ratio_gt_1_2"].sum()),
        "flag_jump_gt_60_count": int(df["flag_jump_gt_60"].sum()),
        "incomplete_seg_days_pct": round(float(comp["flag_incomplete_day"].mean() * 100), 2),
        "pass": int(df.duplicated(["seg_id", "ts"]).sum()) == 0 and float(df["speed_kmh"].max()) <= 150.0,
    }
    write_json(report, path("quality") / f"{run_id}_itic_seg15.json")
    log.info("Aggregated iTIC 15-min table: %s rows across %s segments", f"{len(df):,}", len(dim_seg))
    return report


if __name__ == "__main__":
    import os
    run_itic_aggregation()
