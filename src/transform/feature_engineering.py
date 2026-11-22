"""Build Handoff 2 ML Feature Tables in DuckDB (`gold.hist_segment_profile_train` & `gold.ml_features_15m`).

Implements all 8 Feature Groups from the Project Proposal (Section 6.2 & Hypotheses H1-H5)
with strict As-of Cutoff / Zero-Leakage controls:
  Group 1 - Current Traffic State (at t)
  Group 2 - Temporal Lag & Rolling Dynamics (strictly <= t, exact 15m/30m/45m/60m gap check)
  Group 3 - Historical Profile (Train-only Baseline B1: seg_id x hour x minute x day_type)
  Group 4 - Spatial / Downstream Road Network (H3: downstream_seg_ids at t + district-level at t)
  Group 5 - Weather & Rain x Rush Interaction (H2: available_time <= t)
  Group 6 - Time & Calendar at Target Time t+60m (H5: cyclical sin/cos, friday/payday evening)
  Group 7 - Static Road Segment Attributes (free_flow_kmh Train-only P90, length, lat/lon, direction)
  Group 8 - Traffy Fondue (H4: Train-only static district risk + dynamic 1h/3h/6h rolling <= t)

Usage:
  python -m src.transform.feature_engineering
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import duckdb

from src.common import CFG, get_logger, path, write_json

log = get_logger("transform.feature_engineering")

DB_PATH = path("raw").parent / "bkk_traffic.duckdb"
FEATURE_DB_PATH = path("raw").parent / "bkk_traffic_features.duckdb"


FEATURE_GROUPS: dict[str, list[str]] = {
    "id_and_metadata": [
        "seg_id", "ts", "date", "target_ts", "split", "event_time", "available_time",
    ],
    "targets": [
        "target_speed_kmh", "target_ratio", "target_congested", "new_onset",
    ],
    "quality_flags": [
        "flag_ratio_gt_1_2", "flag_jump_gt_60", "flag_incomplete_day",
    ],
    "group1_current_traffic": [
        "speed_kmh", "ratio", "congestion_level", "is_congested", "n_obs",
    ],
    "group2_temporal_lags_rolling": [
        "speed_lag_15m", "speed_lag_30m", "speed_lag_45m", "speed_lag_60m",
        "ratio_lag_15m", "ratio_lag_30m", "ratio_lag_45m", "ratio_lag_60m",
        "speed_diff_15m", "speed_diff_30m", "ratio_diff_15m",
        "speed_roll_mean_1h", "speed_roll_std_1h", "ratio_roll_mean_1h",
        "congested_slots_last_1h",
    ],
    "group3_historical_train_only": [
        "hist_avg_speed_t", "hist_avg_ratio_t", "hist_pct_congested_t",
        "hist_avg_speed_t60", "hist_avg_ratio_t60", "hist_pct_congested_t60",
        "speed_vs_hist_t", "ratio_vs_hist_t",
    ],
    "group4_spatial_downstream_h3": [
        "downstream_avg_ratio_t", "downstream_min_ratio_t",
        "downstream_avg_speed_t", "downstream_pct_congested_t",
        "downstream_n_active_t",
        "district_avg_ratio_t", "district_pct_congested_t", "district_avg_speed_t",
    ],
    "group5_weather_h2": [
        "precip_mm", "precip_3h_mm", "precip_6h_mm", "temp_c",
        "rain_bin", "is_rainy_gt_1mm", "is_rain_x_rush",
    ],
    "group6_time_calendar_h5": [
        "hour", "minute", "target_hour", "target_minute",
        "target_hour_sin", "target_hour_cos", "dow", "dow_sin", "dow_cos",
        "day_type", "is_weekend", "is_friday", "is_holiday",
        "is_bank_holiday", "is_working_day", "is_payday", "is_school_term",
        "is_friday_evening", "is_payday_evening",
    ],
    "group7_static_road": [
        "free_flow_kmh", "length_m", "lat", "lon", "inbound_outbound",
        "n_downstream", "district", "weather_point",
    ],
    "group8_traffy_h4": [
        "train_flood_reports", "train_flood_per_sq_km", "train_traffic_per_sq_km",
        "traffy_flood_last_1h", "traffy_flood_last_3h", "traffy_flood_last_6h",
        "traffy_traffic_last_1h", "traffy_traffic_last_3h",
    ],
}


def build_ml_features(run_id: str = "manual", force_rebuild: bool = False) -> dict:
    t0 = time.time()
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DuckDB warehouse not found at {DB_PATH}. Run duckdb_loader first.")

    if not force_rebuild and FEATURE_DB_PATH.exists():
        con_ro = None
        try:
            con_ro = duckdb.connect(str(FEATURE_DB_PATH), read_only=True)
            n_ml = con_ro.execute("SELECT COUNT(*) FROM gold.ml_features_15m").fetchone()[0]
            n_hist = con_ro.execute("SELECT COUNT(*) FROM gold.hist_segment_profile_train").fetchone()[0]
            n_cols = con_ro.execute(
                "SELECT COUNT(*) FROM information_schema.columns WHERE table_schema='gold' AND table_name='ml_features_15m'"
            ).fetchone()[0]
            if n_ml > 20_000_000 and n_hist > 100_000 and n_cols >= 75:
                log.info(
                    "gold.ml_features_15m already up-to-date in %s (%s rows x %s cols), skipping rebuild.",
                    FEATURE_DB_PATH.name,
                    f"{n_ml:,}",
                    n_cols,
                )
                return {
                    "cached": True,
                    "database_path": FEATURE_DB_PATH.relative_to(path("raw").parents[1]).as_posix(),
                    "ml_features_rows": int(n_ml),
                    "ml_features_cols": int(n_cols),
                    "hist_profile_rows": int(n_hist),
                    "elapsed_seconds": round(time.time() - t0, 2),
                    "pass": True,
                }
        except Exception:
            pass
        finally:
            if con_ro is not None:
                con_ro.close()

    FEATURE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(FEATURE_DB_PATH))
    con.execute("CREATE SCHEMA IF NOT EXISTS gold;")
    con.execute(f"ATTACH '{DB_PATH.as_posix()}' AS wh (READ_ONLY);")
    in_docker = Path("/opt/airflow").exists()
    n_threads = 4 if in_docker else max(4, (os.cpu_count() or 8) - 2)
    con.execute(f"PRAGMA threads={n_threads};")
    con.execute("PRAGMA memory_limit='8GB';")

    try:
        # ------------------------------------------------------------------
        # STEP 1: Train-Only Historical Profile (Group 3 & Baseline B1)
        # ------------------------------------------------------------------
        t1 = time.time()
        log.info("[FE 1/4] Building gold.hist_segment_profile_train (STRICTLY train split)...")
        con.execute(
            """
            CREATE OR REPLACE TABLE gold.hist_segment_profile_train AS
            SELECT
                g.seg_id,
                g.hour,
                g.minute,
                c.day_type,
                ROUND(AVG(g.speed_kmh), 2)::FLOAT AS hist_avg_speed,
                ROUND(AVG(g.ratio), 4)::FLOAT AS hist_avg_ratio,
                ROUND(AVG(CASE WHEN g.is_congested THEN 1.0 ELSE 0.0 END), 4)::FLOAT AS hist_pct_congested,
                COUNT(*)::SMALLINT AS hist_n_train_days
            FROM wh.gold.itic_seg15_targets g
            JOIN wh.silver.calendar_daily c ON g.date = c.date
            WHERE g.split = 'train'
            GROUP BY g.seg_id, g.hour, g.minute, c.day_type;
            """
        )
        n_hist = con.execute("SELECT COUNT(*) FROM gold.hist_segment_profile_train").fetchone()[0]
        log.info("[FE 1/4] gold.hist_segment_profile_train: %s rows in %.1fs", f"{n_hist:,}", time.time() - t1)

        # ------------------------------------------------------------------
        # STEP 2: Downstream Road Network Features at time t (Group 4 - H3)
        # ------------------------------------------------------------------
        t2 = time.time()
        log.info("[FE 2/4] Computing downstream road graph features at time t (H3)...")
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE tmp_downstream_edges AS
            SELECT
                seg_id,
                TRIM(unnest(string_split(downstream_seg_ids, ','))) AS downstream_seg_id
            FROM wh.silver.dim_segment
            WHERE downstream_seg_ids IS NOT NULL
              AND downstream_seg_ids <> ''
              AND downstream_seg_ids <> 'NONE';
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE tmp_downstream_15m AS
            SELECT
                e.seg_id,
                d.ts,
                ROUND(AVG(d.ratio), 4)::FLOAT AS downstream_avg_ratio_t,
                ROUND(MIN(d.ratio), 4)::FLOAT AS downstream_min_ratio_t,
                ROUND(AVG(d.speed_kmh), 2)::FLOAT AS downstream_avg_speed_t,
                ROUND(AVG(CASE WHEN d.is_congested THEN 1.0 ELSE 0.0 END), 4)::FLOAT AS downstream_pct_congested_t,
                COUNT(*)::TINYINT AS downstream_n_active_t
            FROM tmp_downstream_edges e
            JOIN wh.gold.itic_seg15_targets d
              ON e.downstream_seg_id = d.seg_id
            GROUP BY e.seg_id, d.ts;
            """
        )
        n_ds = con.execute("SELECT COUNT(*) FROM tmp_downstream_15m").fetchone()[0]
        log.info("[FE 2/4] tmp_downstream_15m: %s rows in %.1fs", f"{n_ds:,}", time.time() - t2)

        # ------------------------------------------------------------------
        # STEP 3: District Context + Rolling Traffy Windows (Groups 4 & 8)
        # Complete 15-min grid per district (585,600 rows) -> exact slot rolling!
        # ------------------------------------------------------------------
        t3 = time.time()
        log.info("[FE 3/4] Computing district traffic & rolling Traffy 1h/3h/6h windows (<= t)...")
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE tmp_district_15m_fe AS
            SELECT
                district,
                ts,
                itic_avg_ratio::FLOAT AS district_avg_ratio_t,
                itic_pct_congested::FLOAT AS district_pct_congested_t,
                itic_avg_speed_kmh::FLOAT AS district_avg_speed_t,
                SUM(traffy_flood_15m) OVER (
                    PARTITION BY district ORDER BY ts ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
                )::SMALLINT AS traffy_flood_last_1h,
                SUM(traffy_flood_15m) OVER (
                    PARTITION BY district ORDER BY ts ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
                )::SMALLINT AS traffy_flood_last_3h,
                SUM(traffy_flood_15m) OVER (
                    PARTITION BY district ORDER BY ts ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
                )::SMALLINT AS traffy_flood_last_6h,
                SUM(traffy_traffic_15m) OVER (
                    PARTITION BY district ORDER BY ts ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
                )::SMALLINT AS traffy_traffic_last_1h,
                SUM(traffy_traffic_15m) OVER (
                    PARTITION BY district ORDER BY ts ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
                )::SMALLINT AS traffy_traffic_last_3h
            FROM wh.gold.merged_district_15m;
            """
        )
        log.info("[FE 3/4] tmp_district_15m_fe ready in %.1fs", time.time() - t3)

        # ------------------------------------------------------------------
        # STEP 4: Final gold.ml_features_15m (All 8 Feature Groups)
        # ------------------------------------------------------------------
        t4 = time.time()
        log.info("[FE 4/4] Materializing gold.ml_features_15m (23.8M rows x 8 feature groups)...")
        con.execute(
            """
            CREATE OR REPLACE TABLE gold.ml_features_15m AS
            WITH base_lags AS (
                SELECT
                    g.seg_id,
                    g.ts,
                    g.date,
                    g.hour,
                    g.minute,
                    g.target_ts,
                    EXTRACT(HOUR FROM g.target_ts)::TINYINT AS target_hour,
                    EXTRACT(MINUTE FROM g.target_ts)::TINYINT AS target_minute,
                    strftime(g.target_ts, '%Y-%m-%d') AS target_date,
                    g.split,
                    g.event_time,
                    g.available_time,
                    -- Targets
                    g.target_speed_kmh::FLOAT AS target_speed_kmh,
                    g.target_ratio::FLOAT AS target_ratio,
                    g.target_congested,
                    g.new_onset,
                    -- Quality flags
                    g.flag_ratio_gt_1_2,
                    g.flag_jump_gt_60,
                    g.flag_incomplete_day,
                    -- Group 1: Current Traffic State
                    g.speed_kmh::FLOAT AS speed_kmh,
                    g.ratio::FLOAT AS ratio,
                    g.congestion_level,
                    g.is_congested,
                    g.n_obs,
                    -- Group 2: Exact-gap Temporal Lags (only populated if exact 15/30/45/60m prior exists)
                    CASE WHEN LAG(g.ts, 1) OVER w = g.ts - INTERVAL '15 minutes'
                         THEN LAG(g.speed_kmh, 1) OVER w ELSE NULL END::FLOAT AS speed_lag_15m,
                    CASE WHEN LAG(g.ts, 2) OVER w = g.ts - INTERVAL '30 minutes'
                         THEN LAG(g.speed_kmh, 2) OVER w ELSE NULL END::FLOAT AS speed_lag_30m,
                    CASE WHEN LAG(g.ts, 3) OVER w = g.ts - INTERVAL '45 minutes'
                         THEN LAG(g.speed_kmh, 3) OVER w ELSE NULL END::FLOAT AS speed_lag_45m,
                    CASE WHEN LAG(g.ts, 4) OVER w = g.ts - INTERVAL '60 minutes'
                         THEN LAG(g.speed_kmh, 4) OVER w ELSE NULL END::FLOAT AS speed_lag_60m,
                    CASE WHEN LAG(g.ts, 1) OVER w = g.ts - INTERVAL '15 minutes'
                         THEN LAG(g.ratio, 1) OVER w ELSE NULL END::FLOAT AS ratio_lag_15m,
                    CASE WHEN LAG(g.ts, 2) OVER w = g.ts - INTERVAL '30 minutes'
                         THEN LAG(g.ratio, 2) OVER w ELSE NULL END::FLOAT AS ratio_lag_30m,
                    CASE WHEN LAG(g.ts, 3) OVER w = g.ts - INTERVAL '45 minutes'
                         THEN LAG(g.ratio, 3) OVER w ELSE NULL END::FLOAT AS ratio_lag_45m,
                    CASE WHEN LAG(g.ts, 4) OVER w = g.ts - INTERVAL '60 minutes'
                         THEN LAG(g.ratio, 4) OVER w ELSE NULL END::FLOAT AS ratio_lag_60m,
                    -- Group 2: Rolling 1h window (strictly [ts - 45m, ts])
                    ROUND(AVG(g.speed_kmh) OVER w_1h, 2)::FLOAT AS speed_roll_mean_1h,
                    ROUND(COALESCE(STDDEV_SAMP(g.speed_kmh) OVER w_1h, 0.0), 2)::FLOAT AS speed_roll_std_1h,
                    ROUND(AVG(g.ratio) OVER w_1h, 4)::FLOAT AS ratio_roll_mean_1h,
                    SUM(CASE WHEN g.is_congested THEN 1 ELSE 0 END) OVER w_1h::TINYINT AS congested_slots_last_1h,
                    -- Keys for joins
                    g.district,
                    g.weather_point,
                    g.free_flow_kmh::FLOAT AS free_flow_kmh,
                    g.inbound_outbound
                FROM wh.gold.itic_seg15_targets g
                WINDOW
                    w AS (PARTITION BY g.seg_id ORDER BY g.ts),
                    w_1h AS (PARTITION BY g.seg_id ORDER BY g.ts RANGE BETWEEN INTERVAL '45 minutes' PRECEDING AND CURRENT ROW)
            )
            SELECT
                -- ID & Metadata
                b.seg_id,
                b.ts,
                b.date,
                b.target_ts,
                b.split,
                b.event_time,
                b.available_time,
                -- Targets
                b.target_speed_kmh,
                b.target_ratio,
                b.target_congested,
                b.new_onset,
                -- Quality Flags
                b.flag_ratio_gt_1_2,
                b.flag_jump_gt_60,
                b.flag_incomplete_day,
                -- Group 1: Current Traffic
                b.speed_kmh,
                b.ratio,
                b.congestion_level,
                b.is_congested,
                b.n_obs,
                -- Group 2: Temporal Lags & Rolling
                b.speed_lag_15m,
                b.speed_lag_30m,
                b.speed_lag_45m,
                b.speed_lag_60m,
                b.ratio_lag_15m,
                b.ratio_lag_30m,
                b.ratio_lag_45m,
                b.ratio_lag_60m,
                ROUND(b.speed_kmh - b.speed_lag_15m, 2)::FLOAT AS speed_diff_15m,
                ROUND(b.speed_kmh - b.speed_lag_30m, 2)::FLOAT AS speed_diff_30m,
                ROUND(b.ratio - b.ratio_lag_15m, 4)::FLOAT AS ratio_diff_15m,
                b.speed_roll_mean_1h,
                b.speed_roll_std_1h,
                b.ratio_roll_mean_1h,
                b.congested_slots_last_1h,
                -- Group 3: Historical Profile (Train-Only / Baseline B1)
                h0.hist_avg_speed AS hist_avg_speed_t,
                h0.hist_avg_ratio AS hist_avg_ratio_t,
                h0.hist_pct_congested AS hist_pct_congested_t,
                h60.hist_avg_speed AS hist_avg_speed_t60,
                h60.hist_avg_ratio AS hist_avg_ratio_t60,
                h60.hist_pct_congested AS hist_pct_congested_t60,
                ROUND(b.speed_kmh - h0.hist_avg_speed, 2)::FLOAT AS speed_vs_hist_t,
                ROUND(b.ratio - h0.hist_avg_ratio, 4)::FLOAT AS ratio_vs_hist_t,
                -- Group 4: Spatial / Downstream Road Network (H3)
                ds.downstream_avg_ratio_t,
                ds.downstream_min_ratio_t,
                ds.downstream_avg_speed_t,
                ds.downstream_pct_congested_t,
                COALESCE(ds.downstream_n_active_t, 0)::TINYINT AS downstream_n_active_t,
                dist.district_avg_ratio_t,
                dist.district_pct_congested_t,
                dist.district_avg_speed_t,
                -- Group 5: Weather (H2, available_time <= ts)
                w.precip_mm::FLOAT AS precip_mm,
                w.precip_3h_mm::FLOAT AS precip_3h_mm,
                w.precip_6h_mm::FLOAT AS precip_6h_mm,
                w.temp_c::FLOAT AS temp_c,
                w.rain_bin,
                w.is_rainy_gt_1mm,
                (COALESCE(w.is_rainy_gt_1mm, FALSE) AND (b.target_hour BETWEEN 7 AND 9 OR b.target_hour BETWEEN 16 AND 19)) AS is_rain_x_rush,
                -- Group 6: Time & Calendar at Target Time t+60m (H5)
                b.hour::TINYINT AS hour,
                b.minute::TINYINT AS minute,
                b.target_hour,
                b.target_minute,
                ROUND(SIN(2.0 * PI() * (b.target_hour + b.target_minute / 60.0) / 24.0), 4)::FLOAT AS target_hour_sin,
                ROUND(COS(2.0 * PI() * (b.target_hour + b.target_minute / 60.0) / 24.0), 4)::FLOAT AS target_hour_cos,
                c_tgt.dow::TINYINT AS dow,
                ROUND(SIN(2.0 * PI() * c_tgt.dow / 7.0), 4)::FLOAT AS dow_sin,
                ROUND(COS(2.0 * PI() * c_tgt.dow / 7.0), 4)::FLOAT AS dow_cos,
                c_tgt.day_type,
                c_tgt.is_weekend,
                c_tgt.is_friday,
                c_tgt.is_holiday,
                c_tgt.is_bank_holiday,
                c_tgt.is_working_day,
                c_tgt.is_payday,
                c_tgt.is_school_term,
                (c_tgt.is_friday AND b.target_hour BETWEEN 16 AND 20) AS is_friday_evening,
                (c_tgt.is_payday AND b.target_hour BETWEEN 16 AND 20) AS is_payday_evening,
                -- Group 7: Static Road Segment Attributes
                b.free_flow_kmh,
                s.length_m::FLOAT AS length_m,
                s.lat::FLOAT AS lat,
                s.lon::FLOAT AS lon,
                b.inbound_outbound,
                s.n_downstream::SMALLINT AS n_downstream,
                b.district,
                b.weather_point,
                -- Group 8: Traffy Fondue (H4: Train-only static + rolling <= ts)
                st.train_flood_reports::INTEGER AS train_flood_reports,
                st.train_flood_per_sq_km::FLOAT AS train_flood_per_sq_km,
                st.train_traffic_per_sq_km::FLOAT AS train_traffic_per_sq_km,
                COALESCE(dist.traffy_flood_last_1h, 0)::SMALLINT AS traffy_flood_last_1h,
                COALESCE(dist.traffy_flood_last_3h, 0)::SMALLINT AS traffy_flood_last_3h,
                COALESCE(dist.traffy_flood_last_6h, 0)::SMALLINT AS traffy_flood_last_6h,
                COALESCE(dist.traffy_traffic_last_1h, 0)::SMALLINT AS traffy_traffic_last_1h,
                COALESCE(dist.traffy_traffic_last_3h, 0)::SMALLINT AS traffy_traffic_last_3h
            FROM base_lags b
            JOIN wh.silver.dim_segment s
              ON b.seg_id = s.seg_id
            JOIN wh.silver.calendar_daily c_cur
              ON b.date = c_cur.date
            LEFT JOIN wh.silver.calendar_daily c_tgt
              ON COALESCE(b.target_date, b.date) = c_tgt.date
            LEFT JOIN gold.hist_segment_profile_train h0
              ON b.seg_id = h0.seg_id
             AND b.hour = h0.hour
             AND b.minute = h0.minute
             AND c_cur.day_type = h0.day_type
            LEFT JOIN gold.hist_segment_profile_train h60
              ON b.seg_id = h60.seg_id
             AND b.target_hour = h60.hour
             AND b.target_minute = h60.minute
             AND COALESCE(c_tgt.day_type, c_cur.day_type) = h60.day_type
            LEFT JOIN tmp_downstream_15m ds
              ON b.seg_id = ds.seg_id AND b.ts = ds.ts
            LEFT JOIN tmp_district_15m_fe dist
              ON b.district = dist.district AND b.ts = dist.ts
            LEFT JOIN wh.silver.weather_hourly w
              ON b.weather_point = w.point
             AND date_trunc('hour', b.ts) = w.ts
            LEFT JOIN wh.gold.traffy_district_static_train st
              ON b.district = st.district;
            """
        )
        n_ml = con.execute("SELECT COUNT(*) FROM gold.ml_features_15m").fetchone()[0]
        n_cols = con.execute(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_catalog = 'bkk_traffic_features' AND table_schema='gold' AND table_name='ml_features_15m'"
        ).fetchone()[0]

        split_counts = dict(
            con.execute("SELECT split, COUNT(*) FROM gold.ml_features_15m GROUP BY split ORDER BY split").fetchall()
        )
        coverage = con.execute(
            """
            SELECT
                ROUND(AVG(CASE WHEN speed_lag_15m IS NOT NULL THEN 1.0 ELSE 0.0 END) * 100, 2),
                ROUND(AVG(CASE WHEN hist_avg_ratio_t60 IS NOT NULL THEN 1.0 ELSE 0.0 END) * 100, 2),
                ROUND(AVG(CASE WHEN downstream_avg_ratio_t IS NOT NULL THEN 1.0 ELSE 0.0 END) * 100, 2),
                ROUND(AVG(CASE WHEN precip_mm IS NOT NULL THEN 1.0 ELSE 0.0 END) * 100, 2),
                ROUND(AVG(CASE WHEN target_ratio IS NOT NULL THEN 1.0 ELSE 0.0 END) * 100, 2)
            FROM gold.ml_features_15m
            """
        ).fetchone()

        elapsed = round(time.time() - t0, 2)
        report = {
            "run_id": run_id,
            "database_path": FEATURE_DB_PATH.relative_to(path("raw").parents[1]).as_posix(),
            "database_size_mb": round(FEATURE_DB_PATH.stat().st_size / 1e6, 2),
            "elapsed_seconds": elapsed,
            "hist_profile_rows": int(n_hist),
            "ml_features_rows": int(n_ml),
            "ml_features_cols": int(n_cols),
            "split_counts": {k: int(v) for k, v in split_counts.items()},
            "non_null_coverage_pct": {
                "speed_lag_15m": float(coverage[0]),
                "hist_avg_ratio_t60": float(coverage[1]),
                "downstream_avg_ratio_t": float(coverage[2]),
                "precip_mm": float(coverage[3]),
                "target_ratio": float(coverage[4]),
            },
            "feature_groups": FEATURE_GROUPS,
            "pass": n_ml > 20_000_000 and n_cols >= 75,
        }
        write_json(report, path("quality") / f"{run_id}_feature_engineering.json")
        log.info(
            "[FE DONE] gold.ml_features_15m: %s rows x %s cols in %.1fs (DB size: %.1f MB)",
            f"{n_ml:,}",
            n_cols,
            elapsed,
            report["database_size_mb"],
        )
        return report
    finally:
        con.close()


def build_imputed_ml_features(run_id: str = "manual") -> dict:
    """Create `gold.ml_features_15m_imputed` inside `data/bkk_traffic_features.duckdb`.

    Design principles (academically sound, physically plausible, 0% NULLs, Zero Leakage):
    1. Real Ground Truth Only: Keep only rows with genuine `target_ratio IS NOT NULL` (never fabricate targets).
    2. Train-Only Plausible Imputation + Deterministic +/-3% Jitter:
       - Missing lags (`speed_lag_*`, `ratio_lag_*`) are filled from that segment's own Train-only
         historical average for the same time-of-day & day_type (`hist_avg_speed_t`), plus a small
         reproducible hash-based jitter (+/-3%), clipped strictly to [3.0, 120.0] km/h.
       - Missing downstream road features are filled from the district's real-time traffic at the
         exact same timestamp `t` (fallback: current segment speed/ratio).
       - Missing `length_m` is filled with the median segment length across Bangkok.
    3. Full Transparency: Adds `is_lag_15m_imputed` and `is_downstream_imputed` boolean flags.
    """
    t0 = time.time()
    if not FEATURE_DB_PATH.exists():
        raise FileNotFoundError(f"Missing {FEATURE_DB_PATH}. Run build_ml_features first.")

    con = duckdb.connect(str(FEATURE_DB_PATH))
    in_docker = Path("/opt/airflow").exists()
    n_threads = 4 if in_docker else max(4, (os.cpu_count() or 8) - 2)
    con.execute(f"PRAGMA threads={n_threads};")
    con.execute("PRAGMA memory_limit='8GB';")

    try:
        log.info("[IMPUTE] Building gold.ml_features_15m_imputed (0% NULLs, plausible Train-only imputation)...")
        con.execute(
            """
            CREATE OR REPLACE TABLE gold.ml_features_15m_imputed AS
            WITH med_len AS (
                SELECT MEDIAN(length_m)::FLOAT AS med_length_m
                FROM gold.ml_features_15m
                WHERE length_m IS NOT NULL
            ),
            step1 AS (
                SELECT
                    f.*,
                    (f.speed_lag_15m IS NULL) AS is_lag_15m_imputed,
                    (f.downstream_avg_ratio_t IS NULL) AS is_downstream_imputed,
                    -- Deterministic +/-3% jitter factors in [-0.03, +0.03] per (seg_id, ts, lag)
                    ((ABS(HASH(f.seg_id, f.ts, 15)) % 1001) / 1000.0 - 0.5) * 0.06 AS j15,
                    ((ABS(HASH(f.seg_id, f.ts, 30)) % 1001) / 1000.0 - 0.5) * 0.06 AS j30,
                    ((ABS(HASH(f.seg_id, f.ts, 45)) % 1001) / 1000.0 - 0.5) * 0.06 AS j45,
                    ((ABS(HASH(f.seg_id, f.ts, 60)) % 1001) / 1000.0 - 0.5) * 0.06 AS j60,
                    -- Base Train-only historical fallback (never uses valid/test future data)
                    COALESCE(f.hist_avg_speed_t, f.speed_kmh)::FLOAT AS base_hist_spd_t,
                    COALESCE(f.hist_avg_ratio_t, f.ratio)::FLOAT AS base_hist_rat_t,
                    COALESCE(f.hist_pct_congested_t, CASE WHEN f.is_congested THEN 1.0 ELSE 0.0 END)::FLOAT AS base_hist_cong_t,
                    COALESCE(f.hist_avg_speed_t60, f.hist_avg_speed_t, f.speed_kmh)::FLOAT AS base_hist_spd_t60,
                    COALESCE(f.hist_avg_ratio_t60, f.hist_avg_ratio_t, f.ratio)::FLOAT AS base_hist_rat_t60,
                    COALESCE(f.hist_pct_congested_t60, f.hist_pct_congested_t, CASE WHEN f.is_congested THEN 1.0 ELSE 0.0 END)::FLOAT AS base_hist_cong_t60,
                    ml.med_length_m
                FROM gold.ml_features_15m f
                CROSS JOIN med_len ml
                WHERE f.target_ratio IS NOT NULL
                  AND f.target_speed_kmh IS NOT NULL
                  AND f.target_congested IS NOT NULL
                  AND f.new_onset IS NOT NULL
            ),
            step2 AS (
                SELECT
                    s.*,
                    COALESCE(
                        s.speed_lag_15m,
                        ROUND(LEAST(120.0, GREATEST(3.0, s.base_hist_spd_t * (1.0 + s.j15))), 2)
                    )::FLOAT AS imp_speed_lag_15m,
                    COALESCE(
                        s.speed_lag_30m,
                        ROUND(LEAST(120.0, GREATEST(3.0, s.base_hist_spd_t * (1.0 + s.j30))), 2)
                    )::FLOAT AS imp_speed_lag_30m,
                    COALESCE(
                        s.speed_lag_45m,
                        ROUND(LEAST(120.0, GREATEST(3.0, s.base_hist_spd_t * (1.0 + s.j45))), 2)
                    )::FLOAT AS imp_speed_lag_45m,
                    COALESCE(
                        s.speed_lag_60m,
                        ROUND(LEAST(120.0, GREATEST(3.0, s.base_hist_spd_t * (1.0 + s.j60))), 2)
                    )::FLOAT AS imp_speed_lag_60m
                FROM step1 s
            )
            SELECT
                -- ID & Metadata
                seg_id, ts, date, target_ts, split, event_time, available_time,
                -- Targets (100% genuine, never fabricated)
                target_speed_kmh, target_ratio, target_congested, new_onset,
                -- Quality & Imputation Flags
                flag_ratio_gt_1_2, flag_jump_gt_60, flag_incomplete_day,
                is_lag_15m_imputed, is_downstream_imputed,
                -- Group 1: Current Traffic
                speed_kmh, ratio, congestion_level, is_congested, n_obs,
                -- Group 2: Temporal Lags & Rolling (imputed with realistic Train-only baseline +/-3%)
                imp_speed_lag_15m AS speed_lag_15m,
                imp_speed_lag_30m AS speed_lag_30m,
                imp_speed_lag_45m AS speed_lag_45m,
                imp_speed_lag_60m AS speed_lag_60m,
                COALESCE(ratio_lag_15m, ROUND(imp_speed_lag_15m / NULLIF(free_flow_kmh, 0), 4))::FLOAT AS ratio_lag_15m,
                COALESCE(ratio_lag_30m, ROUND(imp_speed_lag_30m / NULLIF(free_flow_kmh, 0), 4))::FLOAT AS ratio_lag_30m,
                COALESCE(ratio_lag_45m, ROUND(imp_speed_lag_45m / NULLIF(free_flow_kmh, 0), 4))::FLOAT AS ratio_lag_45m,
                COALESCE(ratio_lag_60m, ROUND(imp_speed_lag_60m / NULLIF(free_flow_kmh, 0), 4))::FLOAT AS ratio_lag_60m,
                ROUND(speed_kmh - imp_speed_lag_15m, 2)::FLOAT AS speed_diff_15m,
                ROUND(speed_kmh - imp_speed_lag_30m, 2)::FLOAT AS speed_diff_30m,
                ROUND(ratio - COALESCE(ratio_lag_15m, ROUND(imp_speed_lag_15m / NULLIF(free_flow_kmh, 0), 4)), 4)::FLOAT AS ratio_diff_15m,
                speed_roll_mean_1h,
                speed_roll_std_1h,
                ratio_roll_mean_1h,
                congested_slots_last_1h,
                -- Group 3: Historical Profile (Train-Only)
                base_hist_spd_t AS hist_avg_speed_t,
                base_hist_rat_t AS hist_avg_ratio_t,
                base_hist_cong_t AS hist_pct_congested_t,
                base_hist_spd_t60 AS hist_avg_speed_t60,
                base_hist_rat_t60 AS hist_avg_ratio_t60,
                base_hist_cong_t60 AS hist_pct_congested_t60,
                ROUND(speed_kmh - base_hist_spd_t, 2)::FLOAT AS speed_vs_hist_t,
                ROUND(ratio - base_hist_rat_t, 4)::FLOAT AS ratio_vs_hist_t,
                -- Group 4: Spatial / Downstream (fallback: same-timestamp district traffic -> current segment)
                COALESCE(downstream_avg_ratio_t, district_avg_ratio_t, ratio)::FLOAT AS downstream_avg_ratio_t,
                COALESCE(downstream_min_ratio_t, district_avg_ratio_t, ratio)::FLOAT AS downstream_min_ratio_t,
                COALESCE(downstream_avg_speed_t, district_avg_speed_t, speed_kmh)::FLOAT AS downstream_avg_speed_t,
                COALESCE(downstream_pct_congested_t, district_pct_congested_t, CASE WHEN is_congested THEN 1.0 ELSE 0.0 END)::FLOAT AS downstream_pct_congested_t,
                downstream_n_active_t,
                COALESCE(district_avg_ratio_t, ratio)::FLOAT AS district_avg_ratio_t,
                COALESCE(district_pct_congested_t, CASE WHEN is_congested THEN 1.0 ELSE 0.0 END)::FLOAT AS district_pct_congested_t,
                COALESCE(district_avg_speed_t, speed_kmh)::FLOAT AS district_avg_speed_t,
                -- Group 5: Weather
                precip_mm, precip_3h_mm, precip_6h_mm, temp_c,
                rain_bin, is_rainy_gt_1mm, is_rain_x_rush,
                -- Group 6: Time & Calendar
                hour, minute, target_hour, target_minute,
                target_hour_sin, target_hour_cos,
                COALESCE(dow, EXTRACT(DOW FROM target_ts)::TINYINT)::TINYINT AS dow,
                COALESCE(dow_sin, ROUND(SIN(2.0 * PI() * EXTRACT(DOW FROM target_ts) / 7.0), 4))::FLOAT AS dow_sin,
                COALESCE(dow_cos, ROUND(COS(2.0 * PI() * EXTRACT(DOW FROM target_ts) / 7.0), 4))::FLOAT AS dow_cos,
                COALESCE(day_type, 'workday') AS day_type,
                COALESCE(is_weekend, FALSE) AS is_weekend,
                COALESCE(is_friday, FALSE) AS is_friday,
                COALESCE(is_holiday, FALSE) AS is_holiday,
                COALESCE(is_bank_holiday, FALSE) AS is_bank_holiday,
                COALESCE(is_working_day, TRUE) AS is_working_day,
                COALESCE(is_payday, FALSE) AS is_payday,
                COALESCE(is_school_term, TRUE) AS is_school_term,
                COALESCE(is_friday_evening, FALSE) AS is_friday_evening,
                COALESCE(is_payday_evening, FALSE) AS is_payday_evening,
                -- Group 7: Static Road
                free_flow_kmh,
                COALESCE(length_m, med_length_m)::FLOAT AS length_m,
                lat, lon, inbound_outbound, n_downstream, district, weather_point,
                -- Group 8: Traffy Fondue
                train_flood_reports, train_flood_per_sq_km, train_traffic_per_sq_km,
                traffy_flood_last_1h, traffy_flood_last_3h, traffy_flood_last_6h,
                traffy_traffic_last_1h, traffy_traffic_last_3h
            FROM step2;
            """
        )
        n_rows = con.execute("SELECT COUNT(*) FROM gold.ml_features_15m_imputed").fetchone()[0]
        cols = [
            r[0]
            for r in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_catalog='bkk_traffic_features' AND table_schema='gold' AND table_name='ml_features_15m_imputed'"
            ).fetchall()
        ]
        null_exprs = " + ".join(f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END)' for c in cols)
        total_nulls = int(con.execute(f"SELECT {null_exprs} FROM gold.ml_features_15m_imputed").fetchone()[0])

        elapsed = round(time.time() - t0, 2)
        report = {
            "run_id": run_id,
            "table": "gold.ml_features_15m_imputed",
            "rows": int(n_rows),
            "cols": len(cols),
            "total_null_cells": total_nulls,
            "elapsed_seconds": elapsed,
            "database_size_mb": round(FEATURE_DB_PATH.stat().st_size / 1e6, 2),
            "pass": n_rows > 15_000_000 and total_nulls == 0,
        }
        write_json(report, path("quality") / f"{run_id}_ml_features_imputed.json")
        log.info(
            "[IMPUTE DONE] gold.ml_features_15m_imputed: %s rows x %s cols, total_nulls=%s in %.1fs",
            f"{n_rows:,}",
            len(cols),
            total_nulls,
            elapsed,
        )
        return report
    finally:
        con.close()


if __name__ == "__main__":
    build_ml_features()
    build_imputed_ml_features()
