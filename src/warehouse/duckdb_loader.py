"""Build and populate the DuckDB Medallion Warehouse (`data/bkk_traffic.duckdb`: Bronze -> Silver -> Gold).

Medallion Architecture inside `data/bkk_traffic.duckdb`:
=============================================================================
1. BRONZE SCHEMA (`bronze.*`) - Raw Landing & Lineage Metadata
   - `bronze.raw_manifest_log`         : บัญชีไฟล์ข้อมูลดิบและรหัส SHA-256 checksum
   - `bronze.itic_bkk_daily_raw`       : View เชื่อมไฟล์ CSV รายวันทั้ง 122 วัน (ระดับ 10 นาที / sub-offset)

2. SILVER SCHEMA (`silver.*`) - Cleaned, Validated, Deduplicated, 15-Minute Standardized
   - `silver.dim_district`             : 50 เขต กทม. (แก้คำสะกด ราษฎร์บูรณะ + ผูกโซนวัดฝน)
   - `silver.itic_location_table`      : จุดอ้างอิงถนน iTIC ใน กทม. (4,174 จุด)
   - `silver.dim_segment`              : ช่วงถนนที่มีทิศทาง (`seg_id = from_loc_to_loc`) + พิกัด + ขาเข้า/ออก +
                                         กราฟถนนข้างหน้า (`downstream_seg_ids`) + `free_flow_kmh` (P90 ช่วง Train)
   - `silver.weather_hourly`           : ฝนและอุณหภูมิรายชั่วโมง 3 โซน + ฝนสะสม 3/6 ชม. + `rain_bin`
   - `silver.calendar_daily`           : ปฏิทิน 122 วัน (วันหยุดราชการ/ธนาคาร, วันศุกร์, เงินเดือนออก 3 รอบ)
   - `silver.traffy_district_15m`      : ยอดร้องเรียน Traffy Fondue ยุบรวมราย 15 นาทีต่อเขต (ตัด PII แล้ว)
   - `silver.traffy_district_hourly`   : ยอดร้องเรียน Traffy Fondue รายชั่วโมงต่อเขต
   - `silver.traffy_district_daily`    : ยอดร้องเรียน Traffy Fondue รายวันต่อเขต
   - `silver.itic_segment_15m`         : ความเร็วรถ iTIC ยุบเป็น Median ราย 15 นาทีต่อ `seg_id` (กรอง 0-150 km/h)
   - `silver.quality_completeness_seg_day` : ผลตรวจความครบถ้วนราย `seg_id x date` (>=80% ทั้งวัน, >=90% ช่วงพีค)

3. GOLD SCHEMA (`gold.*`) - Business/Analytics Ready, Merged 15-Minute & Leak-Free for EDA & ML
   - `gold.traffy_district_static_train` : ความเสี่ยงน้ำท่วม/จราจรสะสมระดับเขต (คำนวณเฉพาะช่วง Train เท่านั้น!)
   - `gold.district_hourly_panel`      : ตารางรวม 50 เขต x 2,928 ชั่วโมง (146,400 แถว)
   - `gold.merged_district_15m`        : ตาราง Merge สำเร็จรูป 50 เขต x 11,712 ช่องเวลา 15 นาที (585,600 แถว)
                                         รวม iTIC + Traffy 15 นาที + ฝน + ปฏิทิน + ค่าสะสม Train (ทำ EDA ง่ายที่สุด!)
   - `gold.merged_segment_15m`         : ตาราง Merge สำเร็จรูป ระดับ `ช่วงถนน (seg_id) x 15 นาที`
                                         พร้อม `ratio`, `is_congested`, `target_ratio` (ที่ t+60 นาที), `new_onset`,
                                         Quality Flags และฟีเจอร์จากทุก Dataset สำหรับทำ EDA รายถนนและเทรน LightGBM
=============================================================================
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import duckdb
import pandas as pd

from src.common import CFG, TZ, get_logger, path, write_json
from src.transform.itic_aggregate import build_dim_segment

log = get_logger("warehouse.duckdb")

DB_PATH = path("raw").parent / "bkk_traffic.duckdb"


def build_traffy_15m() -> pd.DataFrame:
    """Aggregate clean Bangkok Traffy tickets to 15-minute intervals per district."""
    df = pd.read_parquet(path("staging") / "traffy_bkk_study_clean.parquet")
    df["ts_15m"] = df["ts"].dt.floor("15min").dt.tz_localize(None)
    agg = (
        df.groupby(["district", "ts_15m"], as_index=False)
        .agg(
            traffy_total_15m=("ticket_id", "count"),
            traffy_traffic_15m=("is_traffic", "sum"),
            traffy_flood_15m=("is_flood", "sum"),
            traffy_road_15m=("is_road", "sum"),
            traffy_unspec_15m=("is_unspecified", "sum"),
        )
        .rename(columns={"ts_15m": "ts"})
        .sort_values(["district", "ts"])
        .reset_index(drop=True)
    )
    return agg


def build_duckdb_warehouse(run_id: str = "manual", force_rebuild: bool = False) -> dict:
    t0 = time.time()
    proc = path("processed")
    csv_root = proc / "csv"
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Idempotent Fast-Path: if data/bkk_traffic.duckdb is already fully populated and valid,
    # verify its Bronze/Silver/Gold tables without re-scanning 20.3 GB of CSVs (critical inside Docker WSL2 mount).
    if not force_rebuild and DB_PATH.exists() and DB_PATH.stat().st_size > 100_000_000:
        con_ro = None
        try:
            con_ro = duckdb.connect(str(DB_PATH), read_only=True)
            n_seg = con_ro.execute("SELECT COUNT(*) FROM silver.itic_segment_15m").fetchone()[0]
            n_dist = con_ro.execute("SELECT COUNT(*) FROM gold.merged_district_15m").fetchone()[0]
            if n_seg > 20_000_000 and n_dist == 585600:
                tables_info = {}
                for full_name in [
                    "bronze.raw_manifest_log",
                    "silver.dim_district",
                    "silver.itic_location_table",
                    "silver.dim_segment",
                    "silver.weather_hourly",
                    "silver.calendar_daily",
                    "silver.traffy_district_15m",
                    "silver.traffy_district_hourly",
                    "silver.traffy_district_daily",
                    "silver.itic_segment_15m",
                    "silver.quality_completeness_seg_day",
                    "gold.traffy_district_static_train",
                    "gold.district_hourly_panel",
                    "gold.itic_seg15_targets",
                    "gold.merged_district_15m",
                    "gold.merged_segment_15m",
                ]:
                    tables_info[full_name] = int(con_ro.execute(f"SELECT COUNT(*) FROM {full_name}").fetchone()[0])
                stats = con_ro.execute(
                    """
                    SELECT
                        MIN(speed_kmh), MAX(speed_kmh), ROUND(AVG(speed_kmh), 2),
                        ROUND(AVG(CASE WHEN is_congested THEN 1.0 ELSE 0.0 END) * 100, 2),
                        SUM(CASE WHEN new_onset THEN 1 ELSE 0 END),
                        SUM(CASE WHEN flag_ratio_gt_1_2 THEN 1 ELSE 0 END),
                        SUM(CASE WHEN flag_jump_gt_60 THEN 1 ELSE 0 END)
                    FROM gold.itic_seg15_targets
                    """
                ).fetchone()
                report = {
                    "database_path": DB_PATH.relative_to(path("raw").parents[1]).as_posix(),
                    "database_size_mb": round(DB_PATH.stat().st_size / 1e6, 2),
                    "elapsed_seconds": round(time.time() - t0, 2),
                    "cached_idempotent": True,
                    "medallion_tables": tables_info,
                    "gold_itic_seg15_summary": {
                        "min_speed_kmh": float(stats[0]),
                        "max_speed_kmh": float(stats[1]),
                        "avg_speed_kmh": float(stats[2]),
                        "pct_congested_lt_0_5": float(stats[3]),
                        "new_onset_events": int(stats[4]),
                        "flag_ratio_gt_1_2_count": int(stats[5]),
                        "flag_jump_gt_60_count": int(stats[6]),
                    },
                    "pass": True,
                }
                write_json(report, path("quality") / f"{run_id}_duckdb_warehouse.json")
                log.info("DuckDB Medallion Warehouse already up-to-date (verified 16 tables in %.2fs)", report["elapsed_seconds"])
                return report
        except Exception as e:
            log.info("Existing DuckDB needs rebuild (%s), proceeding with full build...", e)
        finally:
            if con_ro is not None:
                con_ro.close()

    log.info("Connecting to DuckDB Medallion Warehouse at %s", DB_PATH)
    con = duckdb.connect(str(DB_PATH))
    in_docker = Path("/opt/airflow").exists()
    n_threads = 4 if in_docker else max(4, (os.cpu_count() or 8) - 2)
    con.execute(f"PRAGMA threads={n_threads};")

    # Create Medallion Schemas: bronze -> silver -> gold
    for sch in ("bronze", "silver", "gold"):
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {sch};")

    # -----------------------------------------------------------------------
    # LAYER 1: BRONZE (Raw Manifest & Raw Daily iTIC View)
    # -----------------------------------------------------------------------
    log.info("[BRONZE] Registering raw manifest log and daily iTIC raw view...")
    manifest_path = path("manifest").as_posix()
    con.execute(
        f"CREATE OR REPLACE TABLE bronze.raw_manifest_log AS SELECT * FROM read_json_auto('{manifest_path}');"
    )
    itic_glob = (csv_root / "iTic Traffic" / "*" / "itic_bkk_clean_*.csv").as_posix()
    con.execute(
        f"""
        CREATE OR REPLACE VIEW bronze.itic_bkk_daily_raw AS
        SELECT * FROM read_csv(
            '{itic_glob}',
            header=true,
            columns={{
                'snapshot_time': 'VARCHAR',
                'event_time': 'VARCHAR',
                'from_loc': 'INTEGER',
                'from_offset': 'VARCHAR',
                'from_dir': 'VARCHAR',
                'to_loc': 'INTEGER',
                'to_offset': 'VARCHAR',
                'to_dir': 'VARCHAR',
                'link_speed_kmh': 'DOUBLE',
                'congestion_level': 'TINYINT',
                'adm2_pcode': 'VARCHAR',
                'adm2_name': 'VARCHAR'
            }}
        );
        """
    )

    # -----------------------------------------------------------------------
    # LAYER 2: SILVER (Cleaned, Standardized, 15-Minute Aggregated & Quality Gated)
    # -----------------------------------------------------------------------
    log.info("[SILVER 1/3] Loading cleaned dimension, weather, calendar, and Traffy tables...")
    con.execute(
        f"CREATE OR REPLACE TABLE silver.dim_district AS SELECT * FROM read_parquet('{(proc / 'dim_district.parquet').as_posix()}');"
    )
    con.execute(
        f"CREATE OR REPLACE TABLE silver.calendar_daily AS SELECT * FROM read_parquet('{(proc / 'calendar_daily.parquet').as_posix()}');"
    )
    con.execute(
        f"CREATE OR REPLACE TABLE silver.traffy_district_daily AS SELECT * FROM read_parquet('{(proc / 'traffy_district_daily.parquet').as_posix()}');"
    )

    w_df = pd.read_parquet(proc / "weather_hourly.parquet")
    w_df["ts"] = w_df["ts"].dt.tz_convert(TZ).dt.tz_localize(None)
    w_df["event_start"] = w_df["event_start"].dt.tz_convert(TZ).dt.tz_localize(None)
    w_df["available_time"] = w_df["available_time"].dt.tz_convert(TZ).dt.tz_localize(None)
    con.register("w_df_view", w_df)
    con.execute("CREATE OR REPLACE TABLE silver.weather_hourly AS SELECT * FROM w_df_view;")

    tr_h = pd.read_parquet(proc / "traffy_district_hourly.parquet")
    tr_h["ts"] = tr_h["ts"].dt.tz_convert(TZ).dt.tz_localize(None)
    tr_h["event_start"] = tr_h["event_start"].dt.tz_convert(TZ).dt.tz_localize(None)
    tr_h["available_time"] = tr_h["available_time"].dt.tz_convert(TZ).dt.tz_localize(None)
    con.register("tr_h_view", tr_h)
    con.execute("CREATE OR REPLACE TABLE silver.traffy_district_hourly AS SELECT * FROM tr_h_view;")

    tr_15m = build_traffy_15m()
    con.register("tr_15m_view", tr_15m)
    con.execute("CREATE OR REPLACE TABLE silver.traffy_district_15m AS SELECT * FROM tr_15m_view;")
    tr_15m.to_parquet(proc / "traffy_district_15m.parquet", index=False)
    tr_15m.to_csv(csv_root / "traffy" / "traffy_district_15m.csv", index=False, encoding="utf-8-sig")

    loc_csv = (csv_root / "itic_location_table" / "itic_location_table_bkk_clean.csv").as_posix()
    con.execute(f"CREATE OR REPLACE TABLE silver.itic_location_table AS SELECT * FROM read_csv_auto('{loc_csv}');")

    log.info("[SILVER 2/3] Aggregating 122 daily iTIC files (~200M rows) -> silver.itic_segment_15m...")
    t_agg = time.time()
    seg15_dir = proc / "itic_seg15"
    cached_pqs = sorted(seg15_dir.glob("merged_segment_15m_2022-*.parquet")) if seg15_dir.exists() else []
    if len(cached_pqs) == 4 and not force_rebuild:
        pq_glob = (seg15_dir / "merged_segment_15m_2022-*.parquet").as_posix()
        con.execute(
            f"""
            CREATE OR REPLACE TABLE silver.itic_segment_15m AS
            SELECT seg_id, from_loc, to_loc, ts, speed_kmh, congestion_level, n_obs
            FROM read_parquet('{pq_glob}');
            """
        )
    else:
        con.execute(
            """
            CREATE OR REPLACE TABLE silver.itic_segment_15m AS
            SELECT
                CAST(from_loc AS VARCHAR) || '_' || CAST(to_loc AS VARCHAR) AS seg_id,
                CAST(from_loc AS INTEGER) AS from_loc,
                CAST(to_loc AS INTEGER) AS to_loc,
                time_bucket(INTERVAL '15 minutes', CAST(snapshot_time AS TIMESTAMP)) AS ts,
                ROUND(MEDIAN(CAST(link_speed_kmh AS DOUBLE)), 2) AS speed_kmh,
                CAST(MAX(congestion_level) AS TINYINT) AS congestion_level,
                CAST(COUNT(*) AS SMALLINT) AS n_obs
            FROM bronze.itic_bkk_daily_raw
            WHERE link_speed_kmh > 0 AND link_speed_kmh <= 150
            GROUP BY from_loc, to_loc, time_bucket(INTERVAL '15 minutes', CAST(snapshot_time AS TIMESTAMP));
            """
        )
    n_15m = con.execute("SELECT COUNT(*) FROM silver.itic_segment_15m").fetchone()[0]
    log.info("[SILVER 2/3] Created silver.itic_segment_15m: %s rows in %.1fs", f"{n_15m:,}", time.time() - t_agg)

    log.info("[SILVER 3/3] Building silver.dim_segment and silver.quality_completeness_seg_day...")
    dim_seg_pq = proc / "dim_segment.parquet"
    comp_pq = proc / "quality_completeness_seg_day.parquet"
    train_start, train_end = CFG["split"]["train"]

    if dim_seg_pq.exists() and comp_pq.exists() and not force_rebuild:
        con.execute(f"CREATE OR REPLACE TABLE silver.dim_segment AS SELECT * FROM read_parquet('{dim_seg_pq.as_posix()}');")
        con.execute(
            f"CREATE OR REPLACE TABLE silver.quality_completeness_seg_day AS SELECT * FROM read_parquet('{comp_pq.as_posix()}');"
        )
    else:
        seg_pairs = con.execute("SELECT DISTINCT seg_id, from_loc, to_loc FROM silver.itic_segment_15m").df()
        dim_seg = build_dim_segment(seg_pairs)
        ff_df = con.execute(
            f"""
            SELECT
                seg_id,
                ROUND(QUANTILE_CONT(speed_kmh, 0.90), 2) AS free_flow_kmh,
                COUNT(*) AS train_obs_slots
            FROM silver.itic_segment_15m
            WHERE strftime(ts, '%Y-%m-%d') BETWEEN '{train_start}' AND '{train_end}'
            GROUP BY seg_id
            """
        ).df()
        dim_seg = dim_seg.merge(ff_df, on="seg_id", how="left")
        global_ff = float(dim_seg["free_flow_kmh"].median())
        dim_seg["free_flow_kmh"] = dim_seg["free_flow_kmh"].fillna(global_ff)
        dim_seg["train_obs_slots"] = dim_seg["train_obs_slots"].fillna(0).astype(int)

        con.register("dim_seg_view", dim_seg)
        con.execute("CREATE OR REPLACE TABLE silver.dim_segment AS SELECT * FROM dim_seg_view;")
        dim_seg.to_parquet(dim_seg_pq, index=False)
        dim_seg.to_csv(csv_root / "itic_location_table" / "dim_segment.csv", index=False, encoding="utf-8-sig")

        con.execute(
            """
            CREATE OR REPLACE TABLE silver.quality_completeness_seg_day AS
            SELECT
                seg_id,
                strftime(ts, '%Y-%m-%d') AS date,
                COUNT(*) AS n_slots,
                SUM(CASE WHEN EXTRACT(HOUR FROM ts) BETWEEN 6 AND 21 THEN 1 ELSE 0 END) AS n_peak_slots,
                ROUND(COUNT(*) / 96.0, 4) AS completeness_all,
                ROUND(SUM(CASE WHEN EXTRACT(HOUR FROM ts) BETWEEN 6 AND 21 THEN 1 ELSE 0 END) / 64.0, 4) AS completeness_peak,
                (COUNT(*) < 77 OR SUM(CASE WHEN EXTRACT(HOUR FROM ts) BETWEEN 6 AND 21 THEN 1 ELSE 0 END) < 58) AS flag_incomplete_day
            FROM silver.itic_segment_15m
            GROUP BY seg_id, strftime(ts, '%Y-%m-%d')
            ORDER BY seg_id, date;
            """
        )
        comp_df = con.execute("SELECT * FROM silver.quality_completeness_seg_day").df()
        comp_df.to_parquet(comp_pq, index=False)
        comp_df.to_csv(csv_root / "itic_location_table" / "quality_completeness_seg_day.csv", index=False, encoding="utf-8-sig")

    # -----------------------------------------------------------------------
    # LAYER 3: GOLD (Merged 15-Minute Tables, Targets, Leak-Free Features for EDA & ML)
    # -----------------------------------------------------------------------
    log.info("[GOLD 1/3] Loading gold.traffy_district_static_train and gold.district_hourly_panel...")
    con.execute(
        f"CREATE OR REPLACE TABLE gold.traffy_district_static_train AS SELECT * FROM read_parquet('{(proc / 'traffy_district_static_train.parquet').as_posix()}');"
    )
    con.execute(
        f"CREATE OR REPLACE TABLE gold.district_hourly_panel AS SELECT * FROM read_parquet('{(proc / 'district_hourly_panel.parquet').as_posix()}');"
    )

    valid_start, valid_end = CFG["split"]["valid"]
    test_start, test_end = CFG["split"]["test"]

    log.info("[GOLD 2/3] Building gold.itic_seg15_targets and gold.merged_district_15m (585,600 rows)...")
    dist15_pq = proc / "merged_district_15m.parquet"
    if len(cached_pqs) == 4 and dist15_pq.exists() and not force_rebuild:
        pq_glob = (seg15_dir / "merged_segment_15m_2022-*.parquet").as_posix()
        con.execute(
            f"""
            CREATE OR REPLACE TABLE gold.itic_seg15_targets AS
            SELECT
                seg_id, from_loc, to_loc, ts, date, hour, minute,
                speed_kmh, congestion_level, n_obs, free_flow_kmh, ratio, is_congested,
                target_ts, target_speed_kmh, target_ratio, target_congested, new_onset,
                prev_15m_speed_kmh, flag_ratio_gt_1_2, flag_jump_gt_60,
                district, weather_point, inbound_outbound, split, event_time, available_time,
                flag_incomplete_day
            FROM read_parquet('{pq_glob}');
            """
        )
        con.execute(
            f"CREATE OR REPLACE TABLE gold.merged_district_15m AS SELECT * FROM read_parquet('{dist15_pq.as_posix()}');"
        )
    else:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE gold.itic_seg15_targets AS
            WITH base AS (
                SELECT
                    cur.seg_id,
                    cur.from_loc,
                    cur.to_loc,
                    cur.ts,
                    strftime(cur.ts, '%Y-%m-%d') AS date,
                    CAST(EXTRACT(HOUR FROM cur.ts) AS TINYINT) AS hour,
                    CAST(EXTRACT(MINUTE FROM cur.ts) AS TINYINT) AS minute,
                    cur.speed_kmh,
                    cur.congestion_level,
                    cur.n_obs,
                    seg.free_flow_kmh,
                    ROUND(cur.speed_kmh / NULLIF(seg.free_flow_kmh, 0), 4) AS ratio,
                    (cur.speed_kmh / NULLIF(seg.free_flow_kmh, 0)) < 0.5 AS is_congested,
                    cur.ts + INTERVAL '60 minutes' AS target_ts,
                    fut.speed_kmh AS target_speed_kmh,
                    ROUND(fut.speed_kmh / NULLIF(seg.free_flow_kmh, 0), 4) AS target_ratio,
                    (fut.speed_kmh / NULLIF(seg.free_flow_kmh, 0)) < 0.5 AS target_congested,
                    (NOT ((cur.speed_kmh / NULLIF(seg.free_flow_kmh, 0)) < 0.5))
                        AND ((fut.speed_kmh / NULLIF(seg.free_flow_kmh, 0)) < 0.5) AS new_onset,
                    prev.speed_kmh AS prev_15m_speed_kmh,
                    (cur.speed_kmh / NULLIF(seg.free_flow_kmh, 0)) > 1.2 AS flag_ratio_gt_1_2,
                    COALESCE(ABS(cur.speed_kmh - prev.speed_kmh) > 60.0, FALSE) AS flag_jump_gt_60,
                    seg.district,
                    seg.weather_point,
                    seg.inbound_outbound,
                    CASE
                        WHEN strftime(cur.ts, '%Y-%m-%d') BETWEEN '{train_start}' AND '{train_end}' THEN 'train'
                        WHEN strftime(cur.ts, '%Y-%m-%d') BETWEEN '{valid_start}' AND '{valid_end}' THEN 'valid'
                        WHEN strftime(cur.ts, '%Y-%m-%d') BETWEEN '{test_start}' AND '{test_end}' THEN 'test'
                        ELSE 'other'
                    END AS split,
                    cur.ts AS event_time,
                    cur.ts + INTERVAL '15 minutes' AS available_time
                FROM silver.itic_segment_15m cur
                JOIN silver.dim_segment seg ON cur.seg_id = seg.seg_id
                LEFT JOIN silver.itic_segment_15m fut
                    ON cur.seg_id = fut.seg_id AND fut.ts = cur.ts + INTERVAL '60 minutes'
                LEFT JOIN silver.itic_segment_15m prev
                    ON cur.seg_id = prev.seg_id AND prev.ts = cur.ts - INTERVAL '15 minutes'
            )
            SELECT
                b.*,
                COALESCE(c.flag_incomplete_day, TRUE) AS flag_incomplete_day
            FROM base b
            LEFT JOIN silver.quality_completeness_seg_day c
                ON b.seg_id = c.seg_id AND b.date = c.date
            ORDER BY b.ts, b.seg_id;
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TABLE gold.merged_district_15m AS
            WITH time_slots AS (
                SELECT UNNEST(generate_series(
                    TIMESTAMP '2022-06-01 00:00:00',
                    TIMESTAMP '2022-09-30 23:45:00',
                    INTERVAL '15 minutes'
                )) AS ts
            ),
            grid AS (
                SELECT
                    d.dcode,
                    d.district,
                    d.district_en,
                    d.area_sq_km,
                    d.weather_point,
                    t.ts,
                    strftime(t.ts, '%Y-%m-%d') AS date,
                    CAST(EXTRACT(HOUR FROM t.ts) AS TINYINT) AS hour,
                    CAST(EXTRACT(MINUTE FROM t.ts) AS TINYINT) AS minute,
                    date_trunc('hour', t.ts) AS hour_ts
                FROM silver.dim_district d
                CROSS JOIN time_slots t
            ),
            itic_dist_15m AS (
                SELECT
                    district,
                    ts,
                    COUNT(*) AS itic_active_segments,
                    SUM(n_obs) AS itic_n_obs,
                    ROUND(AVG(speed_kmh), 2) AS itic_avg_speed_kmh,
                    ROUND(MEDIAN(speed_kmh), 2) AS itic_median_speed_kmh,
                    ROUND(QUANTILE_CONT(speed_kmh, 0.10), 2) AS itic_p10_speed_kmh,
                    ROUND(AVG(ratio), 4) AS itic_avg_ratio,
                    ROUND(MEDIAN(ratio), 4) AS itic_median_ratio,
                    SUM(CASE WHEN is_congested THEN 1 ELSE 0 END) AS itic_congested_segments,
                    ROUND(AVG(CASE WHEN is_congested THEN 1.0 ELSE 0.0 END), 4) AS itic_pct_congested,
                    ROUND(AVG(CASE WHEN congestion_level = 3 THEN 1.0 ELSE 0.0 END), 4) AS itic_pct_level3_red,
                    SUM(CASE WHEN new_onset THEN 1 ELSE 0 END) AS itic_new_onset_segments
                FROM gold.itic_seg15_targets
                GROUP BY district, ts
            )
            SELECT
                g.dcode,
                g.district,
                g.district_en,
                g.area_sq_km,
                g.weather_point,
                g.ts,
                g.date,
                g.hour,
                g.minute,
                (g.hour BETWEEN 7 AND 9) AS is_morning_rush,
                (g.hour BETWEEN 16 AND 19) AS is_evening_rush,
                ((g.hour BETWEEN 7 AND 9) OR (g.hour BETWEEN 16 AND 19)) AS is_rush_hour,
                g.date IN ('2022-07-21', '2022-07-20', '2022-09-11') AS is_top_flood_day,
                COALESCE(i.itic_active_segments, 0) AS itic_active_segments,
                COALESCE(i.itic_n_obs, 0) AS itic_n_obs,
                i.itic_avg_speed_kmh,
                i.itic_median_speed_kmh,
                i.itic_p10_speed_kmh,
                i.itic_avg_ratio,
                i.itic_median_ratio,
                COALESCE(i.itic_congested_segments, 0) AS itic_congested_segments,
                COALESCE(i.itic_pct_congested, 0.0) AS itic_pct_congested,
                COALESCE(i.itic_pct_level3_red, 0.0) AS itic_pct_level3_red,
                COALESCE(i.itic_new_onset_segments, 0) AS itic_new_onset_segments,
                COALESCE(tr.traffy_total_15m, 0) AS traffy_total_15m,
                COALESCE(tr.traffy_traffic_15m, 0) AS traffy_traffic_15m,
                COALESCE(tr.traffy_flood_15m, 0) AS traffy_flood_15m,
                COALESCE(tr.traffy_road_15m, 0) AS traffy_road_15m,
                st.train_total_reports,
                st.train_traffic_reports,
                st.train_flood_reports,
                st.train_road_reports,
                st.train_flood_per_day,
                st.train_flood_per_sq_km,
                st.train_traffic_per_sq_km,
                w.precip_mm,
                w.temp_c,
                w.precip_3h_mm,
                w.precip_6h_mm,
                w.rain_bin,
                w.is_rainy_gt_1mm,
                cal.dow,
                cal.day_type,
                cal.is_weekend,
                cal.is_friday,
                cal.is_holiday,
                cal.is_bank_holiday,
                cal.is_working_day,
                cal.is_payday,
                cal.is_school_term,
                CASE
                    WHEN g.date BETWEEN '{train_start}' AND '{train_end}' THEN 'train'
                    WHEN g.date BETWEEN '{valid_start}' AND '{valid_end}' THEN 'valid'
                    WHEN g.date BETWEEN '{test_start}' AND '{test_end}' THEN 'test'
                    ELSE 'other'
                END AS split
            FROM grid g
            LEFT JOIN itic_dist_15m i ON g.district = i.district AND g.ts = i.ts
            LEFT JOIN silver.traffy_district_15m tr ON g.district = tr.district AND g.ts = tr.ts
            LEFT JOIN gold.traffy_district_static_train st ON g.district = st.district
            LEFT JOIN silver.weather_hourly w ON g.weather_point = w.point AND g.hour_ts = w.ts
            LEFT JOIN silver.calendar_daily cal ON g.date = cal.date
            ORDER BY g.dcode, g.ts;
            """
        )

        out_dist_pq = (proc / "merged_district_15m.parquet").as_posix()
        out_dist_csv = (csv_root / "merge-traffy-calendar-open" / "merged_district_15m.csv").as_posix()
        con.execute(f"COPY gold.merged_district_15m TO '{out_dist_pq}' (FORMAT PARQUET);")
        con.execute(f"COPY gold.merged_district_15m TO '{out_dist_csv}' (HEADER, DELIMITER ',');")

    log.info("[GOLD 3/3] Creating gold.merged_segment_15m view and exporting monthly Parquet files...")
    con.execute(
        """
        CREATE OR REPLACE VIEW gold.merged_segment_15m AS
        SELECT
            s.seg_id,
            s.from_loc,
            s.to_loc,
            s.ts,
            s.date,
            s.hour,
            s.minute,
            s.speed_kmh,
            s.congestion_level,
            s.n_obs,
            s.free_flow_kmh,
            s.ratio,
            s.is_congested,
            s.target_ts,
            s.target_speed_kmh,
            s.target_ratio,
            s.target_congested,
            s.new_onset,
            s.prev_15m_speed_kmh,
            s.flag_ratio_gt_1_2,
            s.flag_jump_gt_60,
            s.flag_incomplete_day,
            s.district,
            s.weather_point,
            s.inbound_outbound,
            dim.road_name_th,
            dim.road_name_en,
            dim.lat,
            dim.lon,
            dim.length_m,
            dim.n_downstream,
            dim.downstream_match_type,
            dim.downstream_seg_ids,
            w.precip_mm,
            w.temp_c,
            w.precip_3h_mm,
            w.precip_6h_mm,
            w.rain_bin,
            w.is_rainy_gt_1mm,
            COALESCE(tr.traffy_total_15m, 0) AS traffy_total_15m,
            COALESCE(tr.traffy_traffic_15m, 0) AS traffy_traffic_15m,
            COALESCE(tr.traffy_flood_15m, 0) AS traffy_flood_15m,
            COALESCE(tr.traffy_road_15m, 0) AS traffy_road_15m,
            st.train_flood_reports,
            st.train_flood_per_sq_km,
            st.train_traffic_per_sq_km,
            cal.dow,
            cal.day_type,
            cal.is_weekend,
            cal.is_friday,
            cal.is_holiday,
            cal.is_bank_holiday,
            cal.is_working_day,
            cal.is_payday,
            cal.is_school_term,
            s.split,
            s.event_time,
            s.available_time
        FROM gold.itic_seg15_targets s
        JOIN silver.dim_segment dim ON s.seg_id = dim.seg_id
        LEFT JOIN silver.weather_hourly w ON s.weather_point = w.point AND date_trunc('hour', s.ts) = w.ts
        LEFT JOIN silver.traffy_district_15m tr ON s.district = tr.district AND s.ts = tr.ts
        LEFT JOIN gold.traffy_district_static_train st ON s.district = st.district
        LEFT JOIN silver.calendar_daily cal ON s.date = cal.date;
        """
    )

    # Create top-level convenience views in default `main` schema so queries work with or without schema prefix
    for sch, tbl in [
        ("silver", "dim_district"),
        ("silver", "itic_location_table"),
        ("silver", "dim_segment"),
        ("silver", "weather_hourly"),
        ("silver", "calendar_daily"),
        ("silver", "traffy_district_15m"),
        ("silver", "traffy_district_hourly"),
        ("silver", "traffy_district_daily"),
        ("silver", "itic_segment_15m"),
        ("silver", "quality_completeness_seg_day"),
        ("gold", "traffy_district_static_train"),
        ("gold", "district_hourly_panel"),
        ("gold", "itic_seg15_targets"),
        ("gold", "merged_district_15m"),
        ("gold", "merged_segment_15m"),
    ]:
        con.execute(f"CREATE OR REPLACE VIEW main.{tbl} AS SELECT * FROM {sch}.{tbl};")

    seg15_dir = proc / "itic_seg15"
    seg15_dir.mkdir(parents=True, exist_ok=True)
    if len(cached_pqs) < 4 or force_rebuild:
        for mo in ["2022-06", "2022-07", "2022-08", "2022-09"]:
            out_mo_pq = (seg15_dir / f"merged_segment_15m_{mo}.parquet").as_posix()
            con.execute(
                f"COPY (SELECT * FROM gold.merged_segment_15m WHERE starts_with(date, '{mo}')) TO '{out_mo_pq}' (FORMAT PARQUET);"
            )

    tables_info = {}
    for full_name in [
        "bronze.raw_manifest_log",
        "silver.dim_district",
        "silver.itic_location_table",
        "silver.dim_segment",
        "silver.weather_hourly",
        "silver.calendar_daily",
        "silver.traffy_district_15m",
        "silver.traffy_district_hourly",
        "silver.traffy_district_daily",
        "silver.itic_segment_15m",
        "silver.quality_completeness_seg_day",
        "gold.traffy_district_static_train",
        "gold.district_hourly_panel",
        "gold.itic_seg15_targets",
        "gold.merged_district_15m",
        "gold.merged_segment_15m",
    ]:
        cnt = con.execute(f"SELECT COUNT(*) FROM {full_name}").fetchone()[0]
        tables_info[full_name] = int(cnt)

    stats = con.execute(
        """
        SELECT
            MIN(speed_kmh) AS min_speed,
            MAX(speed_kmh) AS max_speed,
            ROUND(AVG(speed_kmh), 2) AS avg_speed,
            ROUND(AVG(CASE WHEN is_congested THEN 1.0 ELSE 0.0 END) * 100, 2) AS pct_congested,
            SUM(CASE WHEN new_onset THEN 1 ELSE 0 END) AS new_onset_count,
            SUM(CASE WHEN flag_ratio_gt_1_2 THEN 1 ELSE 0 END) AS flag_ratio_gt_1_2_count,
            SUM(CASE WHEN flag_jump_gt_60 THEN 1 ELSE 0 END) AS flag_jump_gt_60_count
        FROM gold.itic_seg15_targets
        """
    ).fetchone()

    con.close()

    report = {
        "database_path": DB_PATH.relative_to(path("raw").parents[1]).as_posix(),
        "database_size_mb": round(DB_PATH.stat().st_size / 1e6, 2),
        "elapsed_seconds": round(time.time() - t0, 1),
        "medallion_tables": tables_info,
        "gold_itic_seg15_summary": {
            "min_speed_kmh": float(stats[0]),
            "max_speed_kmh": float(stats[1]),
            "avg_speed_kmh": float(stats[2]),
            "pct_congested_lt_0_5": float(stats[3]),
            "new_onset_events": int(stats[4]),
            "flag_ratio_gt_1_2_count": int(stats[5]),
            "flag_jump_gt_60_count": int(stats[6]),
        },
        "pass": (
            tables_info["gold.merged_district_15m"] == 585600
            and tables_info["gold.itic_seg15_targets"] > 1_000_000
            and float(stats[0]) > 0
            and float(stats[1]) <= 150.0
        ),
    }
    write_json(report, path("quality") / f"{run_id}_duckdb_warehouse.json")
    log.info("DuckDB Medallion Warehouse built in %.1fs: %s", report["elapsed_seconds"], report)
    return report


if __name__ == "__main__":
    build_duckdb_warehouse()
