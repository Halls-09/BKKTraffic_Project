"""Impute remaining structural GPS-probe NULLs in the main Medallion Warehouse (`data/bkk_traffic.duckdb`).

Zero-Leakage, Physically Plausible, Transparent Imputation:
1. `silver.dim_segment`:
   - Fill 4.5% missing `length_m` with median Bangkok segment length.
2. `gold.merged_district_15m` (585,600 rows):
   - Fill 6.08% late-night missing district speeds (`itic_avg_speed_kmh`, `itic_median_speed_kmh`,
     `itic_p10_speed_kmh`, `itic_avg_ratio`, `itic_median_ratio`) from that district's own
     Train-only historical profile `(district, hour, minute, day_type)` with deterministic +/-3% jitter
     clipped to [3.0, 120.0] km/h.
   - Adds transparent boolean flag `is_speed_imputed`.
3. `gold.itic_seg15_targets` & `gold.merged_segment_15m` (23,840,561 rows):
   - Fill missing `prev_15m_speed_kmh` (23.68%) and missing future `target_speed_kmh`/`target_ratio`/
     `target_congested`/`new_onset` (27.28%) from that segment's own Train-only historical profile
     `(seg_id, hour, minute, day_type)` with deterministic +/-3% jitter clipped to [3.0, 120.0] km/h.
   - Adds transparent boolean flags `is_prev_15m_imputed` and `is_target_imputed` so ML evaluation
     can always filter `WHERE NOT is_target_imputed` for pure Ground Truth metrics.
   - Rebuilds `gold.merged_segment_15m` view to expose all clean 0%-NULL columns + flags.

Usage:
    python -m src.warehouse.impute_warehouse
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import duckdb

from src.common import get_logger, path, write_json

log = get_logger("warehouse.impute")

DB_PATH = path("raw").parent / "bkk_traffic.duckdb"


def impute_warehouse_nulls(run_id: str = "manual") -> dict:
    t0 = time.time()
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing {DB_PATH}")

    con = duckdb.connect(str(DB_PATH))
    in_docker = Path("/opt/airflow").exists()
    n_threads = 4 if in_docker else max(4, (os.cpu_count() or 8) - 2)
    con.execute(f"PRAGMA threads={n_threads};")
    con.execute("PRAGMA memory_limit='8GB';")

    try:
        # ------------------------------------------------------------------
        # 1. silver.dim_segment: fill missing length_m with median
        # ------------------------------------------------------------------
        log.info("[1/4] Imputing silver.dim_segment.length_m with Bangkok median...")
        con.execute(
            """
            UPDATE silver.dim_segment
            SET length_m = (SELECT ROUND(MEDIAN(length_m), 1) FROM silver.dim_segment WHERE length_m IS NOT NULL)
            WHERE length_m IS NULL;
            """
        )

        # ------------------------------------------------------------------
        # 2. gold.merged_district_15m: fill 6.08% late-night missing speeds
        #    using strictly Train-only (district, hour, minute, day_type) profile
        # ------------------------------------------------------------------
        log.info("[2/4] Imputing gold.merged_district_15m (Train-only district profile +/-3%)...")
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE tmp_dist_profile_train AS
            SELECT
                district, hour, minute, day_type,
                AVG(itic_avg_speed_kmh) AS p_avg_spd,
                AVG(itic_median_speed_kmh) AS p_med_spd,
                AVG(itic_p10_speed_kmh) AS p_p10_spd,
                AVG(itic_avg_ratio) AS p_avg_rat,
                AVG(itic_median_ratio) AS p_med_rat
            FROM gold.merged_district_15m
            WHERE split = 'train' AND itic_avg_speed_kmh IS NOT NULL
            GROUP BY 1, 2, 3, 4;
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE tmp_dist_hour_train AS
            SELECT
                district, hour,
                AVG(itic_avg_speed_kmh) AS p_avg_spd,
                AVG(itic_median_speed_kmh) AS p_med_spd,
                AVG(itic_p10_speed_kmh) AS p_p10_spd,
                AVG(itic_avg_ratio) AS p_avg_rat,
                AVG(itic_median_ratio) AS p_med_rat
            FROM gold.merged_district_15m
            WHERE split = 'train' AND itic_avg_speed_kmh IS NOT NULL
            GROUP BY 1, 2;
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE tmp_city_hour_train AS
            SELECT
                hour,
                AVG(itic_avg_speed_kmh) AS p_avg_spd,
                AVG(itic_median_speed_kmh) AS p_med_spd,
                AVG(itic_p10_speed_kmh) AS p_p10_spd,
                AVG(itic_avg_ratio) AS p_avg_rat,
                AVG(itic_median_ratio) AS p_med_rat
            FROM gold.merged_district_15m
            WHERE split = 'train' AND itic_avg_speed_kmh IS NOT NULL
            GROUP BY 1;
            """
        )

        # Drop dependent main view temporarily if needed, rebuild cleanly
        con.execute(
            """
            CREATE OR REPLACE TABLE gold.merged_district_15m_new AS
            WITH j AS (
                SELECT
                    m.*,
                    (m.itic_avg_speed_kmh IS NULL) AS is_speed_imputed,
                    1.0 + (((ABS(HASH(m.district, m.ts, 1)) % 1001) / 1000.0 - 0.5) * 0.06) AS jf,
                    COALESCE(p1.p_avg_spd, p2.p_avg_spd, p3.p_avg_spd) AS b_avg_spd,
                    COALESCE(p1.p_med_spd, p2.p_med_spd, p3.p_med_spd) AS b_med_spd,
                    COALESCE(p1.p_p10_spd, p2.p_p10_spd, p3.p_p10_spd) AS b_p10_spd,
                    COALESCE(p1.p_avg_rat, p2.p_avg_rat, p3.p_avg_rat) AS b_avg_rat,
                    COALESCE(p1.p_med_rat, p2.p_med_rat, p3.p_med_rat) AS b_med_rat
                FROM gold.merged_district_15m m
                LEFT JOIN tmp_dist_profile_train p1
                  ON m.district = p1.district AND m.hour = p1.hour
                 AND m.minute = p1.minute AND m.day_type = p1.day_type
                LEFT JOIN tmp_dist_hour_train p2
                  ON m.district = p2.district AND m.hour = p2.hour
                LEFT JOIN tmp_city_hour_train p3
                  ON m.hour = p3.hour
            )
            SELECT
                dcode, district, district_en, area_sq_km, weather_point,
                ts, date, hour, minute,
                is_morning_rush, is_evening_rush, is_rush_hour, is_top_flood_day,
                itic_active_segments, itic_n_obs,
                COALESCE(itic_avg_speed_kmh, ROUND(LEAST(120.0, GREATEST(3.0, b_avg_spd * jf)), 2)) AS itic_avg_speed_kmh,
                COALESCE(itic_median_speed_kmh, ROUND(LEAST(120.0, GREATEST(3.0, b_med_spd * jf)), 2)) AS itic_median_speed_kmh,
                COALESCE(itic_p10_speed_kmh, ROUND(LEAST(120.0, GREATEST(3.0, b_p10_spd * jf)), 2)) AS itic_p10_speed_kmh,
                COALESCE(itic_avg_ratio, ROUND(LEAST(1.5, GREATEST(0.05, b_avg_rat * jf)), 4)) AS itic_avg_ratio,
                COALESCE(itic_median_ratio, ROUND(LEAST(1.5, GREATEST(0.05, b_med_rat * jf)), 4)) AS itic_median_ratio,
                itic_congested_segments, itic_pct_congested, itic_pct_level3_red, itic_new_onset_segments,
                traffy_total_15m, traffy_traffic_15m, traffy_flood_15m, traffy_road_15m,
                train_total_reports, train_traffic_reports, train_flood_reports, train_road_reports,
                train_flood_per_day, train_flood_per_sq_km, train_traffic_per_sq_km,
                precip_mm, temp_c, precip_3h_mm, precip_6h_mm, rain_bin, is_rainy_gt_1mm,
                dow, day_type, is_weekend, is_friday, is_holiday, is_bank_holiday,
                is_working_day, is_payday, is_school_term, split,
                is_speed_imputed
            FROM j;
            """
        )
        con.execute("DROP TABLE gold.merged_district_15m;")
        con.execute("ALTER TABLE gold.merged_district_15m_new RENAME TO merged_district_15m;")

        # ------------------------------------------------------------------
        # 3. gold.itic_seg15_targets: fill prev_15m_speed_kmh and target_*
        #    using strictly Train-only segment historical profiles (+/-3%)
        # ------------------------------------------------------------------
        log.info("[3/4] Imputing gold.itic_seg15_targets (23.84M rows, Train-only profile +/-3%)...")
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE tmp_seg_profile_train AS
            SELECT
                g.seg_id, g.hour, g.minute, c.day_type,
                AVG(g.speed_kmh) AS p_spd
            FROM gold.itic_seg15_targets g
            JOIN silver.calendar_daily c ON g.date = c.date
            WHERE g.split = 'train'
            GROUP BY 1, 2, 3, 4;
            """
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE gold.itic_seg15_targets_new AS
            WITH s1 AS (
                SELECT
                    g.seg_id, g.from_loc, g.to_loc, g.ts, g.date, g.hour, g.minute,
                    g.speed_kmh, g.congestion_level, g.n_obs, g.free_flow_kmh, g.ratio, g.is_congested,
                    g.target_ts,
                    g.target_speed_kmh,
                    g.prev_15m_speed_kmh,
                    g.flag_ratio_gt_1_2, g.flag_jump_gt_60,
                    g.district, g.weather_point, g.inbound_outbound, g.split,
                    g.event_time, g.available_time, g.flag_incomplete_day,
                    (g.prev_15m_speed_kmh IS NULL) AS is_prev_15m_imputed,
                    (g.target_speed_kmh IS NULL) AS is_target_imputed,
                    1.0 + (((ABS(HASH(g.seg_id, g.ts, 15)) % 1001) / 1000.0 - 0.5) * 0.06) AS j_prev,
                    1.0 + (((ABS(HASH(g.seg_id, g.ts, 60)) % 1001) / 1000.0 - 0.5) * 0.06) AS j_tgt,
                    COALESCE(p_cur.p_spd, g.speed_kmh) AS base_prev_spd,
                    COALESCE(p_tgt.p_spd, p_cur.p_spd, g.speed_kmh) AS base_tgt_spd
                FROM gold.itic_seg15_targets g
                JOIN silver.calendar_daily c_cur ON g.date = c_cur.date
                LEFT JOIN silver.calendar_daily c_tgt ON strftime(g.target_ts, '%Y-%m-%d') = c_tgt.date
                LEFT JOIN tmp_seg_profile_train p_cur
                  ON g.seg_id = p_cur.seg_id AND g.hour = p_cur.hour
                 AND g.minute = p_cur.minute AND c_cur.day_type = p_cur.day_type
                LEFT JOIN tmp_seg_profile_train p_tgt
                  ON g.seg_id = p_tgt.seg_id
                 AND EXTRACT(HOUR FROM g.target_ts)::TINYINT = p_tgt.hour
                 AND EXTRACT(MINUTE FROM g.target_ts)::TINYINT = p_tgt.minute
                 AND COALESCE(c_tgt.day_type, c_cur.day_type) = p_tgt.day_type
            ),
            s2 AS (
                SELECT
                    *,
                    COALESCE(
                        prev_15m_speed_kmh,
                        ROUND(LEAST(120.0, GREATEST(3.0, base_prev_spd * j_prev)), 2)
                    ) AS imp_prev_15m_speed_kmh,
                    COALESCE(
                        target_speed_kmh,
                        ROUND(LEAST(120.0, GREATEST(3.0, base_tgt_spd * j_tgt)), 2)
                    ) AS imp_target_speed_kmh
                FROM s1
            )
            SELECT
                seg_id, from_loc, to_loc, ts, date, hour, minute,
                speed_kmh, congestion_level, n_obs, free_flow_kmh, ratio, is_congested,
                target_ts,
                imp_target_speed_kmh AS target_speed_kmh,
                ROUND(imp_target_speed_kmh / NULLIF(free_flow_kmh, 0), 4) AS target_ratio,
                (imp_target_speed_kmh / NULLIF(free_flow_kmh, 0)) < 0.5 AS target_congested,
                (NOT is_congested) AND ((imp_target_speed_kmh / NULLIF(free_flow_kmh, 0)) < 0.5) AS new_onset,
                imp_prev_15m_speed_kmh AS prev_15m_speed_kmh,
                flag_ratio_gt_1_2, flag_jump_gt_60,
                district, weather_point, inbound_outbound, split,
                event_time, available_time, flag_incomplete_day,
                is_prev_15m_imputed, is_target_imputed
            FROM s2;
            """
        )
        con.execute("DROP TABLE gold.itic_seg15_targets;")
        con.execute("ALTER TABLE gold.itic_seg15_targets_new RENAME TO itic_seg15_targets;")

        # ------------------------------------------------------------------
        # 4. Rebuild gold.merged_segment_15m & main convenience views
        # ------------------------------------------------------------------
        log.info("[4/4] Rebuilding gold.merged_segment_15m view and auditing NULLs...")
        con.execute(
            """
            CREATE OR REPLACE VIEW gold.merged_segment_15m AS
            SELECT
                s.seg_id, s.from_loc, s.to_loc, s.ts, s.date, s.hour, s.minute,
                s.speed_kmh, s.congestion_level, s.n_obs, s.free_flow_kmh, s.ratio, s.is_congested,
                s.target_ts, s.target_speed_kmh, s.target_ratio, s.target_congested, s.new_onset,
                s.prev_15m_speed_kmh, s.flag_ratio_gt_1_2, s.flag_jump_gt_60, s.flag_incomplete_day,
                s.is_prev_15m_imputed, s.is_target_imputed,
                s.district, s.weather_point, s.inbound_outbound,
                dim.road_name_th, dim.road_name_en, dim.lat, dim.lon, dim.length_m,
                dim.n_downstream, dim.downstream_match_type, dim.downstream_seg_ids,
                w.precip_mm, w.temp_c, w.precip_3h_mm, w.precip_6h_mm, w.rain_bin, w.is_rainy_gt_1mm,
                COALESCE(tr.traffy_total_15m, 0) AS traffy_total_15m,
                COALESCE(tr.traffy_traffic_15m, 0) AS traffy_traffic_15m,
                COALESCE(tr.traffy_flood_15m, 0) AS traffy_flood_15m,
                COALESCE(tr.traffy_road_15m, 0) AS traffy_road_15m,
                st.train_flood_reports, st.train_flood_per_sq_km, st.train_traffic_per_sq_km,
                cal.dow, cal.day_type, cal.is_weekend, cal.is_friday, cal.is_holiday,
                cal.is_bank_holiday, cal.is_working_day, cal.is_payday, cal.is_school_term,
                s.split, s.event_time, s.available_time
            FROM gold.itic_seg15_targets s
            JOIN silver.dim_segment dim ON s.seg_id = dim.seg_id
            LEFT JOIN silver.weather_hourly w ON s.weather_point = w.point AND date_trunc('hour', s.ts) = w.ts
            LEFT JOIN silver.traffy_district_15m tr ON s.district = tr.district AND s.ts = tr.ts
            LEFT JOIN gold.traffy_district_static_train st ON s.district = st.district
            LEFT JOIN silver.calendar_daily cal ON s.date = cal.date;
            """
        )
        for tbl in ("merged_district_15m", "itic_seg15_targets", "merged_segment_15m"):
            con.execute(f"CREATE OR REPLACE VIEW main.{tbl} AS SELECT * FROM gold.{tbl};")

        # Audit NULLs across all target tables
        null_summary = {}
        for full_tbl in [
            "silver.dim_segment",
            "gold.district_hourly_panel",
            "gold.merged_district_15m",
            "gold.itic_seg15_targets",
            "gold.merged_segment_15m",
        ]:
            sch, name = full_tbl.split(".")
            cols = [
                r[0]
                for r in con.execute(
                    f"SELECT column_name FROM information_schema.columns "
                    f"WHERE table_catalog='bkk_traffic' AND table_schema='{sch}' AND table_name='{name}'"
                ).fetchall()
            ]
            expr = " + ".join(f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END)' for c in cols)
            n_nulls = int(con.execute(f"SELECT {expr} FROM {full_tbl}").fetchone()[0])
            null_summary[full_tbl] = n_nulls

        elapsed = round(time.time() - t0, 2)
        report = {
            "run_id": run_id,
            "database_path": DB_PATH.relative_to(path("raw").parents[1]).as_posix(),
            "elapsed_seconds": elapsed,
            "null_cells_after_imputation": null_summary,
            "pass": all(v == 0 for v in null_summary.values()),
        }
        write_json(report, path("quality") / f"{run_id}_warehouse_imputation.json")
        log.info("[DONE] Warehouse imputation finished in %.1fs: %s", elapsed, null_summary)
        return report
    finally:
        con.close()


if __name__ == "__main__":
    impute_warehouse_nulls()
