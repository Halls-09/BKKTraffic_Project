"""Automated pytest suite for Thanakorn's Data Engineering Deliverables (Handoff 1).

Tests:
1. Data Contract & Data Dictionary completeness (Proposal Section 12).
2. Manifest SHA-256 checksum integrity (Proposal Section 0 & 7).
3. Quality Gates (Schema, Completeness, Uniqueness, Plausibility, Join Coverage - Section 4.2).
4. Governance (No PII / comment / photo in processed data - Section 4.1).
5. Leakage Controls (Train-only static features, strictly ordered temporal split, available_time - Section 6.2).
"""
from __future__ import annotations

import yaml
import pandas as pd

from src.common import ROOT, path, verify_manifest
from src.quality.gates import verify_all_processed_tables


def test_data_contract_and_dictionary_exist():
    contract_file = ROOT / "contracts" / "data_contract.yaml"
    dict_file = ROOT / "docs" / "data_dictionary.md"
    assert contract_file.exists(), "Missing contracts/data_contract.yaml"
    assert dict_file.exists(), "Missing docs/data_dictionary.md"

    contract = yaml.safe_load(contract_file.read_text(encoding="utf-8"))
    for ds_name in [
        "weather_hourly",
        "calendar_daily",
        "dim_district",
        "traffy_district_hourly",
        "traffy_district_daily",
        "traffy_district_static_train",
        "district_hourly_panel",
    ]:
        assert ds_name in contract["datasets"], f"Missing {ds_name} in data_contract.yaml"
        ds = contract["datasets"][ds_name]
        for req_key in ["owner", "license", "grain", "event_time_col", "available_time_col"]:
            assert req_key in ds, f"Missing {req_key} for {ds_name}"


def test_raw_manifest_checksums():
    problems = verify_manifest()
    assert problems == [], f"Raw manifest checksum mismatch: {problems}"


def test_all_processed_csv_and_parquet_exist():
    proc = path("processed")
    expected_map = {
        "weather_hourly": "Open-Meo/weather_hourly.csv",
        "calendar_daily": "calendar/calendar_daily.csv",
        "dim_district": "traffy/dim_district.csv",
        "traffy_district_hourly": "traffy/traffy_district_hourly.csv",
        "traffy_district_daily": "traffy/traffy_district_daily.csv",
        "traffy_district_static_train": "traffy/traffy_district_static_train.csv",
        "district_hourly_panel": "merge-traffy-calendar-open/district_hourly_panel.csv",
    }
    for name, csv_rel in expected_map.items():
        assert (proc / f"{name}.parquet").exists(), f"Missing {name}.parquet"
        assert (proc / "csv" / csv_rel).exists() or (proc / "csv" / f"{name}.csv").exists(), f"Missing {csv_rel}"


def test_quality_and_leakage_gates_pass():
    summary = verify_all_processed_tables(run_id="pytest")
    assert summary["pass"] is True


def test_proposal_exact_audit_numbers():
    proc = path("processed")
    panel = pd.read_parquet(proc / "district_hourly_panel.parquet")
    assert len(panel) == 50 * 2928  # 146,400 rows
    assert int(panel["n_total"].sum()) == 151248
    assert int(panel["n_traffic"].sum()) == 6008
    assert int(panel["n_flood"].sum()) == 15024
    assert int(panel["n_road"].sum()) == 36191


def test_duckdb_medallion_warehouse_and_15m_tables():
    import duckdb

    db_file = ROOT / "data" / "bkk_traffic.duckdb"
    assert db_file.exists(), "Missing data/bkk_traffic.duckdb"

    con = duckdb.connect(str(db_file), read_only=True)
    expected_counts = {
        "bronze.raw_manifest_log": 5,
        "silver.dim_district": 50,
        "silver.itic_location_table": 4174,
        "silver.dim_segment": 4068,
        "silver.weather_hourly": 8784,
        "silver.calendar_daily": 122,
        "gold.traffy_district_static_train": 50,
        "gold.district_hourly_panel": 146400,
        "gold.merged_district_15m": 585600,
    }
    for tbl, exp in expected_counts.items():
        got = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        assert got == exp, f"{tbl}: expected {exp}, got {got}"

    n_seg15 = con.execute("SELECT COUNT(*) FROM gold.merged_segment_15m").fetchone()[0]
    assert n_seg15 > 20_000_000, f"Expected >20M rows in gold.merged_segment_15m, got {n_seg15}"

    min_spd, max_spd = con.execute("SELECT MIN(speed_kmh), MAX(speed_kmh) FROM gold.itic_seg15_targets").fetchone()
    assert 0 < min_spd < 10 and 80 < max_spd <= 150, f"Unexpected speed range: {min_spd}..{max_spd}"
    con.close()


def test_airflow_medallion_dag_loads():
    from dags.bkk_traffic_pipeline_dag import dag

    assert dag.dag_id == "bkk_traffic_medallion_bronze_silver_gold_ml"


def test_ml_features_and_as_of_cutoff_leakage():
    """Handoff 2: Verify gold.ml_features_15m in data/bkk_traffic_features.duckdb & strict zero-leakage controls."""
    import duckdb

    wh_file = ROOT / "data" / "bkk_traffic.duckdb"
    fe_file = ROOT / "data" / "bkk_traffic_features.duckdb"
    assert fe_file.exists(), "Missing data/bkk_traffic_features.duckdb"

    con = duckdb.connect(str(fe_file), read_only=True)
    con.execute(f"ATTACH '{wh_file.as_posix()}' AS wh (READ_ONLY);")
    try:
        n_ml = con.execute("SELECT COUNT(*) FROM gold.ml_features_15m").fetchone()[0]
        n_cols = con.execute(
            "SELECT COUNT(*) FROM information_schema.columns "
            "WHERE table_catalog = 'bkk_traffic_features' "
            "  AND table_schema='gold' AND table_name='ml_features_15m'"
        ).fetchone()[0]
        assert n_ml > 20_000_000, f"Expected >20M rows in gold.ml_features_15m, got {n_ml}"
        assert n_cols >= 90, f"Expected >=90 columns in gold.ml_features_15m, got {n_cols}"

        # 1) As-of cutoff: 15m bucket [ts, ts+15m) has available_time = ts + 15m < target_ts = ts + 60m
        bad_time = con.execute(
            """
            SELECT COUNT(*) FROM gold.ml_features_15m
            WHERE available_time <> ts + INTERVAL '15 minutes'
               OR available_time >= target_ts
               OR target_ts <> ts + INTERVAL '60 minutes'
            """
        ).fetchone()[0]
        assert bad_time == 0, f"Found {bad_time} rows violating as-of cutoff or 60m horizon!"

        # 2) Train-only Historical Profile verification (must match train split ONLY)
        hist_mismatch = con.execute(
            """
            WITH recomputed AS (
                SELECT g.seg_id, g.hour, g.minute, c.day_type,
                       ROUND(AVG(g.ratio), 4)::FLOAT AS exp_ratio
                FROM wh.gold.itic_seg15_targets g
                JOIN wh.silver.calendar_daily c ON g.date = c.date
                WHERE g.split = 'train'
                GROUP BY 1, 2, 3, 4
            )
            SELECT COUNT(*)
            FROM gold.hist_segment_profile_train h
            JOIN recomputed r
              ON h.seg_id = r.seg_id AND h.hour = r.hour
             AND h.minute = r.minute AND h.day_type = r.day_type
            WHERE ABS(h.hist_avg_ratio - r.exp_ratio) > 2e-4
            """
        ).fetchone()[0]
        assert hist_mismatch == 0, f"Historical profile leaked non-train data ({hist_mismatch} mismatches)!"

        # 3) Exact-gap lag verification (speed_lag_15m must equal speed_kmh at ts - 15m)
        lag_mismatch = con.execute(
            """
            WITH sample_rows AS (
                SELECT seg_id, ts, speed_lag_15m
                FROM gold.ml_features_15m
                WHERE speed_lag_15m IS NOT NULL
                USING SAMPLE reservoir(500 ROWS) REPEATABLE (42)
            )
            SELECT COUNT(*)
            FROM sample_rows s
            LEFT JOIN wh.gold.itic_seg15_targets prev
              ON s.seg_id = prev.seg_id
             AND prev.ts = s.ts - INTERVAL '15 minutes'
            WHERE prev.speed_kmh IS NULL
               OR ABS(s.speed_lag_15m - prev.speed_kmh) > 1e-2
            """
        ).fetchone()[0]
        assert lag_mismatch == 0, f"Found {lag_mismatch} lag-15m timestamp mismatches!"

        # 4) Temporal split ordering strictly preserved
        bounds = {
            r[0]: (r[1], r[2])
            for r in con.execute(
                "SELECT split, MIN(ts), MAX(ts) FROM gold.ml_features_15m GROUP BY split"
            ).fetchall()
        }
        assert bounds["train"][1] < bounds["valid"][0], "Train overlaps or exceeds Valid!"
        assert bounds["valid"][1] < bounds["test"][0], "Valid overlaps or exceeds Test!"
    finally:
        con.close()

