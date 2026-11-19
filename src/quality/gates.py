"""Centralized Quality Gate and Leakage Verification (Proposal Section 4.2 & 6.2).

ตรวจสอบกฎคุณภาพข้อมูลทั้ง 5 ข้อ (Schema, Completeness, Uniqueness, Plausibility, Join Coverage)
และกฎป้องกันข้อมูลรั่วไหล (Leakage Controls ข้อ 6.2) บนทุกตารางใน `data/processed/`
รวมถึงมีฟังก์ชันตรวจข้อมูลความเร็วรถ iTIC (เมื่อนำข้อมูล iTIC เข้าระบบ)
"""
from __future__ import annotations

import pandas as pd

from src.common import CFG, TZ, get_logger, path, verify_manifest, write_json

log = get_logger("quality.gates")

PII_FORBIDDEN_COLS = {"comment", "address", "photo", "after_photo"}


class QualityGateError(RuntimeError):
    """Exception raised when a stop-run quality gate fails (Proposal 4.2)."""


def verify_all_processed_tables(run_id: str = "manual") -> dict:
    """ตรวจ Quality Gate + Governance + Leakage Controls ของตารางที่สร้างเสร็จแล้วทั้งหมด"""
    proc = path("processed")

    weather = pd.read_parquet(proc / "weather_hourly.parquet")
    cal = pd.read_parquet(proc / "calendar_daily.parquet")
    dim_dist = pd.read_parquet(proc / "dim_district.parquet")
    tr_hourly = pd.read_parquet(proc / "traffy_district_hourly.parquet")
    tr_daily = pd.read_parquet(proc / "traffy_district_daily.parquet")
    tr_static = pd.read_parquet(proc / "traffy_district_static_train.parquet")
    panel = pd.read_parquet(proc / "district_hourly_panel.parquet")

    # 1) Manifest checksum check (Reproducibility)
    manifest_problems = verify_manifest()

    # 2) Schema & Plausibility checks
    schema_checks = {
        "weather_timezone_bkk": str(weather["ts"].dt.tz) == TZ,
        "weather_precip_range_ok": bool(weather["precip_mm"].between(0, 200).all()),
        "weather_temp_range_ok": bool(weather["temp_c"].between(10, 45).all()),
        "traffy_counts_non_negative": bool((tr_hourly[["n_total", "n_traffic", "n_flood", "n_road"]] >= 0).all().all()),
    }

    # 3) Completeness checks
    completeness_checks = {
        "weather_8784_rows": len(weather) == 8784 and int(weather.isna().sum().sum()) == 0,
        "calendar_122_days": len(cal) == 122 and int(cal.isna().sum().sum()) == 0,
        "dim_district_50_rows": len(dim_dist) == 50 and int(dim_dist.isna().sum().sum()) == 0,
        "traffy_static_50_rows": len(tr_static) == 50 and int(tr_static.isna().sum().sum()) == 0,
        "panel_146400_rows_no_nulls": len(panel) == 146400 and int(panel.isna().sum().sum()) == 0,
    }

    # 4) Uniqueness checks
    uniqueness_checks = {
        "weather_point_ts_unique": not bool(weather.duplicated(["point", "ts"]).any()),
        "calendar_date_unique": bool(cal["date"].is_unique),
        "dim_district_unique": bool(dim_dist["district"].is_unique and dim_dist["dcode"].is_unique),
        "traffy_hourly_district_ts_unique": not bool(tr_hourly.duplicated(["district", "ts"]).any()),
        "traffy_daily_date_district_unique": not bool(tr_daily.duplicated(["date", "district"]).any()),
        "panel_district_ts_unique": not bool(panel.duplicated(["district", "ts"]).any()),
    }

    # 5) Join coverage checks (Proposal 4.2: stop run if join coverage fails)
    valid_districts = set(dim_dist["district"])
    valid_points = set(weather["point"].unique())
    join_coverage_checks = {
        "all_traffy_hourly_districts_in_dim": set(tr_hourly["district"]).issubset(valid_districts),
        "all_traffy_daily_districts_in_dim": set(tr_daily["district"]).issubset(valid_districts),
        "all_dim_weather_points_valid": set(dim_dist["weather_point"]).issubset(valid_points),
        "panel_covers_all_50_districts": set(panel["district"]) == valid_districts,
    }

    # 6) Governance & Leakage control checks (Proposal 4.1 & 6.2)
    train_start, train_end = CFG["split"]["train"]
    train_panel = panel[panel["split"] == "train"]
    train_flood_by_dist = train_panel.groupby("district")["n_flood"].sum().to_dict()
    static_flood_by_dist = tr_static.set_index("district")["train_flood_reports"].to_dict()

    leakage_and_governance_checks = {
        "no_pii_in_processed_tables": all(
            PII_FORBIDDEN_COLS.isdisjoint(df.columns)
            for df in (weather, cal, dim_dist, tr_hourly, tr_daily, tr_static, panel)
        ),
        "weather_available_time_equals_ts": bool((weather["available_time"] == weather["ts"]).all()),
        "traffy_hourly_available_time_after_hour_end": bool(
            (tr_hourly["available_time"] == tr_hourly["ts"] + pd.Timedelta(hours=1)).all()
        ),
        "traffy_static_computed_strictly_on_train_split": (
            train_flood_by_dist == static_flood_by_dist
            and bool((tr_static["computed_from_split"] == f"train ({train_start}..{train_end})").all())
        ),
        "temporal_split_strictly_ordered": (
            panel.loc[panel["split"] == "train", "date"].max() < panel.loc[panel["split"] == "valid", "date"].min()
            and panel.loc[panel["split"] == "valid", "date"].max() < panel.loc[panel["split"] == "test", "date"].min()
        ),
    }

    all_passed = (
        len(manifest_problems) == 0
        and all(schema_checks.values())
        and all(completeness_checks.values())
        and all(uniqueness_checks.values())
        and all(join_coverage_checks.values())
        and all(leakage_and_governance_checks.values())
    )

    summary = {
        "run_id": run_id,
        "manifest_problems": manifest_problems,
        "schema_and_plausibility": schema_checks,
        "completeness": completeness_checks,
        "uniqueness": uniqueness_checks,
        "join_coverage": join_coverage_checks,
        "leakage_and_governance": leakage_and_governance_checks,
        "pass": bool(all_passed),
    }
    write_json(summary, path("quality") / f"{run_id}_summary_gates.json")
    if not all_passed:
        raise QualityGateError(f"Quality/Leakage Gate FAILED: {summary}")

    log.info("ALL QUALITY & LEAKAGE GATES PASSED (run_id=%s)", run_id)
    return summary


if __name__ == "__main__":
    verify_all_processed_tables()
