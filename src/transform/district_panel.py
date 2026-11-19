"""Build the complete District x Hour panel (50 districts x 2,928 hours = 146,400 rows).

ทำไมต้องสร้างตารางนี้ (Proposal ข้อ 6, 6.1, 6.2 และ 13):
1. ในตาราง `traffy_district_hourly` มีเฉพาะชั่วโมงที่มีการแจ้งเรื่อง (65,119 แถว)
   แต่เวลาทำ EDA หรือคำนวณค่าเฉลี่ย/ความสัมพันธ์กับฝน ต้องรวมชั่วโมงที่ไม่มีการแจ้งเรื่อง (ค่า = 0) ด้วย
2. ตารางนี้สร้าง Complete Time Grid (50 เขต x 2,928 ชั่วโมง = 146,400 แถว ไม่มีรูแหว่ง) แล้วเชื่อม:
   - ข้อมูลเขต (`dim_district`: รหัสเขต, พื้นที่, โซนจุดวัดฝน `weather_point`)
   - ยอดร้องเรียน Traffy รายชั่วโมง (`n_total`, `n_traffic`, `n_flood`, `n_road`, `n_unspecified`)
   - ข้อมูลฝนและอุณหภูมิของโซนนั้น (`precip_mm`, `precip_3h_mm`, `precip_6h_mm`, `rain_bin`, `is_rainy_gt_1mm`, `temp_c`)
   - ข้อมูลปฏิทิน (`day_type`, `is_working_day`, `is_holiday`, `is_friday`, `is_payday`, `is_school_term`)
   - ค่าความเสี่ยงสะสมของเขตจากช่วง Train (`train_flood_reports`, `train_flood_per_sq_km`, `train_traffic_per_sq_km`)
   - ตัวแปรเวลาสำหรับ EDA: `hour`, `is_morning_rush` (07-09 น.), `is_evening_rush` (16-19 น.),
     `is_top_flood_day` (21 ก.ค., 20 ก.ค., 11 ก.ย. 2022) และ `split` (`train` / `valid` / `test`)
"""
from __future__ import annotations

import pandas as pd

from src.common import CFG, TZ, get_logger, path, write_json

log = get_logger("transform.district_panel")

TOP_FLOOD_DAYS = {"2022-07-21", "2022-07-20", "2022-09-11"}


def assign_split(dates: pd.Series) -> pd.Series:
    """แบ่งชุดข้อมูลตามเวลา (Temporal Split) ตาม config.yaml ห้ามสุ่มเด็ดขาด (Proposal ข้อ 6.2)"""
    sp = CFG["split"]
    out = pd.Series("unknown", index=dates.index, dtype="object")
    out.loc[dates.between(sp["train"][0], sp["train"][1])] = "train"
    out.loc[dates.between(sp["valid"][0], sp["valid"][1])] = "valid"
    out.loc[dates.between(sp["test"][0], sp["test"][1])] = "test"
    return out


def build_district_panel(run_id: str = "manual") -> pd.DataFrame:
    proc = path("processed")
    dim_dist = pd.read_parquet(proc / "dim_district.parquet")
    traffy_h = pd.read_parquet(proc / "traffy_district_hourly.parquet")
    weather_h = pd.read_parquet(proc / "weather_hourly.parquet")
    cal_d = pd.read_parquet(proc / "calendar_daily.parquet")
    static_tr = pd.read_parquet(proc / "traffy_district_static_train.parquet")

    # 1) สร้าง Complete Grid: 50 เขต x 2,928 ชั่วโมง (1 มิ.ย. - 30 ก.ย. 2022) = 146,400 แถว
    start, end = CFG["study"]["start"], CFG["study"]["end"]
    hours = pd.date_range(f"{start} 00:00", f"{end} 23:00", freq="h", tz=TZ)
    grid = (
        pd.MultiIndex.from_product([dim_dist["district"], hours], names=["district", "ts"])
        .to_frame()
        .reset_index(drop=True)
    )

    # 2) เชื่อมข้อมูลอ้างอิงประจำเขต (dim_district) + ค่าความเสี่ยงสะสมช่วง Train (traffy_district_static_train)
    static_cols = [
        "dcode",
        "district",
        "district_en",
        "area_sq_km",
        "weather_point",
        "train_total_reports",
        "train_traffic_reports",
        "train_flood_reports",
        "train_road_reports",
        "train_flood_per_day",
        "train_traffic_per_day",
        "train_flood_per_sq_km",
        "train_traffic_per_sq_km",
    ]
    panel = grid.merge(static_tr[static_cols], on="district", how="left")

    # 3) เชื่อมยอดร้องเรียน Traffy Fondue รายชั่วโมง (ชั่วโมงที่ไม่มีรายงาน เติมค่าเป็น 0)
    tr_cols = ["district", "ts", "n_total", "n_traffic", "n_flood", "n_road", "n_unspecified"]
    panel = panel.merge(traffy_h[tr_cols], on=["district", "ts"], how="left")
    for c in ["n_total", "n_traffic", "n_flood", "n_road", "n_unspecified"]:
        panel[c] = panel[c].fillna(0).astype("int32")

    # 4) เชื่อมข้อมูลฝนรายชั่วโมงตามโซน weather_point ของแต่ละเขต
    # หมายเหตุ: ใน Open-Meteo ค่าที่เวลา ts คือฝนของชั่วโมง [ts-1h, ts] ซึ่งมี available_time = ts
    w_cols = [
        "point",
        "ts",
        "precip_mm",
        "temp_c",
        "precip_3h_mm",
        "precip_6h_mm",
        "rain_bin",
        "is_rainy_gt_1mm",
    ]
    panel = panel.merge(
        weather_h[w_cols].rename(columns={"point": "weather_point"}),
        on=["weather_point", "ts"],
        how="left",
    )

    # 5) สร้างฟีเจอร์เวลาและเชื่อมตารางปฏิทิน (calendar_daily)
    panel["date"] = panel["ts"].dt.strftime("%Y-%m-%d")
    panel["hour"] = panel["ts"].dt.hour.astype("int32")
    # ช่วงเร่งด่วนตามที่ระบุใน Proposal หน้า 7 (07-09 น. และ 16-19 น.)
    panel["is_morning_rush"] = panel["hour"].between(7, 9)
    panel["is_evening_rush"] = panel["hour"].between(16, 19)
    panel["is_rush_hour"] = panel["is_morning_rush"] | panel["is_evening_rush"]
    panel["is_top_flood_day"] = panel["date"].isin(TOP_FLOOD_DAYS)

    cal_cols = [
        "date",
        "dow",
        "day_type",
        "is_weekend",
        "is_friday",
        "is_holiday",
        "is_bank_holiday",
        "is_working_day",
        "is_payday",
        "is_school_term",
    ]
    panel = panel.merge(cal_d[cal_cols], on="date", how="left")

    # 6) แบ่งชุดข้อมูล train / valid / test ตามวันที่
    panel["split"] = assign_split(panel["date"])
    panel = panel.sort_values(["dcode", "ts"]).reset_index(drop=True)

    # 7) ตรวจความถูกต้องก่อนบันทึก
    expected_rows = 50 * len(hours)  # 146,400
    null_count = int(panel.isna().sum().sum())
    report = {
        "expected_rows": expected_rows,
        "actual_rows": len(panel),
        "districts": int(panel["district"].nunique()),
        "hours_per_district": len(hours),
        "total_null_cells": null_count,
        "split_row_counts": panel["split"].value_counts().to_dict(),
        "total_study_reports_check": int(panel["n_total"].sum()),
        "total_study_traffic_check": int(panel["n_traffic"].sum()),
        "total_study_flood_check": int(panel["n_flood"].sum()),
        "total_study_road_check": int(panel["n_road"].sum()),
        "pass": (
            len(panel) == expected_rows
            and null_count == 0
            and int(panel["n_total"].sum()) == 151248
            and int(panel["n_traffic"].sum()) == 6008
            and int(panel["n_flood"].sum()) == 15024
            and int(panel["n_road"].sum()) == 36191
        ),
    }
    write_json(report, path("quality") / f"{run_id}_district_panel.json")
    if not report["pass"]:
        raise RuntimeError(f"district_panel quality check failed: {report}")

    out_pq = proc / "district_hourly_panel.parquet"
    csv_dir = proc / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    out_csv = csv_dir / "district_hourly_panel.csv"
    panel.to_parquet(out_pq, index=False)
    panel.to_csv(out_csv, index=False, encoding="utf-8-sig")
    log.info("built district_hourly_panel: %s rows x %s cols -> %s", len(panel), panel.shape[1], out_csv)
    return panel


if __name__ == "__main__":
    build_district_panel()
