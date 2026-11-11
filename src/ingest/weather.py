"""Ingest hourly weather from Open-Meteo Historical Weather API.

แบ่งการทำงานเป็น 2 ขั้นตอน (เพื่อให้เฉพาะขั้นตอนดาวน์โหลดเท่านั้นที่ต้องใช้อินเทอร์เน็ต):

1. download_weather()  : ดึงจาก API -> เก็บไฟล์ดิบที่ data/raw/weather/point=<name>/<start>_<end>.json
                         บันทึกค่า SHA-256 checksum ลง logs/manifest.jsonl และล็อกไฟล์เป็น read-only
2. stage_weather()     : อ่านไฟล์ JSON ดิบ -> แปลงเป็นตาราง Parquet ลง data/staging/ และ data/processed/
                         พร้อมสร้างรายงานตรวจคุณภาพ (Quality Gate) ลง logs/quality/

คำอธิบายแต่ละคอลัมน์ (Features) ในตาราง weather_hourly.parquet:
---------------------------------------------------------------------------
1.  point           (str)       : ชื่อจุดวัดสภาพอากาศใน กทม. 3 จุด ('central', 'north', 'east')
2.  ts              (datetime)  : เวลาสิ้นสุดชั่วโมงที่วัด (Timezone: Asia/Bangkok) เช่น 17:00
3.  precip_mm       (float)     : ปริมาณน้ำฝนรวมใน 1 ชั่วโมงที่ผ่านมา (มม./ชม.) [ช่วง ts - 1 ชม. ถึง ts]
4.  rain_mm         (float)     : ปริมาณฝนที่เป็นของเหลวอย่างเดียว (มม./ชม.) สำหรับไทยจะมีค่าเท่ากับ precip_mm
5.  temp_c          (float)     : อุณหภูมิอากาศที่ความสูง 2 เมตรจากพื้นดิน (องศาเซลเซียส) ณ เวลา ts
6.  lat             (float)     : ละติจูดของพิกัดกริดที่โมเดลสภาพอากาศ Open-Meteo ใช้วัดจริง
7.  lon             (float)     : ลองจิจูดของพิกัดกริดที่โมเดลสภาพอากาศ Open-Meteo ใช้วัดจริง
8.  precip_3h_mm    (float)     : ปริมาณฝนสะสมย้อนหลัง 3 ชั่วโมง (มม.) ใช้แทนภาวะน้ำท่วมขังสะสมบนผิวจราจร
9.  precip_6h_mm    (float)     : ปริมาณฝนสะสมย้อนหลัง 6 ชั่วโมง (มม.) ใช้ดูผลกระทบของฝนตกต่อเนื่องยาวนาน
10. rain_bin        (str)       : ระดับความแรงของฝน แบ่งตามเกณฑ์ EDA ข้อ 6.1 ใน Proposal:
                                  - '0_none'      : ไม่มีฝน (0 มม.)
                                  - '0_0.5_trace' : ฝนปรอยๆ (>0 ถึง 0.5 มม./ชม.)
                                  - '0.5_2'       : ฝนเล็กน้อย (0.5 ถึง 2 มม./ชม.)
                                  - '2_5'         : ฝนปานกลาง (2 ถึง 5 มม./ชม.)
                                  - '5_10'        : ฝนค่อนข้างหนัก (5 ถึง 10 มม./ชม.)
                                  - '10_20'       : ฝนหนัก (10 ถึง 20 มม./ชม.)
                                  - 'gt_20'       : ฝนตกหนักมาก (>20 มม./ชม.)
11. is_rainy_gt_1mm (bool)      : True ถ้าชั่วโมงนั้นฝนตกเกิน 1.0 มม./ชม. (ใช้ทดสอบสมมุติฐาน H2 ใน Proposal ข้อ 5)
12. event_start     (datetime)  : เวลาเริ่มต้นที่ฝนตกรอบนั้นจริงๆ (เท่ากับ ts - 1 ชั่วโมง)
13. available_time  (datetime)  : เวลาที่ข้อมูลแถวนี้พร้อมใช้งานจริงโดยไม่เกิด Data Leakage (เท่ากับ ts)
                                  เช่น แถวเวลา 17:00 คือฝนที่ตกช่วง 16:00-17:00 ดังนั้นรู้ค่าได้ตั้งแต่ 17:00 เป็นต้นไป
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.common import CFG, TZ, get_logger, path, register_raw, write_json

log = get_logger("ingest.weather")

SOURCE = "open-meteo"
W = CFG["weather"]
START, END = CFG["study"]["start"], CFG["study"]["end"]

# ช่วงแบ่งระดับน้ำฝน (มม./ชม.) ตามที่ระบุใน Proposal ข้อ 6.1 (EDA บนข้อมูล iTIC)
RAIN_BINS = [-np.inf, 0.0, 0.5, 2.0, 5.0, 10.0, 10.0 + 10.0, np.inf]
RAIN_LABELS = ["0_none", "0_0.5_trace", "0.5_2", "2_5", "5_10", "10_20", "gt_20"]


def raw_file(point: str) -> Path:
    """คืนค่าพาธของไฟล์ JSON ดิบของแต่ละจุดวัดฝน"""
    return path("raw") / "weather" / f"point={point}" / f"{START}_{END}.json"


def download_weather() -> list[dict]:
    """ดึงข้อมูลฝนรายชั่วโมงจาก Open-Meteo API มาเก็บใน data/raw/
    ถ้ามีไฟล์อยู่แล้วจะข้ามทันที (Idempotent & Raw Immutable)
    """
    records = []
    for name, p in W["points"].items():
        out = raw_file(name)
        if out.exists():
            log.info("skip %s (raw already exists, raw is immutable)", name)
            continue
        params = {
            "latitude": p["lat"],
            "longitude": p["lon"],
            "start_date": START,
            "end_date": END,
            "hourly": ",".join(W["hourly_vars"]),
            "timezone": TZ,
        }
        log.info("download %s %s", name, params)
        r = requests.get(W["api_url"], params=params, timeout=60)
        r.raise_for_status()

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(r.text, encoding="utf-8")
        # บันทึก metadata ว่าดึงมาจาก URL ไหน ด้วยพารามิเตอร์อะไร และใช้สัญญาอนุญาตอะไร
        write_json(
            {
                "source": SOURCE,
                "url": r.url,
                "params": params,
                "license": "CC-BY 4.0 (Open-Meteo)",
            },
            out.with_suffix(".meta.json"),
        )
        # บันทึก SHA-256 checksum ลง manifest และล็อกไฟล์เป็น read-only
        records.append(register_raw(out, SOURCE))
        log.info("saved %s (%s bytes)", out.name, out.stat().st_size)
    return records


def parse_point(point: str) -> pd.DataFrame:
    """อ่านไฟล์ JSON ดิบของจุดวัดฝน 1 จุด แปลงเป็นตาราง DataFrame พื้นฐาน"""
    js = json.loads(raw_file(point).read_text(encoding="utf-8"))
    h = js["hourly"]
    df = pd.DataFrame(
        {
            # ชื่อจุดวัด ('central', 'north', 'east') ใช้เชื่อมกับถนนแต่ละเส้นที่อยู่ใกล้จุดนั้นที่สุด
            "point": point,
            # เวลาประจำชั่วโมง แปลงเป็น Timezone Asia/Bangkok (+07:00)
            "ts": pd.to_datetime(h["time"]).tz_localize(TZ),
            # ปริมาณน้ำฝนรวมในชั่วโมงก่อนหน้า (มม./ชม.)
            "precip_mm": pd.to_numeric(h["precipitation"], errors="coerce"),
            # ปริมาณน้ำฝนเฉพาะที่เป็นเม็ดฝน (มม./ชม.)
            "rain_mm": pd.to_numeric(h["rain"], errors="coerce"),
            # อุณหภูมิที่ความสูง 2 เมตร (องศาเซลเซียส)
            "temp_c": pd.to_numeric(h["temperature_2m"], errors="coerce"),
        }
    )
    # พิกัดจริงของกริดสภาพอากาศที่ API คืนค่ากลับมา
    df["lat"] = js.get("latitude")
    df["lon"] = js.get("longitude")
    return df


def check_weather(df: pd.DataFrame) -> dict:
    """ตรวจ Quality Gate ของข้อมูลฝน (Proposal ข้อ 4.2: ฝนต้องครอบคลุมทุกชั่วโมง ห้ามขาด)"""
    n_days = (pd.Timestamp(END) - pd.Timestamp(START)).days + 1
    expected_hours = n_days * 24  # 122 วัน * 24 ชั่วโมง = 2,928 ชั่วโมงต่อจุด
    expected_ts = pd.date_range(f"{START} 00:00", f"{END} 23:00", freq="h", tz=TZ)

    report = {"expected_hours_per_point": expected_hours, "points": {}}
    ok = True
    for pt, g in df.groupby("point"):
        missing_ts = expected_ts.difference(g["ts"])
        r = {
            "rows": len(g),
            "duplicate_ts": int(g["ts"].duplicated().sum()),
            "missing_hours": len(missing_ts),
            "null_precip": int(g["precip_mm"].isna().sum()),
            "negative_precip": int((g["precip_mm"] < 0).sum()),
            "max_precip_mm": float(g["precip_mm"].max()),
            "total_precip_mm": round(float(g["precip_mm"].sum()), 1),
            "rainy_hours_gt_1mm": int((g["precip_mm"] > 1).sum()),
        }
        r["pass"] = (
            r["rows"] == expected_hours
            and r["duplicate_ts"] == 0
            and r["missing_hours"] == 0
            and r["null_precip"] == 0
            and r["negative_precip"] == 0
        )
        ok &= r["pass"]
        report["points"][pt] = r
    report["pass"] = bool(ok)
    return report


def stage_weather(run_id: str = "manual") -> pd.DataFrame:
    """รวมข้อมูลฝนทั้ง 3 จุด สร้างฟีเจอร์ฝนสะสม/ระดับฝน ตรวจ Quality Gate และเซฟเป็น Parquet"""
    frames = []
    for name in W["points"]:
        if not raw_file(name).exists():
            raise FileNotFoundError(f"missing raw weather for '{name}', run download first")
        frames.append(parse_point(name))
    df = pd.concat(frames, ignore_index=True).sort_values(["point", "ts"]).reset_index(drop=True)

    # สร้างฟีเจอร์ตามที่ระบุใน Proposal:
    g = df.groupby("point", group_keys=False)
    # 1) ฝนสะสม 3 ชั่วโมง และ 6 ชั่วโมงล่าสุด (Proposal ข้อ 6: ต้องมี feature ฝนสะสมเพราะน้ำระบายไม่ทัน)
    df["precip_3h_mm"] = g["precip_mm"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    df["precip_6h_mm"] = g["precip_mm"].rolling(6, min_periods=1).sum().reset_index(level=0, drop=True)
    # 2) แบ่งกลุ่มระดับความแรงของฝนสำหรับทำ EDA (Proposal ข้อ 6.1)
    df["rain_bin"] = pd.cut(df["precip_mm"], bins=RAIN_BINS, labels=RAIN_LABELS).astype(str)
    # 3) ธงบอกว่าฝนตก > 1 มม./ชม. สำหรับทดสอบสมมุติฐาน H2 (Proposal ข้อ 5)
    df["is_rainy_gt_1mm"] = df["precip_mm"] > 1.0

    # กำหนด Event Time และ Available Time เพื่อป้องกัน Data Leakage (Proposal ข้อ 6.2 & 7)
    # ค่าฝนของ Open-Meteo ณ เวลา ts คือผลรวมของชั่วโมง [ts - 1h, ts] ดังนั้นรู้ค่าได้เมื่อถึงเวลา ts
    df["event_start"] = df["ts"] - pd.Timedelta(hours=1)
    df["available_time"] = df["ts"]

    # รันตรวจ Quality Gate ถ้าไม่ผ่านให้หยุดการทำงานทันที (Stop run ตามข้อ 4.2 Join coverage)
    report = check_weather(df)
    write_json(report, path("quality") / f"{run_id}_weather.json")
    if not report["pass"]:
        raise RuntimeError(f"weather quality gate FAILED: {report}")

    out_stage = path("staging") / "weather_hourly.parquet"
    out_proc = path("processed") / "weather_hourly.parquet"
    df.to_parquet(out_stage, index=False)
    df.to_parquet(out_proc, index=False)
    log.info("staged %s rows -> %s and %s", len(df), out_stage, out_proc)
    return df


if __name__ == "__main__":
    download_weather()
    stage_weather()
