"""Build the 2022 Bangkok calendar table (1 row per day).

อ้างอิง Proposal ข้อ 4, 5 (สมมุติฐาน H5), 6.2 และ 8:
- วันหยุดนักขัตฤกษ์และวันหยุดราชการกรณีพิเศษ ปี 2565 (2022) ตามประกาศสำนักนายกรัฐมนตรีและมติ ครม.
- วันเงินเดือนออก (Payday):
    1) ข้าราชการ (ตามปฏิทินกรมบัญชีกลาง ปี 2565 = 3 วันทำการก่อนวันทำการสุดท้ายของเดือน)
    2) บริษัทเอกชนรอบวันที่ 25 (ถ้าตรงเสาร์-อาทิตย์หรือวันหยุด เลื่อนเป็นวันทำการก่อนหน้า)
    3) บริษัทเอกชนรอบสิ้นเดือน (วันทำการสุดท้ายของเดือน)
- ช่วงเปิดภาคเรียน (ตามประกาศ สพฐ. ภาคเรียนที่ 1/2565: 17 พ.ค. - 10 ต.ค. 2022)
- วันศุกร์ (ใช้ทดสอบสมมุติฐาน H5: วันเงินเดือนออกและเย็นวันศุกร์รถติดกว่าปกติ)
- ประเภทวัน (day_type): 'workday' (วันทำงาน), 'weekend' (เสาร์-อาทิตย์), 'holiday' (วันหยุดนักขัตฤกษ์)

หมายเหตุเรื่อง Data Leakage (Proposal ข้อ 6.2 & 8):
ข้อมูลปฏิทินเป็นข้อมูลที่ทราบล่วงหน้าตั้งแต่วันต้นปี ดังนั้นเป็นข้อยกเว้นที่สามารถใช้ค่าของ
"เวลาเป้าหมาย (t + 60 นาที)" ได้โดยตรงโดยไม่เกิด Data Leakage
"""
from __future__ import annotations

import pandas as pd

from src.common import CFG, TZ, get_logger, path, register_raw, write_json

log = get_logger("ingest.calendar")

# วันหยุดราชการและวันหยุดพิเศษของไทยช่วง 1 มิ.ย. - 30 ก.ย. 2022 (พ.ศ. 2565)
# อ้างอิงประกาศสำนักนายกรัฐมนตรี และมติ ครม. วันที่ 4 พ.ย. 2564 (กำหนดวันหยุดราชการกรณีพิเศษปี 2565)
THAI_HOLIDAYS_2022: dict[str, str] = {
    "2022-06-03": "วันเฉลิมพระชนมพรรษาสมเด็จพระนางเจ้าฯ พระบรมราชินี",
    "2022-07-13": "วันอาสาฬหบูชา",
    "2022-07-14": "วันเข้าพรรษา",
    "2022-07-15": "วันหยุดราชการกรณีพิเศษ",
    "2022-07-28": "วันเฉลิมพระชนมพรรษาพระบาทสมเด็จพระเจ้าอยู่หัว",
    "2022-07-29": "วันหยุดราชการกรณีพิเศษ",
    "2022-08-12": "วันเฉลิมพระชนมพรรษาสมเด็จพระบรมราชชนนีพันปีหลวง / วันแม่แห่งชาติ",
}

# วันเข้าพรรษา (14 ก.ค. 2022) เป็นวันหยุดราชการ แต่ไม่ใช่วันหยุดตามประเพณีของสถาบันการเงิน (ธนาคารและเอกชนส่วนใหญ่เปิดทำการ)
NON_BANK_HOLIDAYS_2022: set[str] = {"2022-07-14"}

# ช่วงเปิดภาคเรียนโรงเรียนในสังกัด สพฐ./กทม. ปีการศึกษา 2565
# ภาคเรียนที่ 1: 17 พ.ค. - 10 ต.ค. 2022 (ครอบคลุมช่วงศึกษา มิ.ย.-ก.ย. ทั้งหมด)
SCHOOL_TERMS_2022 = [
    ("2022-05-17", "2022-10-10"),
]


def build_calendar() -> pd.DataFrame:
    """สร้างตารางปฏิทินรายวันตั้งแต่ 2022-06-01 ถึง 2022-09-30 (122 วัน)"""
    start, end = CFG["study"]["start"], CFG["study"]["end"]
    dates = pd.date_range(start, end, freq="D", tz=TZ)
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d")})

    # วันในสัปดาห์: 0=จันทร์, 1=อังคาร, ..., 4=ศุกร์, 5=เสาร์, 6=อาทิตย์
    df["dow"] = dates.dayofweek
    df["day_name"] = dates.day_name()
    df["is_weekend"] = df["dow"].isin([5, 6])
    df["is_friday"] = df["dow"] == 4  # สำหรับทดสอบสมมุติฐาน H5 (เย็นวันศุกร์)

    # วันหยุดราชการ และวันหยุดธนาคาร/เอกชน
    df["holiday_name"] = df["date"].map(THAI_HOLIDAYS_2022).fillna("")
    df["is_holiday"] = df["holiday_name"] != ""
    df["is_bank_holiday"] = df["is_holiday"] & (~df["date"].isin(NON_BANK_HOLIDAYS_2022))

    # วันทำการ (ไม่ใช่วันเสาร์-อาทิตย์ และไม่ใช่วันหยุดราชการ)
    df["is_working_day"] = (~df["is_weekend"]) & (~df["is_holiday"])

    # ประเภทวันหลัก (ใช้คำนวณ Baseline B1 Historical Average ต่อ segment x ช่วงเวลา x ประเภทวัน)
    df["day_type"] = "workday"
    df.loc[df["is_weekend"], "day_type"] = "weekend"
    df.loc[df["is_holiday"], "day_type"] = "holiday"

    # วันเงินเดือนออก 3 รูปแบบในกรุงเทพฯ:
    df["month"] = df["date"].str.slice(0, 7)
    work = df[df["is_working_day"]]

    # 1) วันทำการสุดท้ายของเดือน (เอกชนทั่วไป)
    last_workday = set(work.groupby("month")["date"].max().values)

    # 2) ข้าราชการ (กรมบัญชีกลาง): 3 วันทำการก่อนวันทำการสุดท้ายของเดือน (すなわち ลำดับที่ -4 ของวันทำการในเดือนนั้น)
    gov_payday: set[str] = set()
    # 3) เอกชนรอบวันที่ 25 (ถ้าวันที่ 25 ตรงวันหยุด ให้เลื่อนเป็นวันทำการก่อนหน้า)
    payday_25: set[str] = set()

    for m, g in df.groupby("month"):
        w_dates = g.loc[g["is_working_day"], "date"].tolist()
        if len(w_dates) >= 4:
            gov_payday.add(w_dates[-4])
        cand_25 = [d for d in w_dates if d <= f"{m}-25"]
        if cand_25:
            payday_25.add(cand_25[-1])

    df["is_gov_payday"] = df["date"].isin(gov_payday)
    df["is_payday_25"] = df["date"].isin(payday_25)
    df["is_month_end_workday"] = df["date"].isin(last_workday)
    # รวมเป็นธง is_payday หลักสำหรับทดสอบ H5
    df["is_payday"] = df["is_gov_payday"] | df["is_payday_25"] | df["is_month_end_workday"]

    # เปิดภาคเรียน
    in_term = pd.Series(False, index=df.index)
    for s, e in SCHOOL_TERMS_2022:
        in_term |= df["date"].between(s, e)
    df["is_school_term"] = in_term

    return df.drop(columns=["month"])


def ingest_calendar(run_id: str = "manual") -> pd.DataFrame:
    """บันทึกตารางปฏิทินลง raw (CSV + checksum), staging และ processed (Parquet)"""
    df = build_calendar()

    raw_csv = path("raw") / "calendar" / "calendar_2022.csv"
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    if not raw_csv.exists():
        df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
        register_raw(raw_csv, "team-calendar-2022")

    out_stage = path("staging") / "calendar_daily.parquet"
    out_proc = path("processed") / "calendar_daily.parquet"
    df.to_parquet(out_stage, index=False)
    df.to_parquet(out_proc, index=False)

    report = {
        "rows": len(df),
        "workdays": int(df["is_working_day"].sum()),
        "weekends": int(df["is_weekend"].sum()),
        "holidays": int(df["is_holiday"].sum()),
        "gov_paydays": df.loc[df["is_gov_payday"], "date"].tolist(),
        "paydays_25": df.loc[df["is_payday_25"], "date"].tolist(),
        "month_end_workdays": df.loc[df["is_month_end_workday"], "date"].tolist(),
        "all_paydays": df.loc[df["is_payday"], "date"].tolist(),
        "holiday_dates": THAI_HOLIDAYS_2022,
        "pass": len(df) == 122 and bool(df["date"].is_unique),
    }
    write_json(report, path("quality") / f"{run_id}_calendar.json")
    log.info(
        "calendar ready: %s days (%s workdays, %s holidays, %s paydays)",
        len(df),
        report["workdays"],
        report["holidays"],
        len(report["all_paydays"]),
    )
    return df


if __name__ == "__main__":
    ingest_calendar()
