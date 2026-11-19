"""Ingest and clean Bangkok Traffy Fondue 2022 dataset (motethansen/bkk-traffy-data).

อ้างอิง Proposal ข้อ 4, 4.1 (Governance), 4.2 (Quality Gate), 5 (H4), 6, 6.1, 6.2 (Leakage):
1. ข้อมูลดิบ (Raw):
   - อ่านจากไฟล์ zip ต้นฉบับ `data/raw/traffy/bkk-traffy-data-main.zip`
   - บันทึก SHA-256 checksum ลง `logs/manifest.jsonl` และล็อกไฟล์เป็น read-only
2. การคัดกรองและ Quality Gate (ข้อ 4.2 & 6):
   - ระเบียนดิบทั้งหมดจากโฟลเดอร์ `json-data/`: 295,000 ระเบียน
   - ลบรายการซ้ำด้วย `ticket_id` (Uniqueness gate): เหลือ 185,159 รายงาน (ซ้ำ 109,841 รายการ)
     บันทึกผลลง `logs/audit/traffy_dedupe.json`
   - กรองเฉพาะในกรุงเทพมหานคร (`province == 'กรุงเทพมหานคร'`): 184,395 รายงาน (99.6%)
   - กรองช่วงเวลาศึกษา (1 มิ.ย. - 30 ก.ย. 2022 ตามเวลาไทย): 151,248 รายงาน
     (หมวดจราจร 6,008 / น้ำท่วม 15,024 / ถนน 36,191)
   - เติมชื่อเขตที่ว่างอยู่ 37 รายการ (0.02%) ด้วยพิกัด [lon, lat] ทาบกับ Polygon ของทั้ง 50 เขต กทม.
     จาก `Bangkok-districts.geojson` เพื่อให้ผ่านกฎ Join Coverage (ทุกแถวต้องมีพิกัดและเขต)
3. กฎ Governance (ข้อ 4.1) & Leakage Control (ข้อ 6.2):
   - ลบข้อความร้องเรียน (`comment`), ที่อยู่บ้านเลขที่ (`address`) และรูปภาพ (`photo`, `after_photo`) ทิ้งทั้งหมด
   - เผยแพร่ใน `data/processed/` เฉพาะตารางที่ยุบรวมระดับเขต/ชั่วโมง, ระดับเขต/วัน และระดับเขตสะสม
   - ตาราง `traffy_district_static_train` คำนวณจากรายงานในช่วง Train (1 มิ.ย. - 23 ส.ค. 2022) เท่านั้น!
     ห้ามเอาข้อมูลช่วง Validation/Test มาปนเด็ดขาด (ใช้สำหรับสมมุติฐาน H4 และกลุ่มฟีเจอร์ที่ 8)
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import CFG, TZ, get_logger, path, register_raw, write_json

log = get_logger("ingest.traffy")

SOURCE = "traffy-fondue-github-motethansen"
ZIP_REL = Path("traffy") / "bkk-traffy-data-main.zip"


def raw_zip_path() -> Path:
    return path("raw") / ZIP_REL


# ---------------------------------------------------------------------------
# Helper: Point-in-Polygon (Ray Casting) สำหรับหาเขตจากพิกัด [lon, lat]
# ---------------------------------------------------------------------------
def _point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    """ตรวจว่าจุด (lon, lat) อยู่ในวง Polygon หรือไม่ ด้วยอัลกอริทึม Ray-Casting"""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        intersect = ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / ((yj - yi) + 1e-15) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def load_bkk_districts(z: zipfile.ZipFile) -> tuple[pd.DataFrame, list[dict]]:
    """อ่านไฟล์ Bangkok-districts.geojson จาก zip เพื่อสร้างตาราง 50 เขต กทม. (dim_district)
    และเตรียม Polygon สำหรับเติมชื่อเขตที่หายไป
    """
    gj = json.loads(z.read("bkk-traffy-data-main/Bangkok-districts.geojson"))
    rows = []
    polygons = []
    w_points = CFG["weather"]["points"]

    for feat in gj["features"]:
        props = feat["properties"]
        # ตัดคำว่า "เขต" ข้างหน้าออกเพื่อให้ตรงกับชื่อเขตใน Traffy Fondue
        # และแก้คำสะกด "ราษฏร์บูรณะ" (ฏ ปฏัก) ใน GeoJSON ให้เป็น "ราษฎร์บูรณะ" (ฎ ชฎา) ที่ถูกต้อง
        dname = props["dname"].replace("เขต", "").strip()
        if dname == "ราษฏร์บูรณะ":
            dname = "ราษฎร์บูรณะ"

        coords = feat["geometry"]["coordinates"]
        outer_ring = coords[0]
        arr = np.asarray(outer_ring, dtype=float)
        c_lon, c_lat = float(arr[:, 0].mean()), float(arr[:, 1].mean())

        # จับคู่เขตเข้ากับจุดวัดฝนที่ใกล้จุดศูนย์กลางเขตที่สุด (central / north / east)
        best_pt = min(
            w_points.keys(),
            key=lambda k: (c_lat - w_points[k]["lat"]) ** 2 + (c_lon - w_points[k]["lon"]) ** 2,
        )

        rows.append(
            {
                "dcode": str(props["dcode"]),
                "district": dname,
                "district_en": str(props["dname_e"]),
                "area_sq_km": round(float(props["AREA"]) / 1e6, 3),
                "centroid_lat": round(c_lat, 6),
                "centroid_lon": round(c_lon, 6),
                "weather_point": best_pt,
            }
        )
        polygons.append({"district": dname, "ring": outer_ring, "c_lat": c_lat, "c_lon": c_lon})

    dim_df = pd.DataFrame(rows).sort_values("dcode").reset_index(drop=True)
    return dim_df, polygons


def locate_district(lon: float, lat: float, polygons: list[dict]) -> str:
    """หาชื่อเขต กทม. จากพิกัด (lon, lat) ถ้าอยู่บนเส้นขอบพอดีให้ใช้เขตที่จุดศูนย์กลางใกล้ที่สุด"""
    for p in polygons:
        if _point_in_ring(lon, lat, p["ring"]):
            return p["district"]
    best = min(polygons, key=lambda p: (lat - p["c_lat"]) ** 2 + (lon - p["c_lon"]) ** 2)
    return best["district"]


# ---------------------------------------------------------------------------
# Main Ingestion & Cleaning Pipeline
# ---------------------------------------------------------------------------
def stage_traffy(run_id: str = "manual") -> dict:
    zpath = raw_zip_path()
    if not zpath.exists():
        raise FileNotFoundError(f"Missing raw Traffy archive: {zpath}")

    # บันทึก SHA-256 checksum ของไฟล์ดิบลง manifest (ถ้ายังไม่เคยบันทึก)
    manifest_file = path("manifest")
    already_registered = False
    if manifest_file.exists():
        rel_str = zpath.resolve().relative_to(path("raw").parents[1]).as_posix()
        already_registered = rel_str in manifest_file.read_text(encoding="utf-8")
    if not already_registered:
        register_raw(zpath, SOURCE)

    with zipfile.ZipFile(zpath, "r") as z:
        dim_district, polygons = load_bkk_districts(z)

        # อ่านไฟล์ JSON ทั้งหมดในโฟลเดอร์ json-data/ เรียงตามชื่อไฟล์เพื่อให้ผลลัพธ์เหมือนเดิมทุกครั้ง
        js_files = sorted(
            [n for n in z.namelist() if "json-data/" in n and n.endswith(".json")]
        )
        records: list[dict] = []
        for fname in js_files:
            obj = json.loads(z.read(fname))
            if isinstance(obj, list):
                records.extend(obj)
            elif isinstance(obj, dict) and "results" in obj:
                records.extend(obj["results"])

    raw_count = len(records)
    log.info("loaded %s raw records from %s JSON files", raw_count, len(js_files))

    df_all = pd.DataFrame(records)

    # 1) Uniqueness Gate: ลบ ticket_id ซ้ำ (เก็บรายการแรกตามลำดับไฟล์เหมือนตอนทำ Audit ใน Proposal)
    df_dedup = df_all.drop_duplicates(subset=["ticket_id"], keep="first").copy()
    dedup_count = len(df_dedup)
    dup_removed = raw_count - dedup_count

    # แปลงเวลาจาก UTC เป็นเวลาไทย (Asia/Bangkok)
    df_dedup["ts"] = pd.to_datetime(df_dedup["timestamp"], utc=True, errors="coerce").dt.tz_convert(TZ)

    # สถานะของรายงานทั้งชุด (185,159 รายงาน) เพื่อเทียบตรงกับตารางใน Proposal หน้า 7
    state_counts_all = df_dedup["state"].value_counts(dropna=False).to_dict()

    # 2) กรองเฉพาะในกรุงเทพมหานคร (184,395 รายงาน)
    df_bkk = df_dedup[df_dedup["province"] == "กรุงเทพมหานคร"].copy()
    bkk_count = len(df_bkk)

    # 3) แยกพิกัด [lon, lat] และตรวจสอบความครบถ้วน
    coords_list = df_bkk["coords"].tolist()
    df_bkk["lon"] = [float(c[0]) if isinstance(c, (list, tuple)) and len(c) == 2 else np.nan for c in coords_list]
    df_bkk["lat"] = [float(c[1]) if isinstance(c, (list, tuple)) and len(c) == 2 else np.nan for c in coords_list]

    # 4) ตรวจและเติมชื่อเขตที่หายไป (37 รายการ = 0.02%) ด้วย Polygon ของ 50 เขต กทม.
    miss_dist_mask = df_bkk["district"].isna() | (df_bkk["district"].fillna("").str.strip() == "")
    missing_dist_bkk = int(miss_dist_mask.sum())
    df_bkk["is_district_imputed"] = miss_dist_mask
    if missing_dist_bkk > 0:
        imputed_names = [
            locate_district(lon, lat, polygons)
            for lon, lat in zip(df_bkk.loc[miss_dist_mask, "lon"], df_bkk.loc[miss_dist_mask, "lat"])
        ]
        df_bkk.loc[miss_dist_mask, "district"] = imputed_names

    # 5) แยกหมวดหมู่เรื่องร้องเรียนจากคอลัมน์ problem_type_abdul (เช่น "{น้ำท่วม,ถนน}")
    cats = df_bkk["problem_type_abdul"].fillna("").astype(str)
    df_bkk["is_traffic"] = cats.str.contains("จราจร", regex=False)
    df_bkk["is_flood"] = cats.str.contains("น้ำท่วม", regex=False)
    df_bkk["is_road"] = cats.str.contains("ถนน", regex=False)
    df_bkk["is_unspecified"] = cats.isin(["", "{}"])

    # 6) กรองเฉพาะช่วงเวลาศึกษา (1 มิ.ย. - 30 ก.ย. 2022) -> 151,248 รายงาน
    start_ts = pd.Timestamp(f"{CFG['study']['start']} 00:00:00", tz=TZ)
    end_ts = pd.Timestamp(f"{CFG['study']['end']} 23:59:59.999999", tz=TZ)
    df_study = df_bkk[(df_bkk["ts"] >= start_ts) & (df_bkk["ts"] <= end_ts)].copy()
    df_study = df_study.sort_values("ts").reset_index(drop=True)

    # 7) Governance: ตัด comment, address, photo, after_photo ออกทั้งหมดก่อนบันทึกลง staging
    keep_cols = [
        "ticket_id",
        "ts",
        "district",
        "subdistrict",
        "lat",
        "lon",
        "state",
        "count_reopen",
        "is_traffic",
        "is_flood",
        "is_road",
        "is_unspecified",
        "is_district_imputed",
    ]
    df_study_clean = df_study[keep_cols].copy()
    df_study_clean["count_reopen"] = pd.to_numeric(df_study_clean["count_reopen"], errors="coerce").fillna(0).astype(int)

    # บันทึกตารางรายรายการที่ตัดข้อมูลส่วนบุคคลแล้วไว้ใน staging (ใช้ภายในทีมเท่านั้น)
    out_stage = path("staging") / "traffy_bkk_study_clean.parquet"
    df_study_clean.to_parquet(out_stage, index=False)

    # -----------------------------------------------------------------------
    # สร้างตารางสำหรับส่งมอบใน data/processed/ (.parquet) และ data/processed/csv/ (.csv)
    # -----------------------------------------------------------------------
    csv_dir = path("processed") / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    # ตารางที่ 1: dim_district (ข้อมูลอ้างอิง 50 เขต กทม. + จุดวัดฝนที่ใกล้ที่สุด)
    dim_district.to_parquet(path("processed") / "dim_district.parquet", index=False)
    dim_district.to_csv(csv_dir / "dim_district.csv", index=False, encoding="utf-8-sig")

    # ตารางที่ 2: traffy_district_hourly (จำนวนรายงานต่อเขต x ชั่วโมง สำหรับทำ EDA)
    df_study_clean["hour_ts"] = df_study_clean["ts"].dt.floor("h")
    hourly = (
        df_study_clean.groupby(["district", "hour_ts"], as_index=False)
        .agg(
            n_total=("ticket_id", "count"),
            n_traffic=("is_traffic", "sum"),
            n_flood=("is_flood", "sum"),
            n_road=("is_road", "sum"),
            n_unspecified=("is_unspecified", "sum"),
        )
        .rename(columns={"hour_ts": "ts"})
        .sort_values(["district", "ts"])
        .reset_index(drop=True)
    )
    # ข้อมูลของชั่วโมง [ts, ts+1h) จะทราบผลรวมครบเมื่อสิ้นสุดชั่วโมง คือ ts + 1 ชั่วโมง
    hourly["event_start"] = hourly["ts"]
    hourly["available_time"] = hourly["ts"] + pd.Timedelta(hours=1)
    hourly.to_parquet(path("processed") / "traffy_district_hourly.parquet", index=False)
    hourly.to_csv(csv_dir / "traffy_district_hourly.csv", index=False, encoding="utf-8-sig")

    # ตารางที่ 3: traffy_district_daily (จำนวนรายงานต่อเขต x วัน สำหรับทำ Event Study วันน้ำท่วมหนัก)
    df_study_clean["date"] = df_study_clean["ts"].dt.strftime("%Y-%m-%d")
    daily = (
        df_study_clean.groupby(["date", "district"], as_index=False)
        .agg(
            n_total=("ticket_id", "count"),
            n_traffic=("is_traffic", "sum"),
            n_flood=("is_flood", "sum"),
            n_road=("is_road", "sum"),
            n_unspecified=("is_unspecified", "sum"),
        )
        .sort_values(["date", "district"])
        .reset_index(drop=True)
    )
    daily.to_parquet(path("processed") / "traffy_district_daily.parquet", index=False)
    daily.to_csv(csv_dir / "traffy_district_daily.csv", index=False, encoding="utf-8-sig")

    # ตารางที่ 4: traffy_district_static_train (Static Feature ระดับเขต คำนวณเฉพาะช่วง Train เท่านั้น!)
    # Proposal ข้อ 6.2: "feature จาก Traffy Fondue ระดับเขตคำนวณจากรายงานในช่วง train เท่านั้น"
    train_start, train_end = CFG["split"]["train"]
    train_end_ts = pd.Timestamp(f"{train_end} 23:59:59.999999", tz=TZ)
    df_train = df_study_clean[df_study_clean["ts"] <= train_end_ts]
    train_days = (pd.Timestamp(train_end) - pd.Timestamp(train_start)).days + 1  # 84 วัน

    static_agg = (
        df_train.groupby("district", as_index=False)
        .agg(
            train_total_reports=("ticket_id", "count"),
            train_traffic_reports=("is_traffic", "sum"),
            train_flood_reports=("is_flood", "sum"),
            train_road_reports=("is_road", "sum"),
        )
    )
    # Merge กับ dim_district ให้ครบทั้ง 50 เขต
    static_df = dim_district[["dcode", "district", "district_en", "area_sq_km", "weather_point"]].merge(
        static_agg, on="district", how="left"
    )
    for col in ["train_total_reports", "train_traffic_reports", "train_flood_reports", "train_road_reports"]:
        static_df[col] = static_df[col].fillna(0).astype(int)

    # คำนวณความหนาแน่นต่อวันและต่อตารางกิโลเมตร เพื่อให้เปรียบเทียบระหว่างเขตใหญ่กับเขตเล็กได้เป็นธรรม
    static_df["train_flood_per_day"] = (static_df["train_flood_reports"] / train_days).round(3)
    static_df["train_traffic_per_day"] = (static_df["train_traffic_reports"] / train_days).round(3)
    static_df["train_flood_per_sq_km"] = (static_df["train_flood_reports"] / static_df["area_sq_km"]).round(3)
    static_df["train_traffic_per_sq_km"] = (static_df["train_traffic_reports"] / static_df["area_sq_km"]).round(3)
    static_df["computed_from_split"] = f"train ({train_start}..{train_end})"
    static_df["available_time"] = f"{train_end}T23:59:59+07:00"

    static_df.to_parquet(path("processed") / "traffy_district_static_train.parquet", index=False)
    static_df.to_csv(csv_dir / "traffy_district_static_train.csv", index=False, encoding="utf-8-sig")

    # -----------------------------------------------------------------------
    # บันทึก Audit Log และ Quality Report (Proposal ข้อ 4.2 & 6)
    # -----------------------------------------------------------------------
    top_flood_days = (
        df_study_clean[df_study_clean["is_flood"]]
        .groupby("date")["ticket_id"]
        .count()
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    )
    top_traffic_districts = (
        df_study_clean[df_study_clean["is_traffic"]]
        .groupby("district")["ticket_id"]
        .count()
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    )
    top_flood_districts = (
        df_study_clean[df_study_clean["is_flood"]]
        .groupby("district")["ticket_id"]
        .count()
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    )

    audit_report = {
        "source": SOURCE,
        "json_files_read": len(js_files),
        "raw_records": raw_count,
        "unique_tickets": dedup_count,
        "duplicates_removed": dup_removed,
        "date_range_all": [
            str(df_dedup["ts"].min().date()),
            str(df_dedup["ts"].max().date()),
        ],
        "in_bangkok_records": bkk_count,
        "in_bangkok_pct": round(bkk_count / dedup_count * 100, 2),
        "study_period_bkk_records": len(df_study_clean),
        "study_categories": {
            "traffic": int(df_study_clean["is_traffic"].sum()),
            "flood": int(df_study_clean["is_flood"].sum()),
            "road": int(df_study_clean["is_road"].sum()),
            "unspecified": int(df_study_clean["is_unspecified"].sum()),
            "unspecified_pct_all_dedup": round(
                int(df_dedup["problem_type_abdul"].fillna("").isin(["", "{}"]).sum()) / dedup_count * 100, 2
            ),
        },
        "state_counts_all_dedup": state_counts_all,
        "missing_coords": int(df_study_clean[["lat", "lon"]].isna().any(axis=1).sum()),
        "missing_district_before_impute": missing_dist_bkk,
        "missing_district_pct_before_impute": round(missing_dist_bkk / bkk_count * 100, 4),
        "missing_district_after_impute": int(df_study_clean["district"].isna().sum()),
        "top_traffic_districts_study": top_traffic_districts,
        "top_flood_districts_study": top_flood_districts,
        "top_flood_days_study": top_flood_days,
        "governance_pii_removed": ["comment", "address", "photo", "after_photo"],
        "pass": (
            raw_count == 295000
            and dedup_count == 185159
            and bkk_count == 184395
            and len(df_study_clean) == 151248
            and int(df_study_clean["district"].isna().sum()) == 0
            and int(df_study_clean[["lat", "lon"]].isna().any(axis=1).sum()) == 0
        ),
    }

    write_json(audit_report, path("audit") / "traffy_dedupe.json")
    write_json(audit_report, path("quality") / f"{run_id}_traffy.json")
    if not audit_report["pass"]:
        raise RuntimeError(f"Traffy quality gate failed: {audit_report}")

    log.info(
        "traffy staged & processed: raw=%s -> dedup=%s -> bkk=%s -> study=%s (pass=%s)",
        raw_count,
        dedup_count,
        bkk_count,
        len(df_study_clean),
        audit_report["pass"],
    )
    return audit_report


if __name__ == "__main__":
    stage_traffy()

