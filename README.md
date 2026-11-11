# One-Hour-Ahead Bangkok Traffic Congestion Forecasting

CSS 342 Data Science and Data Engineering 2569

| สมาชิก | บทบาท |
|---|---|
| ธนากร การดี | Data Engineer — ingestion, data contract, quality gate, feature table |
| อภิเชษฐ์ ไชยสัจ | Data Scientist — EDA, hypothesis, baseline, LightGBM, SHAP |
| สฤษดิ์ ศรีดิษ | ML Engineer — orchestration, registry, monitoring, dashboard |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.lock      # ติดตั้งเวอร์ชันที่ล็อกไว้
python run_pipeline.py --stage check  # ตรวจ environment + data snapshot
```

## โครงสร้าง

```
config.yaml        ค่าคงที่ทั้งหมด (ช่วงเวลา, split, threshold, seed)
run_pipeline.py    รันทั้ง pipeline ด้วยคำสั่งเดียว
contracts/         data contract (schema + กฎคุณภาพ)
docs/              data dictionary
src/ingest/        ดึงข้อมูลเข้า data/raw
src/quality/       quality gate
src/transform/     clean + aggregate เป็น segment x 15 นาที
src/features/      feature table + leakage tests
data/              (ไม่ commit) raw / quarantine / staging / processed / features
logs/manifest.jsonl  checksum ของไฟล์ raw (data snapshot)
```

## Data governance

- raw data เป็น read-only และไม่ commit ลง Git
- อ้างอิง iTIC และ Open-Meteo ตาม CC-BY 4.0 ทุกครั้งที่แสดงผล
- ไม่ใช้/เผยแพร่ข้อมูลยานพาหนะรายคัน
- ไม่เผยแพร่ข้อความ/รูปภาพร้องเรียน Traffy Fondue รายรายการ ใช้เฉพาะจำนวนระดับเขต/ชั่วโมง

