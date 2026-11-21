# One-Hour-Ahead Bangkok Traffic Congestion Forecasting

CSS 342 Data Science and Data Engineering 2569

| สมาชิก | รหัส | บทบาท | ความรับผิดชอบ |
|---|---|---|---|
| **ธนากร การดี** | 67090500409 | Data Engineer | ingestion ข้อมูล iTIC/ฝน/Traffy, data contract, quality gate, DuckDB Medallion warehouse, Airflow DAG |
| **อภิเชษฐ์ ไชยสัจ** | 67090500422 | Data Scientist | EDA, สมมุติฐาน H1–H5, baseline B0/B1, LightGBM, SHAP |
| **สฤษดิ์ ศรีดิษ** | 67090500441 | ML Engineer | pipeline orchestration, model registry, monitoring, dashboard |

---

## Quick Start

```powershell
# 1. ติดตั้ง Environment (ครั้งแรกครั้งเดียว)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.lock

# 2. รัน Data Pipeline ทั้งระบบ (Bronze -> Silver -> Gold -> Quality Gates)
python run_pipeline.py

# 3. เปิดหน้าเว็บดูตารางใน DuckDB (http://localhost:8765)
python scripts/inspect_duckdb.py --ui

# 4. เปิดหน้าเว็บ Apache Airflow Medallion Pipeline (http://localhost:8080)
docker compose up -d

# 5. รันชุดทดสอบ Quality Gate & Leakage Controls อัตโนมัติ (7 tests)
pytest -v
```

---

## เอกสารส่งมอบงาน (Documentation & Handoff)

* **คู่มือส่งมอบ Handoff 1 สำหรับอภิเชษฐ์ (โค้ดเริ่มต้นทำ EDA ทั้ง 6 ข้อ):** [docs/HANDOFF_1_EDA_GUIDE.md](docs/HANDOFF_1_EDA_GUIDE.md)
* **พจนานุกรมข้อมูลครบทั้ง 17 ตาราง (`bronze.*`, `silver.*`, `gold.*`):** [docs/data_dictionary.md](docs/data_dictionary.md)
* **สัญญาข้อตกลงข้อมูล (Data Contract):** [contracts/data_contract.yaml](contracts/data_contract.yaml)

---

## โครงสร้างโฟลเดอร์โปรเจกต์

```
BKKTraffic_Project/
├── config.yaml                          # ค่าคงที่ทั้งหมด (ช่วงเวลา, split, threshold, seed)
├── run_pipeline.py                      # สคริปต์รันทั้ง pipeline ด้วยคำสั่งเดียว
├── docker-compose.yaml                  # รัน Apache Airflow Web UI (http://localhost:8080)
├── requirements.txt / requirements.lock # ไลบรารีและการล็อกเวอร์ชัน
├── pytest.ini                           # ตั้งค่าชุดทดสอบอัตโนมัติ
├── BKK_Traffic_Project_Proposal.pdf     # เอกสาร Proposal โครงงาน
│
├── contracts/
│   └── data_contract.yaml               # Data contract และกฎ Quality Gate
├── docs/
│   ├── data_dictionary.md               # พจนานุกรมอธิบายทุกคอลัมน์ทั้ง 17 ตาราง
│   └── HANDOFF_1_EDA_GUIDE.md           # คู่มือและโค้ดเริ่มต้นทำ EDA สำหรับอภิเชษฐ์
├── dags/
│   └── bkk_traffic_pipeline_dag.py      # Apache Airflow Medallion DAG (Bronze -> Silver -> Gold -> ML)
├── notebooks/                           # โฟลเดอร์สำหรับ Jupyter Notebook (EDA / Modeling ของอภิเชษฐ์)
├── scripts/
│   ├── inspect_duckdb.py                # เปิดดูตารางใน DuckDB (CLI และ Web UI http://localhost:8765)
│   └── peek.py                          # ส่องดูโครงสร้างไฟล์ .parquet / .csv
├── src/
│   ├── common.py                        # โหลด config, seed, SHA-256 manifest, logging
│   ├── ingest/                          # weather.py, calendar.py, traffy.py, itic_traffic.py
│   ├── transform/                       # district_panel.py, itic_aggregate.py
│   ├── warehouse/                       # duckdb_loader.py (สร้าง data/bkk_traffic.duckdb)
│   ├── quality/                         # gates.py (Quality Gate 5 ข้อ + Leakage Control)
│   └── features/                        # (Handoff 2) Feature Table 8 กลุ่ม + As-of Cutoff Tests
├── tests/
│   └── test_quality_and_leakage.py      # ชุดทดสอบอัตโนมัติ 7 ข้อ (pytest)
├── logs/
│   ├── manifest.jsonl                   # SHA-256 checksum ของไฟล์ข้อมูลดิบทั้งหมด
│   ├── audit/                           # Audit log การลบรายการซ้ำ (Traffy 295,000 -> 185,159)
│   └── quality/                         # รายงานผล Quality Gate ทุกครั้งที่รัน
└── data/                                # (อยู่ใน .gitignore ไม่ commit ลง Git)
    ├── bkk_traffic.duckdb               # ฐานข้อมูล DuckDB Medallion Warehouse (17 ตาราง)
    ├── raw/                             # ข้อมูลดิบ Read-only (weather, calendar, traffy, itic)
    ├── staging/                         # ข้อมูลระหว่างประมวลผล
    ├── processed/                       # ตาราง .parquet และโฟลเดอร์ csv/ แยกตามแหล่งข้อมูล
    └── features/                        # (Handoff 2) Feature Store สำหรับ LightGBM
```

---

## Data Governance (Proposal ข้อ 4.1)

* ข้อมูลดิบใน `data/raw/` ถูกล็อกเป็น read-only พร้อมบันทึก SHA-256 checksum และไม่ commit ลง Git
* อ้างอิงแหล่งข้อมูลมูลนิธิ iTIC (`CC-BY 4.0`) และ Open-Meteo (`CC-BY 4.0`) ทุกครั้งที่แสดงผล
* ไม่ใช้หรือเผยแพร่ข้อมูลระดับยานพาหนะรายคัน
* ตัดข้อความร้องเรียน ที่อยู่ และรูปภาพของ Traffy Fondue รายรายการออกทั้งหมด เผยแพร่เฉพาะจำนวนนับระดับเขต
