# 🚦 Bangkok 1-Hour-Ahead Traffic Congestion Forecasting & MLOps Platform

> **วิชา:** CSS 342 Data Science and Data Engineering (KMUTT)  
> **บทบาท ML Engineer (สฤษดิ์ ศรีดิษ 67090500441):** Pipeline Orchestration, Model Registry, 3-Layer Monitoring, FastAPI Serving & Interactive Minimalist Dashboard — ขับเคลื่อนด้วย **ข้อมูลจริง iTIC 122 วัน (`1,269,453` link-hours, `448` ช่วงถนนที่มีทิศทางผ่าน Data Contract 4.2 ครบ 100%) + LightGBM (`v4.7.0`) + TreeSHAP (`v0.54.1`)**

---

## 1. ภาพรวมสถาปัตยกรรม 4 แกนหลัก (ML Engineering Scope)

1. **Pipeline Orchestration & Lineage** ([pipeline_runner.py](src/ml_engineering/pipeline_runner.py), [run_pipeline.py](run_pipeline.py))
   * รัน End-to-End DAG อัตโนมัติ: `prepare/upgrade-geom -> train -> ablation -> explain -> forecast -> registry -> monitor -> charts -> dashboard -> model-card`
   * บันทึก **Run Lineage** ([latest_run.json](artifacts/lineage/latest_run.json), [pipeline_run_history.json](artifacts/lineage/pipeline_run_history.json)) พร้อม `run_id`, Git commit, เวลาแต่ละ Stage, และ SHA-256 ของทั้งข้อมูลและโมเดล
2. **Model Registry, Promotion Gates & 3-Strategy Challenger Tournament** ([registry.py](src/ml_engineering/registry.py))
   * เปรียบเทียบ **3 กลยุทธ์หน้าต่างข้อมูลฝึกสอน** ตาม Proposal ข้อ 10 หน้า 11 บนชุด Validation จริง (`Expanding Window 84d`, `Rolling 8 Weeks 56d`, `Same-Season Monsoon`) โดยไม่แอบดู Test
   * ตรวจผ่าน **4 Promotion Gates** ($\Delta\text{F1} \ge +0.01$, Onset Recall $\ge 0.55$ & Drop $\le 0.02$, Rainy F1 ไม่ลดลง, MAE $\le 0.15$) และ **Rolling-Origin 3-Fold Robustness** ($\text{F1 deviation} \le \pm 0.05$)
   * รองรับ **Audited Rollback** ตรวจสอบ SHA-256 ก่อนสลับเวอร์ชันกลับ พร้อมบันทึก [rollback_audit_log.json](artifacts/registry/rollback_audit_log.json)
3. **3-Layer MLOps Monitoring & Circuit-Breaker** ([monitoring_drift.py](src/ml_engineering/monitoring_drift.py))
   * **Layer 1 (Data Quality & Segment Circuit-Breaker):** ตรวจ Location Table Join Coverage (`100.0%`), Weather Completeness (`100.0%`), Feed Freshness SLA (`<= 15 นาที`), Minimum Daily Sample Check (`>= 8,000 แถว/วัน`), และหยุดพยากรณ์ช่วงถนนที่ข้อมูล 24 ชม. หายเกิน `20%`
   * **Layer 2 (Feature & Prediction PSI Drift):** คำนวณ Feature PSI + Prediction Output PSI + Monsoon Stress-Test PSI (ตรวจสอบสาเหตุโดยไม่สั่ง Retrain จาก PSI อย่างเดียวตาม Proposal ข้อ 10)
   * **Layer 3 (Performance Monitoring):** ติดตาม Daily & Rolling 7-Day F1 / Onset Recall / MAE ตลอด 24 วันบน Test แบบแยกเทียบเกณฑ์วันทำงาน (Weekday) กับวันหยุด (Weekend)
4. **Minimalist 3-View Dashboard (Longdo Map SDK + Google Maps SDK) & FastAPI Serving** ([app_dashboard.py](src/ml_engineering/app_dashboard.py), [api_server.py](src/ml_engineering/api_server.py))
   * **View 1 (แผนที่):** วาดเส้นถนน (`Polyline` ต้นทาง $\rightarrow$ ปลายทาง) และหมุดพยากรณ์ `448` ช่วงถนนบน **Longdo Map SDK** (พร้อมชั้นข้อมูล Traffic, เหตุการณ์, CCTV) เทียบ **พยากรณ์ +1 ชม. (ML)** vs **ค่าจริง +1 ชม. (Actual)** vs **ก่อนพยากรณ์ ($t_0$)** + สลับ 3 สถานการณ์ปี 2022 + ค้นหาชื่อถนน/แยก + จำลอง What-If สด
   * **View 2 (Model & Pipeline):** สรุปแหล่งข้อมูลจริง 6 ชุด, Out-of-Time Split, ตารางเทียบ B0/B1/B2 (24h & 06–23h), กราฟ Hourly MAE 24 ชม., ผลตัดกลุ่มฟีเจอร์ B3 Ablation, TreeSHAP + H1–H5, DAG Lineage, 4 Promotion Gates, 3-Fold Robustness และตาราง Rollback
   * **View 3 (Monitoring):** สรุปสถานะ 3-Layer MLOps Monitoring ครบทุกด่าน

---

## 2. ผลลัพธ์บนข้อมูลจริง (Real Out-of-Time Test Set: `246,451` แถว, `448` ช่วงถนน)

| โมเดล | Threshold | **F1 รวม (24h)** | **F1 กลางวัน (06–23h)** | **Onset Recall** ($\ge 0.55$) | Rush Hour F1 | Rainy F1 | **MAE** | RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0: Persistence** | `0.50` | `0.6197` | `0.6337` | `0.0000` ❌ | `0.6610` | `0.6555` | `0.1121` | `0.2012` |
| **B1: Historical Avg** | `0.50` | `0.6008` | `0.6108` | `0.2410` ❌ | `0.6554` | `0.6111` | `0.0983` | `0.1643` |
| **B2: LightGBM Champion (`v4.7.0`, 21 Core Features)** | **`0.345`** | **`0.6978`** ✅ | **`0.7056`** ✅ | **`0.5630`** ✅ | **`0.7365`** | **`0.7124`** | **`0.0902`** ✅ | **`0.1482`** |

* **Rolling-Origin 3-Fold Temporal Robustness:** F1 แกว่งสูงสุดเพียง `±0.0066` (ผ่านเกณฑ์ Proposal `<= ±0.05`)
* **สมมติฐาน H1–H5 (Proposal Section 5 หน้า 6):** ผ่านการพิสูจน์บนชุด Test จริงครบทั้ง 5 ข้อ (ดูรายละเอียดใน [MODEL_CARD.md](MODEL_CARD.md))

---

## 3. Data Contract & Data Dictionary (Proposal Section 4.2 & 12)

| Field / Group | Type | Unit | Event Time | Available Time | Source / Owner | License |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| `link_id` (`from_loc_to_loc`) | `string` | Segment ID | Static | $t_0$ | มูลนิธิ iTIC (Location Table) | CC-BY-SA 4.0 |
| `lat_from, lon_from, lat_to, lon_to` | `float32` | Degrees (WGS84) | Static | $t_0$ | มูลนิธิ iTIC (Location Table) | CC-BY-SA 4.0 |
| `red_rate_t0`, `red_rate_lag_1h..24h` | `float32` | Ratio `[0, 1]` | $[t-24\text{h}, t]$ | $\le t_0$ | มูลนิธิ iTIC (Historical Traffic) | CC-BY 4.0 |
| `target_red_rate_t1h` (Target) | `float32` | Ratio `[0, 1]` | $t + 60\text{m}$ | $t + 60\text{m}$ | มูลนิธิ iTIC (Historical Traffic) | CC-BY 4.0 |
| `downstream_red_rate_t0`, `upstream_red_rate_t0` | `float32` | Ratio `[0, 1]` | $t_0$ | $\le t_0$ | iTIC Directed Topology Graph | CC-BY-SA 4.0 |
| `precip_mm_t0`, `precip_3h_mm_t0`, `temp_c_t0` | `float32` | `mm`, `°C` | $[t-6\text{h}, t]$ | $\le t_0$ | Open-Meteo Historical Weather API | CC-BY 4.0 |
| `train_flood_per_sq_km`, `rain_x_flood_interaction` | `float32` | `reports/km²` | Train Period | Static ($\le \text{Aug 23}$) | NECTEC Traffy Fondue Open Data | Open Data |
| `is_holiday`, `is_working_day`, `is_payday` | `int8` | Binary `{0,1}` | $t + 60\text{m}$ | Known in advance | ปฏิทินวันหยุด ธปท. (BOT 2022) | Public Domain |

---

## 4. วิธีรันใช้งาน (Single Master CLI: `run_pipeline.py`)

```bash
# ติดตั้งแพ็กเกจ (หรือใช้ requirements-lock.txt สำหรับ Reproducibility 100%)
pip install -r requirements.txt

# 1. รัน End-to-End MLOps DAG ครบทุกขั้นตอน
python run_pipeline.py run-all

# 2. ดูสถานะ Champion Model, Promotion Gates และ PSI Drift
python run_pipeline.py status

# 3. เปิดเซิร์ฟเวอร์ Dashboard + FastAPI (http://127.0.0.1:8050 และ /docs)
python run_pipeline.py dashboard

# 4. รันชุดทดสอบอัตโนมัติ (Data Contract 4.2, Zero Leakage, 3-Fold Robustness, PSI, Registry/Rollback, FastAPI)
python run_pipeline.py test
```
