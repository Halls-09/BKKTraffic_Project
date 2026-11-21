# Data Dictionary — BKK Traffic Congestion Forecasting Project (DuckDB Medallion Warehouse)

> **Deliverable ข้อ 12 (Data contract/dictionary):** ระบุ `field`, `type`, `unit`, `event_time`, `available_time`, `owner`, `license` ครบทั้ง **17 ตาราง (`bronze.*`, `silver.*`, `gold.*`)** ในฐานข้อมูล **`data/bkk_traffic.duckdb`** และไฟล์ใน `data/processed/` เพื่อส่งมอบให้ **อภิเชษฐ์ ไชยสัจ (Data Scientist)** ทำ EDA และเทรนโมเดล LightGBM

* **Owner (Data Engineer):** ธนากร การดี (67090500409)
* **Database File:** `data/bkk_traffic.duckdb` (ขนาด ~786 MB, เปิดดูผ่านเว็บด้วย `python inspect_duckdb.py --ui`)
* **Timezone มาตรฐานของทั้งโปรเจกต์:** `Asia/Bangkok` (UTC+07:00)
* **ช่วงเวลาศึกษา (Study Period):** `2022-06-01` ถึง `2022-09-30` (122 วัน / 2,928 ชั่วโมง / 11,712 ช่วงเวลา 15 นาที)
* **การแบ่งชุดข้อมูลตามเวลา (Temporal Split — ข้อ 0 & 6.2):**
  * `train`: `2022-06-01` ถึง `2022-08-23` (84 วัน)
  * `valid`: `2022-08-24` ถึง `2022-09-06` (14 วัน)
  * `test`: `2022-09-07` ถึง `2022-09-30` (24 วัน)

---

## สารบัญทั้ง 17 ตารางใน `data/bkk_traffic.duckdb` (Bronze ➔ Silver ➔ Gold)

| ชั้น (Schema) | ชื่อตารางใน DuckDB | จำนวนแถว | Grain (1 แถวคืออะไร) | ผู้ใช้งานหลัก |
|---|---|---|---|---|
| **1. Bronze** | `bronze.raw_manifest_log` | `5` | 1 ไฟล์ข้อมูลดิบต้นฉบับ + รหัส SHA-256 Checksum | DE / Audit |
| **1. Bronze** | `bronze.itic_bkk_daily_raw` | `~200,000,000` | 1 ระเบียนความเร็ว iTIC ดิบ (ทุก 10 นาที × sub-offset จาก 122 ไฟล์รายวัน) | DE |
| **2. Silver** | `silver.dim_district` | `50` | 1 เขตปกครองของกรุงเทพมหานคร (ครบ 50 เขต) | DS (EDA) |
| **2. Silver** | `silver.itic_location_table` | `4,174` | 1 รหัสจุดอ้างอิงถนน (`loc_code`) ของ iTIC ในเขต กทม. | DE / DS |
| **2. Silver** | `silver.dim_segment` | `4,068` | 1 ช่วงถนนที่มีทิศทาง (`seg_id = from_loc_to_loc`) + พิกัด + ขาเข้า/ออก + ถนนข้างหน้า (`downstream_seg_ids`) + `free_flow_kmh` | DS (EDA / H3) |
| **2. Silver** | `silver.weather_hourly` | `8,784` | 1 จุดวัดฝน (`central`/`north`/`east`) × 1 ชั่วโมง | DS (EDA / H2) |
| **2. Silver** | `silver.calendar_daily` | `122` | 1 วัน (วันหยุดราชการ/ธนาคาร, วันศุกร์, วันเงินเดือนออก 3 รอบ, เปิดเทอม) | DS (EDA / H5) |
| **2. Silver** | `silver.traffy_district_15m` | `104,251` | 1 เขต × 1 ช่วงเวลา 15 นาทีที่มีการรายงาน Traffy Fondue (ตัด PII แล้ว) | DS |
| **2. Silver** | `silver.traffy_district_hourly` | `65,119` | 1 เขต × 1 ชั่วโมงที่มีการรายงาน Traffy Fondue | DS (EDA) |
| **2. Silver** | `silver.traffy_district_daily` | `6,084` | 1 วัน × 1 เขต (สำหรับทำ Event Study วันน้ำท่วมหนัก 21 ก.ค. ฯลฯ) | DS (EDA) |
| **2. Silver** | `silver.itic_segment_15m` | `23,840,561` | 1 ช่วงถนน (`seg_id`) × 15 นาที (ยุบด้วยค่ามัธยฐาน Median, กรอง 0–150 km/h) | DE / DS |
| **2. Silver** | `silver.quality_completeness_seg_day` | `481,506` | 1 ช่วงถนน (`seg_id`) × 1 วัน (เช็กความครบถ้วน $\ge 80\%$ ทั้งวัน, $\ge 90\%$ ช่วงพีค) | DS (Missing Heatmap) |
| **3. Gold** | `gold.traffy_district_static_train` | `50` | 1 เขต กทม. (ความเสี่ยงน้ำท่วม/จราจรสะสม **คำนวณเฉพาะช่วง Train เท่านั้น!**) | DS (H4 / ML) |
| **3. Gold** | `gold.district_hourly_panel` | `146,400` | กริดเต็ม 50 เขต × 2,928 ชั่วโมง (Traffy + ฝน + ปฏิทิน) | DS (EDA รายชั่วโมง) |
| **3. Gold** | `gold.itic_seg15_targets` | `23,840,561` | 1 ช่วงถนน (`seg_id`) × 15 นาที พร้อม `ratio`, `target_ratio` ($t+60$ นาที), `new_onset` และ Quality Flags | DS (EDA / ML) |
| **3. Gold** | ⭐ **`gold.merged_district_15m`** | **`585,600`** | **กริดเต็ม 50 เขต × 11,712 ช่องเวลา 15 นาที (รวม iTIC + Traffy 15 นาที + ฝน + ปฏิทิน)** | ⭐ **DS (EDA หลักระดับเขต)** |
| **3. Gold** | ⭐ **`gold.merged_segment_15m`** | **`23,840,561`** | **1 ช่วงถนน (`seg_id`) × 15 นาที (รวม iTIC + Targets + ถนน + ฝน + Traffy + ปฏิทิน)** | ⭐ **DS (EDA ถนน & LightGBM)** |

---

## รายละเอียดคอลัมน์ของตารางหลักในชั้น Silver และ Gold

### A. ตาราง `silver.dim_segment` (`4,068 ช่วงถนนที่มีทิศทางใน กทม.`)
* **Source & License:** มูลนิธิ iTIC Location Table v3.2 (`CC-BY-SA 4.0`) + HDX/BMA District Boundaries

| Field | Type | Unit | Event Time | Available Time | คำอธิบาย |
|---|---|---|---|---|---|
| `seg_id` | `VARCHAR` | `{from_loc}_{to_loc}` | Static | Static | รหัสช่วงถนนที่มีทิศทาง เช่น `10009_10010` |
| `from_loc`, `to_loc` | `INTEGER` | TMC Code | Static | Static | รหัสจุดเริ่มต้นและจุดสิ้นสุดของช่วงถนน |
| `road_name_th`, `road_name_en` | `VARCHAR` | — | Static | Static | ชื่อถนนภาษาไทยและภาษาอังกฤษ (เช่น `ถนนพหลโยธิน`) |
| `from_place_th`, `to_place_th` | `VARCHAR` | — | Static | Static | ชื่อแยก/สถานที่ต้นทางและปลายทางของช่วงถนน |
| `lat`, `lon` | `DOUBLE` | `degrees` | Static | Static | พิกัดกึ่งกลางของช่วงถนน |
| `length_m` | `DOUBLE` | `meters` | Static | Static | ระยะทางของช่วงถนนคำนวณด้วยสูตร Haversine (เมตร) |
| `inbound_outbound` | `VARCHAR` | `inbound`/`outbound` | Static | Static | ทิศทางมุ่งหน้าเข้าเมือง (`inbound`) หรือออกเมือง (`outbound`) เทียบกับใจกลาง กทม. (ข้อ 6.1) |
| `adm2_pcode`, `district`, `district_en` | `VARCHAR` | — | Static | Static | รหัสเขตและชื่อเขต กทม. ที่ถนนตั้งอยู่ |
| `weather_point` | `VARCHAR` | `central`/`north`/`east` | Static | Static | โซนจุดวัดฝนที่อยู่ใกล้ที่สุด |
| `n_downstream` | `INTEGER` | จำนวนเส้น | Static | Static | จำนวนช่วงถนนข้างหน้าที่เชื่อมต่อถัดไป |
| `downstream_seg_ids` | `VARCHAR` | comma-separated | Static | Static | รหัส `seg_id` ของถนนข้างหน้า (ตรงจุดที่ `next.from_loc == this.to_loc` ใช้ทดสอบสมมุติฐาน **H3**) |
| `free_flow_kmh` | `DOUBLE` | `km/h` | ช่วง Train | `2022-08-23 23:59:59` | ความเร็วคล่องตัว (Percentile 90 ของถนนเส้นนั้น **คำนวณเฉพาะช่วง Train เท่านั้น!** ข้อ 6.2) |
| `train_obs_slots` | `INTEGER` | ช่องเวลา | ช่วง Train | `2022-08-23 23:59:59` | จำนวนช่องเวลา 15 นาทีที่มีข้อมูลในช่วง Train |

---

### B. ตาราง `gold.itic_seg15_targets` และ `gold.merged_segment_15m` (`23,840,561 แถว`)
* **Grain:** `seg_id × ts` (1 แถว = 1 ช่วงถนนที่มีทิศทาง × 1 ช่วงเวลา 15 นาที)
* **Source & License:** มูลนิธิ iTIC (`CC-BY 4.0`) + Open-Meteo (`CC-BY 4.0`) + Traffy Fondue + ปฏิทิน 2022

| Field | Type | Unit | Event Time | Available Time | คำอธิบาย |
|---|---|---|---|---|---|
| `seg_id`, `from_loc`, `to_loc` | `VARCHAR`/`INT` | — | `ts` | `ts + 15m` | รหัสช่วงถนนที่มีทิศทาง |
| `ts`, `date`, `hour`, `minute` | `TIMESTAMP`/ฯลฯ | เวลาไทย | `ts .. ts+15m` | `ts + 15m` | เวลาเริ่มต้นของช่อง 15 นาที (`:00, :15, :30, :45`) |
| `speed_kmh` | `DOUBLE` | `km/h` | `ts .. ts+15m` | `ts + 15m` | ความเร็วมัธยฐาน (Median) ในช่วง 15 นาทีนั้น (กรอง `0 < speed <= 150`) |
| `congestion_level` | `TINYINT` | `1, 2, 3` | `ts .. ts+15m` | `ts + 15m` | ระดับสีรถติดของ iTIC (`1`=เขียวคล่องตัว, `2`=เหลืองชะลอตัว, `3`=แดงติดขัด) |
| `n_obs` | `SMALLINT` | จุดวัด | `ts .. ts+15m` | `ts + 15m` | จำนวนระเบียนย่อยที่นำมายุบรวมในช่อง 15 นาทีนั้น |
| `free_flow_kmh` | `DOUBLE` | `km/h` | ช่วง Train | สิ้นสุดช่วง Train | ความเร็ว Free-flow (Percentile 90 จากช่วง Train เท่านั้น) |
| `ratio` | `DOUBLE` | สัดส่วน | `ts .. ts+15m` | `ts + 15m` | `speed_kmh / free_flow_kmh` ณ เวลาปัจจุบัน $t$ |
| `is_congested` | `BOOLEAN` | `True/False` | `ts .. ts+15m` | `ts + 15m` | `True` เมื่อ `ratio < 0.5` ณ เวลาปัจจุบัน $t$ |
| `target_ts` | `TIMESTAMP` | เวลาไทย | `ts + 60m` | `ts + 75m` | เวลาเป้าหมายอีก 1 ชั่วโมงข้างหน้า ($t + 60$ นาที) |
| `target_speed_kmh` | `DOUBLE` | `km/h` | `ts + 60m` | `ts + 75m` | ความเร็วจริงของถนนเส้นนี้ ณ เวลา $t + 60$ นาที |
| `target_ratio` | `DOUBLE` | สัดส่วน | `ts + 60m` | `ts + 75m` | **Target หลักของโมเดล Regression:** `congestion ratio` ณ เวลา $t + 60$ นาที |
| `target_congested` | `BOOLEAN` | `True/False` | `ts + 60m` | `ts + 75m` | **Target คลาส "ติด":** `True` เมื่อ `target_ratio < 0.5` |
| `new_onset` | `BOOLEAN` | `True/False` | `ts -> ts+60m` | `ts + 75m` | **เหตุการณ์ "เริ่มติดใหม่" (Guardrail Metric):** ตอนนี้ไม่ติด (`is_congested=False`) แต่อีก 1 ชม. ติด (`target_congested=True`) |
| `flag_ratio_gt_1_2` | `BOOLEAN` | `True/False` | `ts` | `ts + 15m` | Quality Flag ข้อ 4.2 (Plausibility): `True` เมื่อ `ratio > 1.2` |
| `flag_jump_gt_60` | `BOOLEAN` | `True/False` | `ts` | `ts + 15m` | Quality Flag ข้อ 4.2 (Plausibility): `True` เมื่อความเร็วกระโดดเกิน 60 กม./ชม. ใน 15 นาที |
| `flag_incomplete_day` | `BOOLEAN` | `True/False` | `date` | สิ้นวัน | Quality Flag ข้อ 4.2 (Completeness): `True` เมื่อวันนั้นมีข้อมูลไม่ถึง 80% (หรือช่วง 06–22 น. ไม่ถึง 90%) |
| `precip_mm`, `precip_3h_mm`, `precip_6h_mm`, `rain_bin`, `is_rainy_gt_1mm`, `temp_c` | จาก `weather_hourly` | `mm` / `°C` | ชั่วโมงล่าสุด | $\le ts$ | ข้อมูลฝนและอุณหภูมิของโซนถนนเส้นนั้น (ใช้ค่าชั่วโมงล่าสุดที่รู้ผลแล้ว ไม่เกิด Leakage) |
| `traffy_total_15m`, `traffy_traffic_15m`, `traffy_flood_15m`, `traffy_road_15m` | `BIGINT` | รายงาน | `ts .. ts+15m` | `ts + 15m` | จำนวนรายงาน Traffy ในเขตนั้นในช่วง 15 นาทีนั้น |
| `train_flood_reports`, `train_flood_per_sq_km`, `train_traffic_per_sq_km` | จาก `traffy_district_static_train` | รายงาน / $\text{km}^2$ | ช่วง Train | สิ้นสุดช่วง Train | ความเสี่ยงน้ำท่วมและจราจรสะสมของเขตนั้น (คำนวณจากช่วง Train เท่านั้น สำหรับสมมุติฐาน **H4**) |
| `dow`, `day_type`, `is_weekend`, `is_friday`, `is_holiday`, `is_bank_holiday`, `is_working_day`, `is_payday`, `is_school_term` | จาก `calendar_daily` | — | `date` | ทราบล่วงหน้า | ข้อมูลปฏิทินทั้งหมด (สำหรับ Baseline B1 และสมมุติฐาน **H5**) |
| `split` | `VARCHAR` | `train`/`valid`/`test` | `date` | Static | ชุดข้อมูลตามการแบ่งเวลาใน Proposal ข้อ 0 |

---

### C. ตาราง `gold.merged_district_15m` (`585,600 แถว = 50 เขต × 11,712 ช่องเวลา 15 นาที`)
* ตารางสรุประดับ **เขต × 15 นาที** (ไม่มีช่องเวลาแหว่ง) ที่รวมตัวชี้วัดความเร็วรถของเขตนั้น (`itic_active_segments`, `itic_avg_speed_kmh`, `itic_median_speed_kmh`, `itic_p10_speed_kmh`, `itic_avg_ratio`, `itic_pct_congested`, `itic_pct_level3_red`, `itic_new_onset_segments`) เข้ากับ **Traffy ราย 15 นาที + ค่าสะสมช่วง Train + ฝน Open-Meteo + ปฏิทิน + ช่วงเวลาเร่งด่วน (`is_morning_rush`, `is_evening_rush`, `is_top_flood_day`) + `split`**
* **เหมาะที่สุดสำหรับให้อภิเชษฐ์เริ่มต้นทำ EDA ภาพรวมระดับเขตและระดับเวลา!**
