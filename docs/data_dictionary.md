# Data Dictionary — BKK Traffic Congestion Forecasting Project

> **Deliverable ข้อ 12 (Data contract/dictionary):** ระบุ `field`, `type`, `unit`, `event_time`, `available_time`, `owner`, `license` ครบทุกตารางใน `data/processed/` เพื่อส่งมอบให้ **อภิเชษฐ์ ไชยสัจ (Data Scientist)** ทำ EDA และสร้างโมเดล

* **Owner (Data Engineer):** ธนากร การดี (67090500409)
* **Timezone มาตรฐานของทั้งโปรเจกต์:** `Asia/Bangkok` (UTC+07:00)
* **ช่วงเวลาศึกษา (Study Period):** `2022-06-01` ถึง `2022-09-30` (122 วัน / 2,928 ชั่วโมง / 11,712 ช่วงเวลา 15 นาที)
* **การแบ่งชุดข้อมูลตามเวลา (Temporal Split — ข้อ 0 & 6.2):**
  * `train`: `2022-06-01` ถึง `2022-08-23` (84 วัน)
  * `valid`: `2022-08-24` ถึง `2022-09-06` (14 วัน)
  * `test`: `2022-09-07` ถึง `2022-09-30` (24 วัน)

---

## 1. ตาราง `weather_hourly` (`data/processed/weather_hourly.csv` & `.parquet`)

* **Grain:** `point × ts` (3 จุดวัด × 2,928 ชั่วโมง = **8,784 แถว**)
* **Source & License:** Open-Meteo Historical Weather API (`CC-BY 4.0`)
* **Owner:** ธนากร การดี

| Field | Type | Unit | Event Time | Available Time | คำอธิบาย |
|---|---|---|---|---|---|
| `point` | `string` | — | `ts - 1h .. ts` | `ts` | โซนจุดวัดสภาพอากาศใน กทม. (`central`, `north`, `east`) |
| `ts` | `datetime[Asia/Bangkok]` | เวลาไทย | `ts - 1h .. ts` | `ts` | เวลาสิ้นสุดชั่วโมงที่วัด (เช่น `17:00` คือฝนช่วง `16:00–17:00`) |
| `precip_mm` | `float64` | `mm/h` | `ts - 1h .. ts` | `ts` | ปริมาณน้ำฝนรวมใน 1 ชั่วโมงที่ผ่านมา |
| `rain_mm` | `float64` | `mm/h` | `ts - 1h .. ts` | `ts` | ปริมาณน้ำฝนเฉพาะของเหลว (เท่ากับ `precip_mm` ในไทย) |
| `temp_c` | `float64` | `°C` | `ts` | `ts` | อุณหภูมิอากาศที่ความสูง 2 เมตร ณ เวลา `ts` |
| `lat` | `float64` | `degrees` | Static | Static | ละติจูดของจุดกริดสภาพอากาศ Open-Meteo |
| `lon` | `float64` | `degrees` | Static | Static | ลองจิจูดของจุดกริดสภาพอากาศ Open-Meteo |
| `precip_3h_mm` | `float64` | `mm` | `ts - 3h .. ts` | `ts` | ปริมาณฝนสะสมย้อนหลัง 3 ชั่วโมง (ใช้แทนน้ำท่วมขังสะสม) |
| `precip_6h_mm` | `float64` | `mm` | `ts - 6h .. ts` | `ts` | ปริมาณฝนสะสมย้อนหลัง 6 ชั่วโมง |
| `rain_bin` | `string` | ระดับฝน | `ts - 1h .. ts` | `ts` | ระดับความแรงของฝนตามเกณฑ์ EDA ข้อ 6.1 (`0_none`, `0_0.5_trace`, `0.5_2`, `2_5`, `5_10`, `10_20`, `gt_20`) |
| `is_rainy_gt_1mm` | `bool` | `True/False` | `ts - 1h .. ts` | `ts` | `True` เมื่อ `precip_mm > 1.0` มม./ชม. (ใช้ทดสอบสมมุติฐาน H2) |
| `event_start` | `datetime[Asia/Bangkok]` | เวลาไทย | — | `ts` | เวลาเริ่มต้นของชั่วโมงที่วัดฝน (`ts - 1h`) |
| `available_time` | `datetime[Asia/Bangkok]` | เวลาไทย | — | `ts` | เวลาที่ค่าฝนแถวนี้พร้อมใช้โดยไม่เกิด Data Leakage (`= ts`) |

---

## 2. ตาราง `calendar_daily` (`data/processed/calendar_daily.csv` & `.parquet`)

* **Grain:** `date` (122 วัน = **122 แถว**)
* **Source & License:** ประกาศสำนักนายกรัฐมนตรี/มติ ครม. ปี 2565, ปฏิทินกรมบัญชีกลาง, ประกาศ สพฐ. (`Public Domain`)
* **Owner:** ธนากร การดี
* **Leakage Note (ข้อ 6.2 & 8):** ข้อมูลปฏิทินทราบล่วงหน้าตั้งแต่วันต้นปี จึงใช้ค่าของเวลาเป้าหมาย ($t + 60$ นาที) ได้โดยไม่เกิด Leakage

| Field | Type | Unit | Event Time | Available Time | คำอธิบาย |
|---|---|---|---|---|---|
| `date` | `string` | `YYYY-MM-DD` | `date` | ทราบล่วงหน้า | วันที่ (`2022-06-01` ถึง `2022-09-30`) |
| `dow` | `int32` | `0..6` | `date` | ทราบล่วงหน้า | วันในสัปดาห์ (`0` = จันทร์, ..., `4` = ศุกร์, `5` = เสาร์, `6` = อาทิตย์) |
| `day_name` | `string` | — | `date` | ทราบล่วงหน้า | ชื่อวันภาษาอังกฤษ (`Monday`..`Sunday`) |
| `is_weekend` | `bool` | `True/False` | `date` | ทราบล่วงหน้า | `True` ถ้าเป็นวันเสาร์หรืออาทิตย์ |
| `is_friday` | `bool` | `True/False` | `date` | ทราบล่วงหน้า | `True` ถ้าเป็นวันศุกร์ (ใช้ทดสอบสมมุติฐาน H5 ช่วง 16–20 น.) |
| `holiday_name` | `string` | — | `date` | ทราบล่วงหน้า | ชื่อวันหยุดราชการภาษาไทย (ว่างถ้าไม่ใช่วันหยุด) |
| `is_holiday` | `bool` | `True/False` | `date` | ทราบล่วงหน้า | `True` ถ้าเป็นวันหยุดราชการ/วันหยุดพิเศษตามมติ ครม. (รวม 7 วัน) |
| `is_bank_holiday` | `bool` | `True/False` | `date` | ทราบล่วงหน้า | `True` ถ้าเป็นวันหยุดธนาคาร/เอกชน (ตัดวันเข้าพรรษา 14 ก.ค. ออก) |
| `is_working_day` | `bool` | `True/False` | `date` | ทราบล่วงหน้า | `True` ถ้าเป็นวันทำงานปกติ (ไม่ใช่เสาร์-อาทิตย์ และไม่ใช่วันหยุดราชการ) |
| `day_type` | `string` | — | `date` | ทราบล่วงหน้า | ประเภทวัน (`workday`, `weekend`, `holiday`) สำหรับคำนวณ Baseline B1 |
| `is_gov_payday` | `bool` | `True/False` | `date` | ทราบล่วงหน้า | วันเงินเดือนออกข้าราชการตามกรมบัญชีกลาง (3 วันทำการก่อนสิ้นเดือน) |
| `is_payday_25` | `bool` | `True/False` | `date` | ทราบล่วงหน้า | วันเงินเดือนออกเอกชนรอบวันที่ 25 (เลื่อนขึ้นถ้าชนวันหยุด) |
| `is_month_end_workday` | `bool` | `True/False` | `date` | ทราบล่วงหน้า | วันทำการสุดท้ายของเดือน |
| `is_payday` | `bool` | `True/False` | `date` | ทราบล่วงหน้า | `True` ถ้าตรงกับวันเงินเดือนออกรอบใดรอบหนึ่ง (ใช้ทดสอบ H5) |
| `is_school_term` | `bool` | `True/False` | `date` | ทราบล่วงหน้า | `True` ถ้าอยู่ในช่วงเปิดภาคเรียน สพฐ. (17 พ.ค. – 10 ต.ค. 2022) |

---

## 3. ตาราง `dim_district` (`data/processed/dim_district.csv` & `.parquet`)

* **Grain:** `district` (ครบ **50 เขตของกรุงเทพมหานคร**)
* **Source & License:** `Bangkok-districts.geojson` (`motethansen/bkk-traffy-data`, `Open Data`)
* **Owner:** ธนากร การดี

| Field | Type | Unit | Event Time | Available Time | คำอธิบาย |
|---|---|---|---|---|---|
| `dcode` | `string` | รหัสเขต 4 หลัก | Static | Static | รหัสเขตมาตรฐาน กทม. (`1001`..`1050`) |
| `district` | `string` | — | Static | Static | ชื่อเขตภาษาไทย (ไม่มีคำว่า "เขต" นำหน้า เช่น `จตุจักร`, `ราษฎร์บูรณะ`) |
| `district_en` | `string` | — | Static | Static | ชื่อเขตภาษาอังกฤษ |
| `area_sq_km` | `float64` | $\text{km}^2$ | Static | Static | พื้นที่เขต (ตารางกิโลเมตร) |
| `centroid_lat` | `float64` | `degrees` | Static | Static | ละติจูดจุดศูนย์กลางเขต |
| `centroid_lon` | `float64` | `degrees` | Static | Static | ลองจิจูดจุดศูนย์กลางเขต |
| `weather_point` | `string` | — | Static | Static | โซนจุดวัดฝนที่อยู่ใกล้จุดศูนย์กลางเขตที่สุด (`central`, `north`, `east`) |

---

## 4. ตาราง `traffy_district_hourly` และ `traffy_district_daily`

* **Grain:**
  * `traffy_district_hourly`: `district × ts` (**65,119 แถว** เฉพาะชั่วโมงที่มีรายงาน)
  * `traffy_district_daily`: `date × district` (**6,084 แถว**)
* **Source & License:** NECTEC Traffy Fondue Open API (`github.com/motethansen/bkk-traffy-data`, `Open Data`)
* **Governance (ข้อ 4.1):** ลบ `comment`, `address`, `photo`, `after_photo` ออกทั้งหมด เผยแพร่เฉพาะจำนวนนับระดับเขต
* **Owner:** ธนากร การดี

| Field | Type | Unit | Event Time | Available Time | คำอธิบาย |
|---|---|---|---|---|---|
| `district` | `string` | — | `ts` / `date` | `ts + 1h` | ชื่อเขต กทม. (เติมเขตที่ว่าง 37 รายการด้วย Polygon ครบ 100% แล้ว) |
| `ts` / `date` | `datetime` / `string` | เวลาไทย | `ts` / `date` | `ts + 1h` | เวลาเริ่มต้นชั่วโมง (ใน hourly) หรือวันที่ (ใน daily) |
| `n_total` | `int64` | รายงาน | `ts .. ts+1h` | `ts + 1h` | จำนวนเรื่องร้องเรียนทั้งหมดในชั่วโมง/วันนั้น (รวมทุกหมวดกว่า 20 หมวด) |
| `n_traffic` | `int64` | รายงาน | `ts .. ts+1h` | `ts + 1h` | จำนวนรายงานที่มีป้ายหมวด `จราจร` (รวมทั้งช่วงศึกษา = 6,008) |
| `n_flood` | `int64` | รายงาน | `ts .. ts+1h` | `ts + 1h` | จำนวนรายงานที่มีป้ายหมวด `น้ำท่วม` (รวมทั้งช่วงศึกษา = 15,024) |
| `n_road` | `int64` | รายงาน | `ts .. ts+1h` | `ts + 1h` | จำนวนรายงานที่มีป้ายหมวด `ถนน` (รวมทั้งช่วงศึกษา = 36,191) |
| `n_unspecified` | `int64` | รายงาน | `ts .. ts+1h` | `ts + 1h` | จำนวนรายงานที่ไม่ระบุหมวด (`{}`) |
| `event_start` | `datetime[Asia/Bangkok]` | เวลาไทย | `ts` | `ts + 1h` | เวลาเริ่มต้นชั่วโมงที่รวบรวมรายงาน (`= ts`) |
| `available_time` | `datetime[Asia/Bangkok]` | เวลาไทย | — | `ts + 1h` | เวลาที่ยอดรวมของชั่วโมงนั้นนิ่งและพร้อมใช้งาน (`= ts + 1 ชั่วโมง`) |

---

## 5. ตาราง `traffy_district_static_train` (`data/processed/traffy_district_static_train.csv` & `.parquet`)

* **Grain:** `district` (ครบ **50 เขต กทม.**)
* **Leakage Control (ข้อ 6.2):** คำนวณจากรายงานในช่วง **Train (`2022-06-01` ถึง `2022-08-23`, 84 วัน) เท่านั้น!** ห้ามนำข้อมูลช่วง Validation หรือ Test มาคำนวณรวมเด็ดขาด ใช้เป็น Static Feature ประจำเขตสำหรับทดสอบสมมุติฐาน **H4**

| Field | Type | Unit | Event Time | Available Time | คำอธิบาย |
|---|---|---|---|---|---|
| `dcode`, `district`, `district_en`, `area_sq_km`, `weather_point` | ตาม `dim_district` | — | Static | Static | ข้อมูลอ้างอิงประจำเขต |
| `train_total_reports` | `int64` | รายงาน | `2022-06-01..2022-08-23` | `2022-08-23 23:59:59+07:00` | รายงานทั้งหมดในเขตนั้นเฉพาะช่วง Train (84 วัน) |
| `train_traffic_reports` | `int64` | รายงาน | `2022-06-01..2022-08-23` | `2022-08-23 23:59:59+07:00` | รายงานหมวดจราจรสะสมในช่วง Train |
| `train_flood_reports` | `int64` | รายงาน | `2022-06-01..2022-08-23` | `2022-08-23 23:59:59+07:00` | รายงานหมวดน้ำท่วมสะสมในช่วง Train (ใช้ทดสอบสมมุติฐาน H4) |
| `train_road_reports` | `int64` | รายงาน | `2022-06-01..2022-08-23` | `2022-08-23 23:59:59+07:00` | รายงานหมวดถนนสะสมในช่วง Train |
| `train_flood_per_day` | `float64` | รายงาน/วัน | ช่วง Train | สิ้นสุดช่วง Train | ค่าเฉลี่ยรายงานน้ำท่วมต่อวันในช่วง Train (`train_flood_reports / 84`) |
| `train_traffic_per_day` | `float64` | รายงาน/วัน | ช่วง Train | สิ้นสุดช่วง Train | ค่าเฉลี่ยรายงานจราจรต่อวันในช่วง Train |
| `train_flood_per_sq_km` | `float64` | รายงาน/$\text{km}^2$ | ช่วง Train | สิ้นสุดช่วง Train | ความหนาแน่นรายงานน้ำท่วมต่อตารางกิโลเมตรในช่วง Train |
| `train_traffic_per_sq_km` | `float64` | รายงาน/$\text{km}^2$ | ช่วง Train | สิ้นสุดช่วง Train | ความหนาแน่นรายงานจราจรต่อตารางกิโลเมตรในช่วง Train |
| `computed_from_split` | `string` | — | — | — | ระบุแหล่งช่วงเวลาที่ใช้คำนวณ (`train (2022-06-01..2022-08-23)`) |
| `available_time` | `string` | เวลาไทย | — | `2022-08-23T23:59:59+07:00` | เวลาสิ้นสุดช่วง Train |

---

## 6. ตาราง `district_hourly_panel` (`data/processed/district_hourly_panel.csv` & `.parquet`)

* **Grain:** `district × ts` (Complete Grid: **50 เขต × 2,928 ชั่วโมง = 146,400 แถว** ไม่มีช่องเวลาแหว่ง)
* **Purpose:** ตารางสำเร็จรูปที่เชื่อมข้อมูล **เขต + Traffy รายชั่วโมง (เติม 0 ในชั่วโมงที่ไม่มีรายงาน) + ฝนตามโซนของเขต + ปฏิทิน + ค่าความเสี่ยงน้ำท่วมช่วง Train + ป้ายแบ่งชุด `split` (`train`/`valid`/`test`)** เข้าไว้ด้วยกัน เพื่อให้อภิเชษฐ์เปิดไฟล์เดียวทำ EDA และวิเคราะห์ปัจจัยเชิงพื้นที่ได้ทันที
