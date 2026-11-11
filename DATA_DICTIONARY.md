# 📖 พจนานุกรมข้อมูลและสัญญาข้อมูล (Data Dictionary & Data Contract Specification)

> **โครงการ:** ระบบพยากรณ์จุดรถติดในกรุงเทพมหานครล่วงหน้า 1 ชั่วโมง และวิเคราะห์ปัจจัยที่ทำให้รถติด (CSS 342)  
> **มาตรฐานตามข้อกำหนดหัวข้อที่ 12 (Deliverables):** ระบุ `field`, `type`, `unit`, `event_time`, `available_time`, `owner`, และ `license` ครบถ้วนทุกแหล่งข้อมูล

---

## 1. แหล่งข้อมูล สิทธิ์การใช้งาน และผู้รับผิดชอบ (Data Sources, Ownership & Licensing)

| แหล่งข้อมูล (Dataset) | ผู้ดูแล/เจ้าของข้อมูล (Owner) | สัญญาอนุญาต (License) | ความถี่ (Grain) | กฎการป้องกัน Data Leakage (`available_time`) |
| :--- | :--- | :--- | :--- | :--- |
| **iTIC Historical Traffic** | มูลนิธิศูนย์ข้อมูลจราจรอัจฉริยะไทย (iTIC) | **CC-BY 4.0** | ราย 15 นาที ต่อช่วงถนนที่มีทิศทาง | ใช้เฉพาะข้อมูลที่มี `ts_15m <= t`; คำนวณ P90 Free-flow เฉพาะช่วง Train (`1 มิ.ย. – 15 ส.ค. 2022`) |
| **Open-Meteo Weather** | Open-Meteo Historical Weather API (ERA5/Observed) | **CC-BY 4.0** | ราย 15–60 นาที ตามโซนพื้นที่ | โหมด `observed_lag_only`: ใช้เฉพาะฝนที่สังเกตได้จริง ณ เวลา `<= t` (ห้ามใช้ฝนจริงในอนาคต `t+60`) |
| **Traffy Fondue Reports** | กทม. & NECTEC Open API (`bkk-traffy-data`) | **Open Data BMA** | รายเหตุการณ์ (รวมสะสมระดับพื้นที่) | ลบรายการซ้ำ (`ticket_id` / พิกัดซ้ำ) และคำนวณความหนาแน่นน้ำท่วมเฉพาะจากชุด Train (`<= train_end`) |
| **Traffic Incidents** | มูลนิธิ iTIC / Longdo Event Feed | **CC-BY 4.0** | รายเหตุการณ์ | ใช้เวลาที่รับแจ้งเหตุ (`reported_time <= t`) **ห้าม**ใช้เวลาเกิดเหตุจริง (`event_time`) |

---

## 2. รายละเอียดฟีเจอร์และตัวแปรใน Gold Feature Store (`gold_features.parquet`)

| ชื่อตัวแปร (Field Name) | ชนิดข้อมูล (Type) | หน่วย (Unit) | เวลาเกิดเหตุ (`event_time`) | เวลาที่ข้อมูลพร้อมใช้ (`available_time`) | กลุ่มฟีเจอร์ / คำอธิบาย |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `seg_id` | `VARCHAR` | รหัสข้อความ | Static | Static | รหัสช่วงถนนพร้อมทิศทาง (Primary Key ร่วมกับ `ts_t`) |
| `ts_t` | `TIMESTAMP_TZ` | เวลา (`Asia/Bangkok`) | $t$ | $t$ | เวลาอ้างอิงที่ออกคำพยากรณ์ (ปัดลงทุก 15 นาที) |
| `ratio_t0` | `DOUBLE` | สัดส่วน $[0.0, 1.2]$ | $[t-15\text{m}, t]$ | $t$ | **G1:** อัตราส่วนความเร็วมัธยฐานต่อความเร็ว Free-flow (P90) ณ เวลา $t$ |
| `ratio_lag_15m` .. `60m` | `DOUBLE` | สัดส่วน $[0.0, 1.2]$ | $t-15\text{m} \dots t-60\text{m}$ | $\le t-15\text{m}$ | **G1:** ค่า Congestion Ratio ย้อนหลัง 15, 30, 45, 60 นาที |
| `delta_ratio_15m`, `30m` | `DOUBLE` | $\Delta$ สัดส่วน | $t$ | $t$ | **G1:** อัตราการชะลอตัวของรถในช่วง 15 และ 30 นาทีล่าสุด |
| `rolling_std_60m` | `DOUBLE` | ส่วนเบี่ยงเบนมาตรฐาน | $[t-60\text{m}, t]$ | $t$ | **G1:** ความผันผวนของสภาพจราจรใน 1 ชั่วโมงล่าสุด |
| `hist_avg_ratio_target_slot` | `DOUBLE` | สัดส่วน $[0.0, 1.2]$ | ชุดข้อมูล Train | ก่อนเริ่มทำนาย | **G2:** ค่าเฉลี่ย Ratio ในอดีตของถนนเส้นนั้น ณ เวลาเป้าหมายแยกวันทำงาน/วันหยุด |
| `hist_p25_ratio_target_slot` | `DOUBLE` | สัดส่วน $[0.0, 1.2]$ | ชุดข้อมูล Train | ก่อนเริ่มทำนาย | **G2:** เปอร์เซ็นไทล์ที่ 25 ของ Ratio ในอดีต ณ ช่วงเวลาเป้าหมาย |
| `current_diff_from_hist` | `DOUBLE` | $\Delta$ สัดส่วน | $t$ | $t$ | **G2:** ผลต่างระหว่างสภาพจราจรปัจจุบันกับค่าเฉลี่ยปกติของเวลานั้น |
| `target_hour_sin`, `cos` | `DOUBLE` | ค่าตรีโกณมิติ $[-1, 1]$ | $t+60\text{m}$ | ทราบล่วงหน้า ณ $t$ | **G3:** การเข้ารหัสเวลาเป้าหมายแบบวงกลม (Cyclical Encoding 96 ช่วงเวลา) |
| `is_morning_rush`, `evening` | `INTEGER` | Binary $\{0, 1\}$ | $t+60\text{m}$ | ทราบล่วงหน้า ณ $t$ | **G3:** ช่วงเวลาเร่งด่วนเช้า (06:30–09:30) และเย็น (16:00–20:00) |
| `is_weekend`, `is_public_holiday` | `INTEGER` | Binary $\{0, 1\}$ | วันของ $t$ | ทราบล่วงหน้า ณ $t$ | **G4:** วันหยุดสุดสัปดาห์ และวันหยุดนักขัตฤกษ์ราชการไทยปี 2022 |
| `is_friday_evening`, `is_payday_window` | `INTEGER` | Binary $\{0, 1\}$ | $t$ | ทราบล่วงหน้า ณ $t$ | **G4:** เย็นวันศุกร์ (16:00–21:00) และช่วงเงินเดือนออก (วันที่ 25–2) |
| `rain_mm_t0` | `DOUBLE` | มิลลิเมตร (`mm`) | $[t-15\text{m}, t]$ | $t$ | **G5:** ปริมาณฝนที่สังเกตได้จริง ณ เวลา $t$ |
| `rain_cum_3h`, `rain_cum_6h` | `DOUBLE` | มิลลิเมตรสะสม (`mm`) | $[t-3\text{h}, t]$ | $t$ | **G5:** ปริมาณฝนสะสมย้อนหลัง 3 และ 6 ชั่วโมง (ตัวแทนน้ำท่วมขังรอการระบาย) |
| `downstream_min_ratio_t0` | `DOUBLE` | สัดส่วน $[0.0, 1.2]$ | $t$ | $t$ | **G6:** ค่า Ratio ต่ำสุดของถนนข้างหน้า (Downstream Neighbors) เพื่อจับ Spillback |
| `spatial_gradient_t0` | `DOUBLE` | $\Delta$ สัดส่วน | $t$ | $t$ | **G6:** ความต่างของความเร็วถนนข้างหน้าเทียบกับถนนเส้นปัจจุบัน |
| `active_incident_on_seg` | `INTEGER` | Binary $\{0, 1\}$ | $\le t$ | `reported_time <= t` | **G7:** มีอุบัติเหตุที่รับแจ้งแล้วและยังไม่ปิดงานบนสายทาง |
| `traffy_flood_buffer500m` | `INTEGER` | จำนวนเรื่องร้องเรียน | ชุดข้อมูล Train | ก่อนเริ่มทำนาย | **G8:** จำนวนรายงานน้ำท่วม Traffy Fondue ในรัศมี 500 เมตรรอบถนน |
| `rain_x_traffy_flood_interaction` | `DOUBLE` | คะแนนปฏิสัมพันธ์ | $t$ | $t$ | **G8:** $\text{rain\_cum\_3h} \times \log(1 + \text{traffy\_flood\_buffer500m})$ (ทดสอบสมมุติฐาน H4) |
| **`target_ratio_t60`** | `DOUBLE` | สัดส่วน $[0.0, 1.2]$ | $t+60\text{m}$ | $t+60\text{m}$ | **Target (Regression):** อัตราส่วนความเร็วในอีก 1 ชั่วโมงข้างหน้า |
| **`target_is_congested_t60`** | `INTEGER` | Binary $\{0, 1\}$ | $t+60\text{m}$ | $t+60\text{m}$ | **Target (Classification):** $1$ เมื่อ `target_ratio_t60 < 0.50` ("รถติด") |
| **`is_onset_event`** | `INTEGER` | Binary $\{0, 1\}$ | $t \rightarrow t+60\text{m}$ | $t+60\text{m}$ | **Guardrail Subset:** $1$ เมื่อปัจจุบันคล่องตัว (`ratio_t0 >= 0.55`) แต่อีก 1 ชม. ติด (`< 0.50`) |
