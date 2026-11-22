# 📘 รายงานวิเคราะห์ EDA ฉบับสมบูรณ์: โครงสร้างฟีเจอร์ จุดที่ควรทำเพิ่ม และแผนการนำข้อมูลไปใช้งานต่อ
**(Comprehensive EDA Analysis, Complete Feature Audit & Downstream Modeling Roadmap)**

> **โปรเจกต์:** Bangkok Traffic Congestion Analysis & 1-Hour-Ahead Forecasting (CSS 342 KMUTT)  
> **ฐานข้อมูลหลัก:** [bkk_traffic.duckdb](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/data/bkk_traffic.duckdb) (`1 มิ.ย. – 30 ก.ย. 2022` | `122 วัน` | `23,840,561` ระเบียนระดับ 15 นาที | `4,068` ช่วงถนน | ครบทั้ง `50` เขต กทม.)  
> **เอกสารที่เกี่ยวข้อง:** [EDA_INSIGHTS_REPORT.md](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/EDA_INSIGHTS_REPORT.md) | [BLINDSPOT_ANALYSIS.md](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/BLINDSPOT_ANALYSIS.md) | [eda_report.html](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/outputs/eda/eda_report.html)

---

## ส่วนที่ 1: สรุปภาพรวมการทำ EDA ทั้งหมดอย่างละเอียด (Comprehensive EDA Synthesis)

ฐานข้อมูล `bkk_traffic.duckdb` ถูกออกแบบในรูปแบบ **Medallion Architecture** (`bronze` $\rightarrow$ `silver` $\rightarrow$ `gold`) โดยตารางหลักที่ใช้ในการวิเคราะห์และต่อยอดคือ:
* **`gold.merged_segment_15m`** (`23,840,561` แถว, `58` คอลัมน์): ข้อมูลจราจรระดับช่วงถนนทุก 15 นาที ผสานสภาพอากาศ, Traffy Fondue และปฏิทิน
* **`gold.merged_district_15m`** (`585,600` แถว = `50 เขต × 122 วัน × 96 สล็อต/วัน`): ข้อมูลสรุประดับเขตทุก 15 นาที

### 1.1 ข้อค้นพบหลัก 5 มิติจากการทำ EDA (กราฟที่ 01–11)

| มิติการวิเคราะห์ | กราฟอ้างอิง | ข้อค้นพบเชิงตัวเลขที่สำคัญที่สุด (Key Empirical Findings) |
| :--- | :---: | :--- |
| **1. มิติเชิงพื้นที่ (Spatial & Urban Rings)** | 01, 02, 06 | • **กทม. ชั้นใน (21 เขต)** ติดหนักกว่า **ชั้นนอก (11 เขต)** ถึง **`+63.1%`** (`21.58%` vs `13.23%`, ช้ากว่าเกือบ `10 km/h`)<br>• เขตติดหนักสุด: **คลองเตย (`32.0%`)**, **ราชเทวี (`31.3%`)**, **บางรัก (`28.9%`)** และช้าสุดคือ **สัมพันธวงศ์ (`21 km/h`)**<br>• จุดคอขวดเรื้อรังอันดับ 1–2: **ถนนรางน้ำ (`80.9%`)** และ **ทางพิเศษศรีรัช ด่านประชาชื่น (`80.2%`)** |
| **2. มิติเวลาและปฏิทิน (Temporal & Calendar)** | 03, 04, 08 | • จังหวะพีคระดับ 15 นาที: เช้า **`07:45 น.` (`41.3%`)** vs เย็น **`17:45 น.` (`46.1%`)** และ **ศุกร์แห่งชาติ `17:45 น.` พุ่งถึง `49.0%`**<br>• **Interaction พีคสุด:** **เย็นวันศุกร์ในสัปดาห์ที่เงินเดือนออก** ติดหนักที่สุดในทุกกลุ่ม (**`44.8%`**) แซงวันทำงานสิ้นเดือน (`41.8%`) และเย็นวันทำงานปกติ (`37.4%`)<br>• **Pre-Holiday Exodus:** วันก่อนหยุดยาว (เช่น 11 ส.ค., 2 มิ.ย.) ติดอันดับ Top 5 วันที่รถติดสุดทั้งปี |
| **3. มิติฝนและน้ำท่วม (Weather & Flood Chain)** | 05, 07, 10 | • **`84.8%`** ของถนนทั้ง 4,068 ช่วงแย่ลงเมื่อฝนตก (`>1 mm/ชม.`), ดันรถติดเพิ่ม **`+36.1%`** ($p \approx 0$)<br>• **ห่วงโซ่ผลกระทบ:** ฝนแรง (แยกชั่วโมงเร่งด่วนแล้วพุ่งทางเดียวถึง `38%`) $\rightarrow$ ปะทะเขตเสี่ยงน้ำท่วม (`+7.3 pp`) $\rightarrow$ เกิดแจ้งน้ำท่วมสด Traffy **`>=2 เรื่อง/15น.` ดันรถติดพุ่ง `34.8%` (`+80.4%`)**<br>• **จุดอิ่มตัวท่อระบายน้ำ:** ฝนสะสม 6 ชม. เกิน **`15–30 mm`** ทำให้อัตราแจ้งน้ำท่วมพุ่ง **7.1 เท่า** และฝนสะสม 3 ชม. (`r = +0.17`) อธิบายรถติดดีกว่าฝนชั่วโมงปัจจุบัน (`r = +0.12`) ถึง **`+44%`**<br>• **ปัจจัยน้ำท่วมระดับเขต:** ไม่ได้เกิดจากปริมาณฝนเฉลี่ยต่างกัน (`r = +0.10`) แต่สัมพันธ์สูงสุดกับ **ปัญหาโครงสร้างถนน (`traffy_road`, `r = +0.67`)** และ **โซนฝั่งตะวันออก (`678 เรื่อง/เขต`, สูงกว่าโซนกลาง `4.4 เท่า`)** |
| **4. มิติการกระจายตัวทางสถิติ (Distributions & Log Transform)** | 09 | • `speed_kmh`: `Mean 38.0`, `Median 36.0`, `IQR 21.5–50.0 km/h`, `Skew +0.87`<br>• `ratio`: `P25 = 0.500` ตรงกับเกณฑ์รถติดพอดี (`23.1%` ติดขัด), Outlier ขวา (`>1.2`) มี `2.94%`<br>• ตัวแปรฝน (`Skew +5.64`, ศูนย์ `64%`), Traffy (`Skew +16.5`, ศูนย์ `82–98%`), ขนาดพื้นที่เขต (`Skew +3.24`) และความยาวถนน (`Skew +3.65`) เบ้ขวาจัด $\rightarrow$ เมื่อทำ **`log1p(x)`** ความเบ้ลดเข้าใกล้ `0` ทันที (`length_m`: `+3.65 -> +0.13`, `area_sq_km`: `+3.34 -> +0.31`, `precip_6h`: `+2.04 -> +0.52`) |
| **5. มิติคุณภาพข้อมูลและพลวัตเป้าหมาย (Quality, Transition & Drift)** | 11 | • **Transition Matrix (`17.34M` แถวที่มี Target +1h):** `25.16%` ของสล็อต **เปลี่ยนสถานะใน 1 ชม.** (`12.54%` เป็น **New Onset โล่ง$\rightarrow$ติด** จำนวน `2.17M` เหตุการณ์ และ `12.62%` คลายตัว)<br>• **Covariate Shift (`Train` vs `Test`):** ชุด `Test` (7–30 ก.ย.) มีชั่วโมงฝนตกพุ่งถึง **`13.34%`** เพิ่มขึ้น **`+75.3%`** จากชุด `Train` (`7.61%`) และฝนสะสม 3 ชม. เพิ่มเกือบ 2 เท่า (`1.74` vs `0.89 mm`) |

---

## ส่วนที่ 2: วิเคราะห์ฟีเจอร์ทั้งหมดที่เราใช้ (Complete Feature Audit — 58 คอลัมน์ใน `gold.merged_segment_15m`)

จากการตรวจสอบ Schema ทั้ง 58 คอลัมน์ แบ่งออกเป็น **7 กลุ่มหลัก** พร้อมวิเคราะห์บทบาท ปัญหาที่พบ และวิธีแปลงฟีเจอร์ก่อนเข้าโมเดล:

### กลุ่มที่ 1: Identifiers, Timestamps & Targets (10 คอลัมน์ — ห้ามนำ Target เข้าเป็น Feature!)
| คอลัมน์ | ประเภท | บทบาทและการวิเคราะห์จาก EDA |
| :--- | :---: | :--- |
| `seg_id`, `from_loc`, `to_loc` | ID | รหัสช่วงถนน (`4,068` ช่วง) — **พบ Self-Loop `233` ช่วง (`1,296,868` แถว = `5.44%`) ที่ `from_loc == to_loc` ต้องกรองทิ้งก่อนเทรน!** |
| `ts`, `date`, `event_time`, `available_time` | Time | ใช้คุม Zero-Leakage (`available_time = event_time + 15m`) และจัดเรียงลำดับเวลา |
| `target_ts`, `target_speed_kmh`, `target_ratio`, `target_congested`, `new_onset` | **Targets (+1h)** | เป้าหมายล่วงหน้า 1 ชม. — **มีค่า `NULL` `27.3%` (`6.50M` แถว)** เนื่องจากไม่มีรถ probe ในอีก 1 ชม. ต้องกรอง `target_congested IS NOT NULL` เหลือ **`17,336,060` แถวที่เทรนได้จริง** |

### กลุ่มที่ 2: Real-Time Traffic State & Short-Term Momentum (7 คอลัมน์ — พลังทำนายสูงสุด)
| คอลัมน์ | การกระจายตัว / สถิติ | การวิเคราะห์และการใช้งานต่อ |
| :--- | :--- | :--- |
| `speed_kmh` | Mean `38.0`, SD `20.9`, Skew `+0.87` | ความเร็วปัจจุบัน ณ $t_0$ — สัมพันธ์กับ $t+1\text{h}$ สูงที่สุด |
| `free_flow_kmh` | Mean `55.0`, Median `54.0` | ความเร็วอ้างอิงรถโล่งของแต่ละถนน (คำนวณจาก Train เท่านั้น ไม่รั่ว) |
| `ratio` | Mean `0.682`, P25 `0.500`, Skew `+0.84` | ตัวชี้วัดที่ปรับมาตรฐานข้ามประเภทถนนแล้ว (ซอย vs ทางด่วนเทียบกันได้) |
| `congestion_level` | `{1, 2, 3}` (Red = `32.3%`) | ระดับสีจาก iTIC (เขียว/เหลือง/แดง) |
| `is_congested` | Binary (`23.1%` True) | สถานะติดขัดปัจจุบัน (`ratio < 0.5`) — ถ้าใช้เดาอย่างเดียวจะได้ `0%` Recall ตอนเกิด `new_onset` |
| `prev_15m_speed_kmh` | Lag -15 นาที | ใช้สร้างความเร็วเปลี่ยน (`delta_speed_15m = speed_kmh - prev_15m_speed_kmh`) เพื่อจับจังหวะรถเริ่มชะลอตัวก่อนติด! |
| `n_obs` | จำนวนรถ probe | บอกความน่าเชื่อถือของค่าความเร็ว (ถ้าน้อยมาก เช่น 1–2 คัน ค่าความเร็วจะแกว่งสูง) |

### กลุ่มที่ 3: Spatial Coordinates, Road Geometry & Network Topology (10 คอลัมน์)
| คอลัมน์ | ข้อค้นพบ / จุดบอดจาก EDA | การจัดการก่อนเข้าโมเดล |
| :--- | :--- | :--- |
| `district`, `weather_point` | 50 เขต / 3 สถานี (`central`, `north`, `east`) | ใช้ทำ Categorical Encoding หรือ Target Encoding (และสร้าง `urban_ring_zone`: Inner/Middle/Outer) |
| `inbound_outbound` | `inbound` (`1,868`) / `outbound` (`2,200`) | มีความต่าง `5–6 pp` ระหว่างเช้า–เย็น ควรใช้ร่วมกับ `hour` (Interaction) |
| `road_name_th`, `road_name_en` | ชื่อสายทาง | ใช้สำหรับ Visualization/Explainability (หรือ Group Encoding) |
| `lat`, `lon` | พิกัดจุดกึ่งกลาง | บอกตำแหน่งเชิงพื้นที่และระยะห่างจาก CBD (`dist_to_cbd_km`) |
| `length_m` | Median `518m`, Max `10.2km`, **Skew `+3.65`** | เป็นระยะเส้นตรง Haversine (ไม่ใช่เส้นโค้งจริง) และเบ้ขวาจัด $\rightarrow$ **แนะนำแปลงด้วย `log1p(length_m)`** (`Skew -> +0.13`) |
| `n_downstream`, `downstream_match_type`, `downstream_seg_ids` | `3+` ชุมทางติดหนักสุด (`26.2%`), แต่ `33.9%` แมตช์ด้วยรัศมี `400m` | ใช้ `downstream_seg_ids` ไปดึง **ความเร็ว/สถานะรถติดของถนนปลายน้ำ ณ $t_0$ (`downstream_ratio_t0`)** เพื่อจับการล้นแถวคอย (Spillback)! |

### กลุ่มที่ 4: Meteorological & Accumulated Rain (6 คอลัมน์)
| คอลัมน์ | การกระจายตัว / สถิติ | การวิเคราะห์และการใช้งานต่อ |
| :--- | :--- | :--- |
| `precip_mm` | ศูนย์ `64.0%`, Skew `+5.64` | ฝนชั่วโมงปัจจุบัน ($r = +0.12$ กับรถติด) — สัญญาณอ่อนกว่าฝนสะสม |
| `precip_3h_mm` | ศูนย์ `48.6%`, Skew `+3.53` | **ฟีเจอร์ฝนที่ทรงพลังที่สุด (`r = +0.17`, แรงกว่า 1 ชม. `+44%`)** แนะนำทำ `log1p` ร่วมด้วย |
| `precip_6h_mm` | ศูนย์ `34.0%`, Skew `+2.58` | **ตัวจับจุดอิ่มตัวท่อระบายน้ำ (`>= 15 mm` น้ำท่วมพุ่ง 7.1 เท่า)** แนะนำทำ `log1p` |
| `rain_bin`, `is_rainy_gt_1mm` | Categorical / Binary (`9.1%` True) | ใช้ได้ดีกับโมเดลเชิงเส้นและทำ Interaction กับ `is_rush_hour` |
| `temp_c` | Mean `27.6°C`, Skew `+0.72` | สะท้อนจังหวะกลางวัน/แดดออก vs ฝนตกอากาศเย็น |

### กลุ่มที่ 5: Traffy Fondue Real-Time & Static District Risk (7 คอลัมน์)
| คอลัมน์ | การกระจายตัว / จุดบอดจาก EDA | การจัดการก่อนเข้าโมเดล |
| :--- | :--- | :--- |
| `traffy_flood_15m` | ศูนย์ `98.1%`, Skew `+15.6` | **สัญญาณเรียลไทม์ที่แรงมาก!** (`>=2 เรื่อง/15น.` ดันรถติดพุ่ง `+80.4%`) ควรทำ Rolling 1h/3h (`traffy_flood_last_3h`) เพื่อลดความถี่ของค่าศูนย์ |
| `traffy_traffic_15m`, `traffy_road_15m`, `traffy_total_15m` | ศูนย์ `82–95%`, Skew `+8.3` ถึง `+16.5` | สัญญาณเหตุฉุกเฉิน/ถนนชำรุดสด แนะนำทำ Rolling 1h/3h + `log1p` |
| `train_flood_reports`, `train_flood_per_sq_km`, `train_traffic_per_sq_km` | คำนวณจาก Train เท่านั้น (Zero-Leakage) | **จุดบอด Area Bias:** `train_flood_per_sq_km` กดเขตพื้นที่ใหญ่ (บางขุนเทียน, ลาดกระบัง) ให้ดูเสี่ยงต่ำ $\rightarrow$ **ควรสร้าง `train_flood_per_road_km` เพิ่ม** และทำ `log1p` |

### กลุ่มที่ 6: Temporal & Calendar Behavior (11 คอลัมน์)
| คอลัมน์ | การวิเคราะห์จาก EDA | ข้อเสนอแนะ |
| :--- | :--- | :--- |
| `hour`, `minute`, `dow` | จังหวะพีค `07:45` และ `17:45` ชัดเจนมาก | แปลงเป็น **Cyclical Sin/Cos** (`sin_time`, `cos_time`, `sin_dow`, `cos_dow`) เพื่อให้ 23:45 เชื่อมต่อกับ 00:00 |
| `day_type`, `is_weekend`, `is_holiday`, `is_bank_holiday`, `is_working_day` | วันทำงานติดกว่าวันหยุด `+44%` | ฟีเจอร์หลักที่ขาดไม่ได้ (ต้องแยก `is_holiday` ออกจาก `dow` เสมอ) |
| `is_friday`, `is_payday` | **ศุกร์ในสัปดาห์ Payday พุ่งสูงสุด `44.8%`** | แยก `is_month_end_workday` ออกจาก `is_payday_25` (เพราะวันที่ 25 แทบไม่ต่างจากวันปกติ) |
| **`is_school_term`** | **⚠️ ค่าคงที่ (`100% = True`) ตลอด มิ.ย.–ก.ย.** | **ตัดทิ้ง (Drop Feature!)** เพราะไม่มี Variance เลยในช่วงข้อมูลนี้ (เก็บไว้จะเปลืองมิติเปล่าๆ) |

### กลุ่มที่ 7: Data Quality Flags & Split Governance (4 คอลัมน์)
* `flag_ratio_gt_1_2` (`2.94%`), `flag_jump_gt_60` (`0.57%`), `flag_incomplete_day` (`77.23%`), `split` (`train` / `valid` / `test`)
* ใช้สำหรับ **Data Filtering / Sample Weighting / Monitoring** ไม่ใช่นำไปเป็นฟีเจอร์พยากรณ์โดยตรง (ยกเว้น `flag_incomplete_day` ใช้เป็นเกณฑ์คัดกรองคุณภาพตอนเทรน)

---

## ส่วนที่ 3: สิ่งที่เราควรทำเพิ่มจากจุดนี้ (Actionable Gaps & Next Engineering Steps)

### 3.1 การคลีนข้อมูลที่ "ต้องทำทันที" ก่อนเข้าสู่ขั้นตอน Modeling (Data Hygiene Fixes)
1. **ตัด Self-Loop ทิ้ง (`WHERE from_loc != to_loc`):** กำจัดข้อมูลขยะ `233` ช่วงถนน (`1,296,868` แถว) ที่วัดระยะได้ `0 เมตร`
2. **แก้บั๊ก Sensor Outage Zero-Fill ในตารางระดับเขต (`gold.merged_district_15m`):** เปลี่ยนค่า `itic_pct_congested = 0.0` ใน `35,619` สล็อตที่ `itic_active_segments = 0` ให้เป็น **`NULL`** (เพื่อไม่ให้วันเซ็นเซอร์ล่มอย่าง 20–21 ก.ค. ถูกดึงค่าเฉลี่ยลง)
3. **กรอง Target ที่สมบูรณ์ (`WHERE target_congested IS NOT NULL`):** ใช้เฉพาะ `17.34M` แถวที่มีเฉลยจริงใน +1 ชม. และพิจารณา Clip/Filter สล็อตที่ `flag_jump_gt_60 = TRUE` (`0.57%`) ออกตอนเทรน
4. **ตัดฟีเจอร์ค่าคงที่ (`is_school_term`) ออก** เพราะเป็น `True` 100% ตลอด 4 เดือน

### 3.2 ฟีเจอร์ใหม่ที่ EDA พิสูจน์แล้วว่า "ควรสร้างเพิ่ม" (High-Impact Feature Engineering)
| ฟีเจอร์ใหม่ที่แนะนำ | สูตร / วิธีสร้าง | เหตุผลเชิงหลักฐานจาก EDA |
| :--- | :--- | :--- |
| **`downstream_ratio_t0`** | ค่าเฉลี่ย `ratio` ณ $t_0$ ของถนนปลายน้ำใน `downstream_seg_ids` | จับการล้นแถวคอยย้อนกลับ (Spatial Spillback) ก่อนที่ถนนเส้นนี้จะติดในอีก 1 ชม. |
| **`speed_lag_30m`, `60m`, `24h`, `7d`** & **`delta_speed_15m/30m`** | `speed(t) - speed(t-15m)`, `speed(t - 24h)` | จับโมเมนตัมการชะลอตัว (Early Onset Signal) และรูปแบบประจำเวลาเดียวกันของเมื่อวาน/สัปดาห์ก่อน |
| **`hist_slot_avg_ratio`** | ค่าเฉลี่ย `ratio` ของ `(seg_id, day_type, hour, minute)` คำนวณจาก **Train เท่านั้น** | เป็น Baseline ประจำจุดถนน+ช่วงเวลาที่แข็งแรงมาก (ลด Error ตอนรถกำลังจะติด) |
| **`is_pre_holiday`** | `1` ถ้าพรุ่งนี้เป็นวันหยุดยาว (`>= 3 วัน`) | EDA พบว่าวันก่อนหยุดยาว (11 ส.ค., 2 มิ.ย., 27 ก.ค.) ติดอันดับ Top 5 วันที่รถติดหนักที่สุดของปี |
| **`is_friday_payday_week`** | `is_friday AND is_payday_week` | EDA พบว่าเย็นศุกร์ในสัปดาห์เงินเดือนออกพุ่งสูงสุดถึง **`44.8%`** (สูงกว่าทุกวันในสัปดาห์) |
| **`urban_ring_zone`** | จัดกลุ่ม 50 เขตเป็น `Inner (21)` / `Middle (18)` / `Outer (11)` | แยกโครงสร้างความติดขัด (`21.6%` vs `16.9%` vs `13.2%`) และโซนน้ำท่วม (`Middle 52.7%`) |
| **`rain_x_flood_interaction`** | `log1p(precip_3h_mm) * log1p(train_flood_per_road_km)` | จับแรงกระแทกทวีคูณเมื่อฝนสะสมปะทะเขตเสี่ยงน้ำท่วม (พร้อมแก้ Area Bias ด้วย `road_km`) |
| **`traffy_flood_last_3h`** | ผลรวม `traffy_flood_15m` ย้อนหลัง 3 ชม. ในเขต | ลดปัญหาค่าศูนย์ `98.1%` ของสล็อต 15 นาที และสะท้อนภาวะน้ำท่วมขังต่อเนื่อง |

---

## ส่วนที่ 4: เราจะเอา Data นี้ไปทำอะไรต่อ? (End-to-End Project Roadmap)

เมื่อจบเฟส **EDA (`feature/eda`)** แล้ว ข้อมูลใน `bkk_traffic.duckdb` พร้อมถูกนำไปต่อยอดใน **4 ขั้นตอนหลัก** เพื่อสร้างระบบพยากรณ์และเครื่องมือช่วยตัดสินใจจริง:

### ระยะที่ 1: กำหนดโจทย์พยากรณ์ (Problem Formulation — 1-Hour-Ahead Forecasting)
* **Task A (Primary — Binary Classification):** ทำนาย **`target_congested`** (ถนนช่วงนี้จะติดขัด `ratio < 0.5` ในอีก 1 ชั่วโมงข้างหน้าหรือไม่?) โดยเน้นตัวชี้วัด **F1-Score** และ **Onset Recall** (ความแม่นยำในการเตือนล่วงหน้าสำหรับ `2.17M` เหตุการณ์ที่รถกำลังจะพลิกจากโล่งเป็นติด!)
* **Task B (Secondary — Regression):** ทำนาย **`target_ratio`** หรือ **`target_speed_kmh`** ล่วงหน้า 1 ชั่วโมง (วัดผลด้วย **MAE / RMSE**) เพื่อประเมินความเร็วและเวลาเดินทางจริง

### ระยะที่ 2: สร้าง Feature Pipeline & เปรียบเทียบโมเดล (Modeling & Ablation Study)
* **Out-of-Time Validation (ห้าม Random Split!):**
  * `Train`: 1 มิ.ย. – 23 ส.ค. (`16.41M` แถว)
  * `Valid`: 24 ส.ค. – 6 ก.ย. (`2.81M` แถว) — ใช้จูน Hyperparameters & Decision Threshold
  * `Test`: 7 ก.ย. – 30 ก.ย. (`4.62M` แถว) — ทดสอบจริงในฤดูมรสุมที่ฝนหนักขึ้น `+75%`
* **ลำดับขั้นโมเดล (Model Benchmark Ladder):**
  1. **Baseline B0 (Persistence):** ทายว่าอีก 1 ชม. = สภาพตอนนี้ *(รู้จาก EDA แล้วว่าจะได้ `0%` Onset Recall และพลาด `25.2%` ของทุกช่วงเวลาที่เปลี่ยนสถานะ)*
  2. **Baseline B1 (Historical Slot Profile):** ทายด้วยค่าเฉลี่ยประจำ `(seg_id, day_type, hour, minute)` *(จับจังหวะประจำวันได้ แต่แพ้ตอนฝนตก/น้ำท่วม/เหตุฉุกเฉิน)*
  3. **Model B2 (Gradient Boosted Decision Trees — LightGBM / CatBoost / XGBoost):** เหมาะที่สุดกับข้อมูล Tabular ขนาดใหญ่ (`17.34M` แถว) ที่มี Zero-Inflated Features (`precip`, `traffy`), Non-Linear Thresholds (`precip_6h >= 15mm`), และตัวแปร Categorical จำนวนมาก — พร้อมใช้ **Sample Weighting** เน้นช่วง `new_onset` และช่วงฝนตกเพื่อสู้ Covariate Shift เดือน ก.ย.
  4. **Ablation Study & Explainability (TreeSHAP):** ทดลองตัดกลุ่มฟีเจอร์ออกทีละกลุ่ม (เช่น ตัด Weather, ตัด Traffy, ตัด Downstream Graph, ตัด Calendar) เพื่อพิสูจน์เชิงปริมาณว่าแต่ละกลุ่มข้อมูลช่วยเพิ่ม F1 / Onset Recall กี่เปอร์เซ็นต์ และใช้ **SHAP** อธิบายสาเหตุรถติดรายจุด

### ระยะที่ 3: การออกแบบระบบ MLOps & การเฝ้าระวัง (Monitoring & Drift Guardrails)
* เนื่องจาก EDA ค้นพบ **2 ความเสี่ยงหลักในระบบจริง** ได้แก่ (1) **iTIC Sensor Outage (`6.1%` สล็อตล่ม)** และ (2) **Seasonal Monsoon Covariate Shift (ฝนเดือน ก.ย. พุ่ง `+75%`)**
* ระบบ Production จึงควรมี:
  * **Circuit-Breaker:** ตรวจจับ `itic_active_segments == 0` หรือ `Completeness < 80%` แล้วเปลี่ยนเป็น `NULL` / ใช้ Fallback Profile แทนการส่งค่า `0%` เข้าโมเดล
  * **Drift & Performance Monitor:** ติดตาม PSI ของ `precip_3h_mm` และ Daily F1 / Onset Recall แยกวันทำงาน–วันหยุด–วันฝนตก

### ระยะที่ 4: การนำไปใช้ประโยชน์จริง (Real-World Downstream Use Cases)
1. **ระบบแจ้งเตือนผู้เดินทางล่วงหน้า 1 ชั่วโมง (Commuter Early-Warning App / Map Overlay):** แจ้งเตือนก่อนเกิด `new_onset` (เช่น เตือนตั้งแต่ 06:30 น. และ 16:30 น. โดยเฉพาะวันศุกร์สัปดาห์เงินเดือนออกและวันก่อนหยุดยาว) พร้อมแสดงเส้นทางเลี่ยงคอขวด
2. **ศูนย์เฝ้าระวังน้ำท่วม–จราจร กทม. (BMA Flood-Traffic Operations):** ใช้กฎเตือนภัยจาก EDA (**ฝนสะสม 6 ชม. `>= 15 mm`** + **แจ้งน้ำท่วม Traffy สด `>= 2 เรื่อง/15น.`**) สั่งการเครื่องสูบน้ำและตำรวจจราจรล่วงหน้าใน **โซนชั้นกลาง–ฝั่งตะวันออก** (`ประเวศ`, `บางกะปิ`, `สวนหลวง`, `วังทองหลาง`, `ลาดกระบัง`, `วัฒนา`)
3. **การวางผังเมืองและแก้คอขวดเชิงโครงสร้าง (Urban & Drainage Planning):** จัดลำดับความสำคัญการซ่อมผิวจราจรและลอกท่อระบายน้ำในเขตที่มีค่า `traffy_road` และ `flood_per_road_km` สูงสุด รวมถึงปรับจังหวะสัญญาณไฟใน Top 10 คอขวดเรื้อรัง (`ถนนรางน้ำ`, `ด่านประชาชื่น`, `อรุณอมรินทร์ 49`, `ถนนอาจณรงค์`)
