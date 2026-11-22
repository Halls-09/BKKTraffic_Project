# 📊 Bangkok Traffic Congestion — Seaborn EDA & Statistical Distribution Report

> **แหล่งข้อมูลหลัก:** [bkk_traffic.duckdb](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/data/bkk_traffic.duckdb) (`23,840,561` ระเบียนระดับ 15 นาที | `4,068` ช่วงถนน | `50` เขต กทม. | 1 มิ.ย. – 30 ก.ย. 2022)  
> **หน้าเว็บดูกราฟรวมทั้งหมด 12 รูป (คลิกเปิดในเบราว์เซอร์ได้เลย):** [eda_report.html](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/outputs/eda/eda_report.html)  
> **เอกสารวิเคราะห์จุดบอดและปัจจัยน้ำท่วมเชิงลึก:** [BLINDSPOT_ANALYSIS.md](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/BLINDSPOT_ANALYSIS.md)

---

## 1. ตารางสถิติพรรณนาและการกระจายตัวของตัวแปร (Statistical Distributions, Quantiles & Skewness)

| ตัวแปร (Variable) | ระดับข้อมูล | Mean | SD | P25 (Q1) | **Median (Q2)** | P75 (Q3) | P95 | **Skewness** | **% ค่าศูนย์ (Zero-Inflation)** | ลักษณะการกระจายตัว (Distribution Shape) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `speed_kmh` (ความเร็วรถจริง) | Segment 15m (`23.84M`) | `37.97` | `20.85` | `21.50` | **`36.00`** | `50.00` | `77.00` | **`+0.87`** | `0.0%` | เบ้ขวาปานกลาง (IQR = `28.5 km/h`) |
| `free_flow_kmh` (ความเร็วอ้างอิง) | Segment 15m (`23.84M`) | `55.00` | `18.52` | `42.00` | **`54.00`** | `67.00` | `88.00` | `+0.41` | `0.0%` | ใกล้เคียงโค้งปกติ (Normal-like) |
| `ratio` (`speed / free_flow`) | Segment 15m (`23.84M`) | `0.682` | `0.279` | **`0.500`** | **`0.684`** | `0.844` | `1.077` | **`+0.84`** | `0.0%` | **P25 = `0.500` ตรงเกณฑ์รถติดพอดี** (Outlier `>1.2` = `2.94%`) |
| `itic_pct_congested` (% รถติดเขต) | District 15m (`585.6k`) | `19.24%` | `15.82%` | `7.02%` | **`15.79%`** | `28.57%` | `50.00%` | **`+0.95`** | `12.42%` | เบ้ขวา (ช่วงดึกรถโล่งเป็นศูนย์ 12.4%) |
| `precip_mm` (ฝนรายชั่วโมง) | District 15m (`585.6k`) | `0.37` | `1.19` | `0.00` | **`0.00`** | `0.20` | `2.00` | **`+5.64`** 🔥 | **`64.03%`** | **Zero-Inflated & Heavy-Tailed** (`90.9% <= 1 mm`) |
| `precip_3h_mm` (ฝนสะสม 3 ชม.) | District 15m (`585.6k`) | `1.13` | `2.53` | `0.00` | **`0.10`** | `0.90` | `6.60` | **`+3.53`** | `48.61%` | การทำ Rolling 3 ชม. ลดความเบ้จาก `+5.64` เหลือ `+3.53` |
| `precip_6h_mm` (ฝนสะสม 6 ชม.) | District 15m (`585.6k`) | `2.29` | `3.94` | `0.00` | **`0.40`** | `2.80` | `11.00` | **`+2.58`** | `33.98%` | ลดความเบ้ลงกว่าครึ่ง เหมาะใช้เป็นฟีเจอร์น้ำรอระบาย |
| `traffy_total_15m` (แจ้งรวม 15น.) | District 15m (`585.6k`) | `0.26` | `0.74` | `0.00` | **`0.00`** | `0.00` | `1.00` | **`+16.49`** 🔥 | **`82.42%`** | เบ้ขวาจัดแบบ Power-Law / Poisson (Max = `123` เรื่อง/15น.) |
| `traffy_flood_15m` (แจ้งน้ำท่วม 15น.) | District 15m (`585.6k`) | `0.03` | `0.22` | `0.00` | **`0.00`** | `0.00` | `0.00` | **`+15.60`** 🔥 | **`98.12%`** | เกิดเฉพาะช่วงพายุฝนหนัก (Max = `13` เรื่อง/15น.) |
| `area_sq_km` (ขนาดพื้นที่เขต) | District Static (`50 เขต`) | `31.71` | `41.07` | `9.22` | **`19.18`** | `35.86` | `122.21` | **`+3.24`** 🔥 | `0.0%` | เบ้ขวาสูงมาก (`1.39` ถึง `240.45 km²` ต่างกัน **173 เท่า**) |

---

## 2. วิธีรันง่ายๆ ใน 1 คำสั่ง

```powershell
# รันทั้งหมด (12 กราฟ Seaborn + หน้าเว็บดูกราฟรวม HTML + สรุป JSON)
python eda.py

# หรือเลือกรันเฉพาะบางกราฟ (เร็ว ~2 วินาที)
python eda.py 9 10 11 12

# ดูรายการกราฟทั้งหมด
python eda.py --list
```

---

## 3. สรุปกราฟทั้ง 12 รูป (8 กราฟ Domain/Spatial + 4 กราฟ Statistical Distributions)

### หมวดที่ 1: เชิงพื้นที่ เวลา สภาพอากาศ และปฏิทิน (Charts 01–08)
1. **[01_bkk_map_baseline.png](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/outputs/eda/charts/01_bkk_map_baseline.png)** *(Baseline Map)* — แผนที่ 4,068 ช่วงถนน พร้อมชื่อเขตสำคัญ แสดงจุดรถติดสะสมและพื้นที่อ่อนไหวต่อฝน (**84.8%** ของถนนแย่ลงเมื่อฝนตก, ค่ากลาง `+4.55 pp`, p90 `+13.09 pp`)
2. **[02_bkk_map_morning_vs_evening.png](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/outputs/eda/charts/02_bkk_map_morning_vs_evening.png)** — แผนที่เทียบชั่วโมงเร่งด่วนวันทำงาน **เช้า 07–09 น. (`34.2%`)** vs **เย็น 17–19 น. (`40.9%`)**
3. **[03_temporal_heatmap_and_15m.png](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/outputs/eda/charts/03_temporal_heatmap_and_15m.png)** — Seaborn Heatmap รายชั่วโมง × วัน (ไม่รวมวันหยุดนักขัตฤกษ์) + กราฟละเอียดทุก 15 นาที พบจุดพีคทั้งสัปดาห์ที่ **วันศุกร์ 17:45 น. (`49.0%`)**
4. **[04_inbound_outbound_tidal.png](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/outputs/eda/charts/04_inbound_outbound_tidal.png)** — การไหลแบบน้ำขึ้น–น้ำลงระหว่าง **ขาเข้า (Inbound)** vs **ขาออก (Outbound)** วันทำงาน
5. **[05_monsoon_timeline_122d.png](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/outputs/eda/charts/05_monsoon_timeline_122d.png)** — ไทม์ไลน์ 122 วัน (หลังตัดสล็อตเซ็นเซอร์ล่ม) เทียบ รถติด vs ฝนสะสม vs แจ้งน้ำท่วม Traffy Fondue
6. **[06_top_districts_and_bottlenecks.png](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/outputs/eda/charts/06_top_districts_and_bottlenecks.png)** — เทียบ 12 เขตติดหนักสุด (**คลองเตย `32.0%`**, **ราชเทวี `31.3%`**) กับ 3 เขตคล่องตัวสุด (**หนองจอก `8.5%`**) และ Top 10 คอขวดเรื้อรัง
7. **[07_rain_and_traffy_flood_impact.png](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/outputs/eda/charts/07_rain_and_traffy_flood_impact.png)** — ห่วงโซ่ผลกระทบ: ความแรงฝน (แยกเร่งด่วน vs นอกเร่งด่วน) $\rightarrow$ เขตเสี่ยงน้ำท่วม $\rightarrow$ แจ้งน้ำท่วมสด Traffy 15 นาที
8. **[08_calendar_payday_and_correlation.png](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/outputs/eda/charts/08_calendar_payday_and_correlation.png)** — อันดับความติดขัดช่วงเย็น: **เย็นศุกร์ในสัปดาห์เงินเดือนออก (`44.8%`)** > วันทำงานสิ้นเดือน (`41.8%`) > เย็นศุกร์ทั่วไป (`41.0%`) > วันทำงานปกติ (`37.4%`) พร้อมเมทริกซ์สหสัมพันธ์ 10 ปัจจัยหลัก

### หมวดที่ 2: การกระจายตัวทางสถิติของแต่ละฟีเจอร์, Log Transform, Outliers และ Distribution Shift (Charts 09–11)
9. **[09_statistical_distributions_and_skewness.png](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/outputs/eda/charts/09_statistical_distributions_and_skewness.png)** — **พล็อตการกระจายตัวแยกครบทั้ง 12 ฟีเจอร์หลัก (4×3 Grid: Histogram + KDE + Mean/Median + Skewness):**
   * **แถว 1–2 (6 ฟีเจอร์สเกลปกติ):** `speed_kmh` (`Skew +0.87`), `free_flow_kmh` (`Skew +0.41`), `Speed Ratio` (`Skew +0.84`), `% Congested` (`Skew +0.95`), `dist_speed` (`Skew +0.36`), `temp_c` (`Skew +0.72`)
   * **แถว 3–4 (6 ฟีเจอร์เบ้ขวาจัดที่ผ่านการทำ Log Transform `log(1 + x)`):** แสดงการลดลงของค่าความเบ้ (`Raw Skew` $\rightarrow$ `Log1p Skew`) อย่างชัดเจน ได้แก่:
     * `length_m` (ความยาวช่วงถนน): `Raw Skew +3.41` $\rightarrow$ **`Log1p Skew -0.58`**
     * `area_sq_km` (ขนาดพื้นที่เขต): `Raw Skew +3.28` $\rightarrow$ **`Log1p Skew +0.32`**
     * `train_flood_per_sq_km` (ความหนาแน่นน้ำท่วมเขต): `Raw Skew +1.03` $\rightarrow$ **`Log1p Skew -0.21`**
     * `train_traffic_per_sq_km` (ความหนาแน่นจราจรเขต): `Raw Skew +1.98` $\rightarrow$ **`Log1p Skew +0.35`**
     * `precip_3h_mm` (ฝนสะสม 3 ชม.): `Raw Skew +2.47` $\rightarrow$ **`Log1p Skew +0.92`**
     * `precip_6h_mm` (ฝนสะสม 6 ชม.): `Raw Skew +2.04` $\rightarrow$ **`Log1p Skew +0.52`**
10. **[10_flood_factors_and_area_bias.png](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/outputs/eda/charts/10_flood_factors_and_area_bias.png)** — **Seaborn Regplot** ความสัมพันธ์เชิงเส้นระหว่างปัญหาโครงสร้างถนน (`traffy_road`) กับปริมาณน้ำท่วม (`traffy_flood`, $r = +0.67$) และ **`Area Denominator Bias`** ที่ทำให้เขตใหญ่ (`ลาดกระบัง 831 เรื่อง`, `บางขุนเทียน 645 เรื่อง`) ถูกจัดเป็น `Low Flood Risk`
11. **[11_data_quality_outliers_and_split_shift.png](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/outputs/eda/charts/11_data_quality_outliers_and_split_shift.png)** — ฮิสโตแกรม + KDE ของความครบถ้วนข้อมูล (`Completeness` ช่วงเร่งด่วน vs ทั้งวัน), การกระจายแบบหางหนัก (Laplacian Heavy-Tailed) ของ `Speed Change (15m)` พร้อมขอบเขต Outlier `±60 km/h` (`0.57%`), และ **Distribution Shift (Train vs Valid vs Test)** ที่ชี้ว่าชุด Test เดือน ก.ย. มีชั่วโมงฝนตกพุ่งสูงขึ้น **`+75%`** จากชุด Train (`13.3%` vs `7.6%`)
12. **[12_root_cause_impact_drivers.png](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/outputs/eda/charts/12_root_cause_impact_drivers.png)** — **Executive Impact Panel: 6 กลไกต้นตอหลักที่ทำให้รถติดในกรุงเทพฯ** (1) กับดักหลังฝนหยุด/น้ำรอระบาย (`35.1 km/h`), (2) โดมิโนแถวคอยล้นแยก (`+79.3%` Spillback Risk), (3) แยกถี่ `<500 ม.` ติดหนักกว่า `2.3x`, (4) กฎพาเรโต `Top 20% = 48.9%` ของเหตุรถติดทั้งเมือง, (5) วันก่อนหยุดยาว (`45.3%`), และ (6) Perfect Storm ซ้อนทับ (`44.8%`)

---

## 4. ไฟล์ผลลัพธ์ทั้งหมด
- [eda.py](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/eda.py) — คำสั่งรันสั้นจากหน้าหลักโปรเจกต์
- [run_eda.py](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/src/eda/run_eda.py) — โค้ด Seaborn + DuckDB หลัก (ครบ 12 กราฟ)
- [eda_report.html](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/outputs/eda/eda_report.html) — หน้าเว็บแกลเลอรีดูกราฟทั้ง 12 รูปพร้อมสรุป Insight
- [BLINDSPOT_ANALYSIS.md](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/BLINDSPOT_ANALYSIS.md) — รายงานวิเคราะห์จุดบอดของโปรเจกต์และปัจจัยน้ำท่วมเชิงลึก
- [eda_summary.json](file:///c:/Users/kafil/Desktop/KMUTT_Sarit/Project/DataSci/BKKTraffic_Project/outputs/eda/eda_summary.json) — ค่าสถิติและผลการทดสอบทั้งหมด
