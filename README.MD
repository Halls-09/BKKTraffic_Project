# 🚦 Bangkok Traffic Congestion — Exploratory Data Analysis (DuckDB + Seaborn)

> **วิชา:** CSS 342 Data Science and Data Engineering (KMUTT)  
> **Branch:** `feature/eda`  
> **ฐานข้อมูลหลัก:** `data/bkk_traffic.duckdb` (`23,840,561` ระเบียนระดับ 15 นาที | `4,068` ช่วงถนน | ครบทั้ง `50` เขต กทม. | 1 มิ.ย. – 30 ก.ย. 2022)

---

## 1. วิธีติดตั้งและรันใช้งาน (Quickstart — 3 ขั้นตอน)

```powershell
# 1. ติดตั้งไลบรารีที่จำเป็น
pip install -r requirements.txt

# 2. วางไฟล์ฐานข้อมูลไว้ที่ data/bkk_traffic.duckdb

# 3. รัน EDA สร้างกราฟทั้งหมด 11 รูป + หน้าเว็บดูกราฟรวม HTML + สรุป JSON
python eda.py
```

### คำสั่งเสริม
```powershell
python eda.py --list      # ดูรายการกราฟทั้งหมด (1–11)
python eda.py 1 9         # เลือกรันเฉพาะบางกราฟ (เร็ว ~2–3 วินาที)
```

---

## 2. โครงสร้างโปรเจกต์

```text
BKKTraffic_Project/
├── eda.py                      # จุดเริ่มต้นสำหรับรัน EDA (python eda.py)
├── requirements.txt            # แพ็กเกจ Python ที่ต้องใช้
├── pyproject.toml              # ข้อมูลโปรเจกต์
├── EDA_INSIGHTS_REPORT.md      # รายงานสรุป Insights & ตารางสถิติการกระจายตัว
├── BLINDSPOT_ANALYSIS.md       # รายงานวิเคราะห์จุดบอด & ปัจจัยน้ำท่วมเชิงลึก
├── data/
│   └── bkk_traffic.duckdb      # ฐานข้อมูลหลัก (ไม่ถูก commit ขึ้น Git)
├── src/
│   └── eda/
│       └── run_eda.py          # โค้ดวิเคราะห์ DuckDB SQL + Seaborn Visualization
└── outputs/
    └── eda/
        ├── eda_report.html     # หน้าเว็บแกลเลอรีเปิดดูกราฟครบ 11 รูปพร้อมสรุป
        ├── eda_summary.json    # ค่าสถิติและการทดสอบ Welch's t-test
        └── charts/             # ไฟล์รูปกราฟ PNG ความละเอียดสูง (01–11)
```

---

## 3. รายการกราฟทั้ง 11 รูป (`outputs/eda/charts/`)

| # | ไฟล์กราฟ | หัวข้อการวิเคราะห์ |
| :-: | :--- | :--- |
| 01 | `01_bkk_map_baseline.png` | แผนที่ฐาน กทม. (4,068 ช่วงถนน): จุดรถติดสะสม & ผลกระทบฝนตก |
| 02 | `02_bkk_map_morning_vs_evening.png` | แผนที่เปรียบเทียบชั่วโมงเร่งด่วนวันทำงาน: เช้า (07–09 น.) vs เย็น (17–19 น.) |
| 03 | `03_temporal_heatmap_and_15m.png` | Heatmap รายชั่วโมง × วัน & กราฟละเอียดทุก 15 นาที (ศุกร์แห่งชาติ 17:45 น.) |
| 04 | `04_inbound_outbound_tidal.png` | รูปแบบการไหลตามทิศทางวันทำงาน: ขาเข้าเมือง vs ขาออกเมือง |
| 05 | `05_monsoon_timeline_122d.png` | แนวโน้มรายวัน 122 วัน: รถติด vs ฝนสะสม vs แจ้งน้ำท่วม Traffy Fondue |
| 06 | `06_top_districts_and_bottlenecks.png` | 12 เขตที่รถติดที่สุด vs 3 เขตคล่องตัวสุด & Top 10 คอขวดเรื้อรัง |
| 07 | `07_rain_and_traffy_flood_impact.png` | ห่วงโซ่ผลกระทบ: ความแรงฝน (เร่งด่วน/นอกเร่งด่วน) $\rightarrow$ เขตเสี่ยงท่วม $\rightarrow$ Traffy สด |
| 08 | `08_calendar_payday_and_correlation.png` | พฤติกรรมช่วงเย็นตามปฏิทิน (ศุกร์สัปดาห์เงินเดือนออก) & สหสัมพันธ์ 10 ปัจจัยหลัก |
| 09 | `09_statistical_distributions_and_skewness.png` | การกระจายตัวแยกครบ 12 ฟีเจอร์ (Histogram + KDE) & ผลของ Log Transform (`log1p`) |
| 10 | `10_flood_factors_and_area_bias.png` | ปัจจัยน้ำท่วม (`traffy_road` vs `traffy_flood`, $r = +0.67$) & อคติขนาดพื้นที่เขต |
| 11 | `11_data_quality_outliers_and_split_shift.png` | ความครบถ้วนข้อมูล (`Completeness`), `Speed Change Outliers` & `Train/Valid/Test Shift` |
