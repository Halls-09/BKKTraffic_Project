# คู่มือส่งมอบข้อมูลรอบที่ 1 (Handoff 1) สำหรับงาน EDA

* **ผู้ส่งมอบ (Data Engineer):** ธนากร การดี (67090500409)
* **ผู้รับมอบ (Data Scientist):** อภิเชษฐ์ ไชยสัจ (67090500422)
* **เอกสารประกอบ:** พจนานุกรมข้อมูลครบทั้ง 17 ตารางที่ [docs/data_dictionary.md](data_dictionary.md) และสัญญาข้อตกลงข้อมูลที่ [contracts/data_contract.yaml](../contracts/data_contract.yaml)

---

## 1. สิ่งที่ส่งมอบให้ (Deliverables)

ข้อมูลทุกชุด (iTIC Traffic Status 122 วัน + iTIC Location Table + Open-Meteo ฝน 3 โซน + ปฏิทิน 2022 + Traffy Fondue 50 เขต) ผ่านการทำความสะอาด ตรวจ Quality Gate 5 ข้อ และยุบรวมเป็นหน่วยเวลา **ราย 15 นาที** เก็บไว้ครบทั้งหมดในฐานข้อมูล **DuckDB** ไฟล์เดียว:

* **ฐานข้อมูลหลัก:** `data/bkk_traffic.duckdb` (แบ่งเป็น 3 ชั้น: `bronze.*` $\rightarrow$ `silver.*` $\rightarrow$ `gold.*`)
* **ตารางหลัก 3 ตารางที่อภิเชษฐ์ต้องใช้ทำ EDA:**
  1. **`gold.merged_district_15m`** (`585,600` แถว = 50 เขต × 11,712 ช่องเวลา 15 นาที) — รวมความเร็วรถ iTIC + Traffy 15 นาที + ฝน + ปฏิทิน + ค่าสะสมช่วง Train *(ใช้ทำ EDA ภาพรวมระดับเขตและเวลาได้เร็วที่สุด)*
  2. **`gold.merged_segment_15m`** (`23,840,561` แถว = 4,068 ช่วงถนนที่มีทิศทาง × 15 นาที) — รวม `speed_kmh`, `free_flow_kmh` (P90 ช่วง Train), `ratio`, `is_congested` (`< 0.5`), `target_ratio` ที่ $t+60$ นาที, `new_onset`, ทิศขาเข้า/ออก (`inbound_outbound`), ถนนข้างหน้า (`downstream_seg_ids`), ฝน, Traffy และปฏิทิน
  3. **`silver.quality_completeness_seg_day`** (`481,506` แถว = ช่วงถนน × วัน) — สัดส่วนความครบถ้วนของข้อมูลรายวันและช่วงพีค 06:00–22:00 น. *(ใช้ทำ Missing Heatmap)*

*(สามารถเปิดคลิกดูตารางในเบราว์เซอร์ได้ทันทีด้วยคำสั่ง `python scripts/inspect_duckdb.py --ui` แล้วเปิด `http://localhost:8765`)*

---

## 2. โค้ดเริ่มต้นสำหรับโหลดข้อมูลใน Jupyter Notebook

```python
import duckdb
import pandas as pd

# เชื่อมต่อฐานข้อมูลในโหมดอ่านอย่างเดียว (read_only=True) ป้องกันการแก้ข้อมูลโดยไม่ตั้งใจ
con = duckdb.connect("data/bkk_traffic.duckdb", read_only=True)

# ตรวจสอบรายชื่อตารางทั้งหมดในชั้น Silver และ Gold
con.sql("""
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE table_schema IN ('silver', 'gold')
    ORDER BY 1, 2
""").show()
```

---

## 3. โค้ดตัวอย่างจับคู่กับโจทย์ EDA ทั้ง 6 ข้อใน Proposal (ข้อ 6.1 หน้า 8)

### ข้อ 1: Missing Heatmap ราย segment × วัน และสัดส่วนข้อมูลตามชั่วโมง (Probe กลางคืนบาง)
```python
# 1.1 ข้อมูลสำหรับทำ Missing Heatmap (segment x date)
df_comp = con.sql("""
    SELECT seg_id, date, completeness_all, completeness_peak, flag_incomplete_day
    FROM silver.quality_completeness_seg_day
""").df()

# 1.2 สัดส่วนจำนวนช่วงถนนที่มีข้อมูลตามชั่วโมง (00:00 - 23:00)
df_by_hour = con.sql("""
    SELECT hour, COUNT(*) AS n_rows, COUNT(DISTINCT seg_id) AS active_segments,
           ROUND(AVG(n_obs), 2) AS avg_obs_per_15m
    FROM gold.itic_seg15_targets
    GROUP BY hour
    ORDER BY hour
""").df()
```

### ข้อ 2: Congestion Ratio เฉลี่ยรายชั่วโมง แยกวันทำงาน/วันหยุด และแยกขาเข้า/ขาออกเมือง
```python
df_ratio_hour = con.sql("""
    SELECT
        hour,
        day_type,
        inbound_outbound,
        ROUND(AVG(ratio), 4) AS avg_ratio,
        ROUND(AVG(speed_kmh), 2) AS avg_speed_kmh,
        ROUND(AVG(CASE WHEN is_congested THEN 1.0 ELSE 0.0 END) * 100, 2) AS pct_congested
    FROM gold.merged_segment_15m
    GROUP BY hour, day_type, inbound_outbound
    ORDER BY day_type, inbound_outbound, hour
""").df()
```

### ข้อ 3: สัดส่วนถนนที่ติดตามระดับฝน (`rain_bin`: ไม่มีฝน, 0.5–2, 2–5, 5–10, 10–20, >20 มม./ชม.)
```python
df_rain_effect = con.sql("""
    SELECT
        rain_bin,
        COUNT(*) AS n_seg_slots,
        ROUND(AVG(ratio), 4) AS avg_ratio,
        ROUND(AVG(speed_kmh), 2) AS avg_speed_kmh,
        ROUND(AVG(CASE WHEN is_congested THEN 1.0 ELSE 0.0 END) * 100, 2) AS pct_congested
    FROM gold.merged_segment_15m
    GROUP BY rain_bin
    ORDER BY avg_ratio ASC
""").df()
```

### ข้อ 4: ความสัมพันธ์ระหว่างการติดของ "ถนนข้างหน้า" กับถนนเส้นนี้ที่ Lag 15–60 นาที (สมมุติฐาน H3)
```python
# ดึงความสัมพันธ์ระหว่าง ratio ของถนนข้างหน้า (downstream) ณ เวลา t-15, t-30, t-60 กับถนนเส้นนี้ ณ เวลา t
df_downstream_corr = con.sql("""
    WITH edges AS (
        SELECT seg_id, split_part(downstream_seg_ids, ',', 1) AS down_seg_id
        FROM silver.dim_segment
        WHERE n_downstream > 0
    )
    SELECT
        CORR(cur.ratio, d0.ratio)  AS corr_lag0,
        CORR(cur.ratio, d15.ratio) AS corr_lag15m,
        CORR(cur.ratio, d30.ratio) AS corr_lag30m,
        CORR(cur.ratio, d60.ratio) AS corr_lag60m
    FROM gold.itic_seg15_targets cur
    JOIN edges e ON cur.seg_id = e.seg_id
    LEFT JOIN gold.itic_seg15_targets d0  ON d0.seg_id  = e.down_seg_id AND d0.ts  = cur.ts
    LEFT JOIN gold.itic_seg15_targets d15 ON d15.seg_id = e.down_seg_id AND d15.ts = cur.ts - INTERVAL '15 minutes'
    LEFT JOIN gold.itic_seg15_targets d30 ON d30.seg_id = e.down_seg_id AND d30.ts = cur.ts - INTERVAL '30 minutes'
    LEFT JOIN gold.itic_seg15_targets d60 ON d60.seg_id = e.down_seg_id AND d60.ts = cur.ts - INTERVAL '60 minutes'
    WHERE cur.split = 'train'
""").df()
```

### ข้อ 5: Event Study — ความเร็วในวันที่มีรายงานน้ำท่วมสูงสุดจาก Traffy Fondue (21 ก.ค., 20 ก.ค., 11 ก.ย.) vs วันปกติ
```python
df_event_study = con.sql("""
    SELECT
        is_top_flood_day,
        hour,
        ROUND(AVG(itic_avg_speed_kmh), 2) AS avg_speed_kmh,
        ROUND(AVG(itic_avg_ratio), 4) AS avg_ratio,
        ROUND(AVG(itic_pct_congested) * 100, 2) AS pct_congested,
        SUM(traffy_flood_15m) AS total_flood_reports
    FROM gold.merged_district_15m
    WHERE itic_active_segments > 0
    GROUP BY is_top_flood_day, hour
    ORDER BY is_top_flood_day, hour
""").df()
```

### ข้อ 6: เทียบเขตที่มีรายงานจราจร/น้ำท่วมใน Traffy Fondue มาก กับเขตที่ความเร็วจาก iTIC ต่ำจริง (สมมุติฐาน H4)
```python
df_district_compare = con.sql("""
    SELECT
        district,
        MAX(train_traffic_reports) AS traffy_traffic_train,
        MAX(train_flood_reports) AS traffy_flood_train,
        ROUND(AVG(CASE WHEN is_rainy_gt_1mm THEN itic_avg_ratio END), 4) AS avg_ratio_rainy,
        ROUND(AVG(CASE WHEN NOT is_rainy_gt_1mm THEN itic_avg_ratio END), 4) AS avg_ratio_dry,
        ROUND(AVG(itic_pct_congested) * 100, 2) AS pct_congested_all
    FROM gold.merged_district_15m
    WHERE itic_active_segments > 0
    GROUP BY district
    ORDER BY avg_ratio_rainy ASC
""").df()
```

---

## 4. ข้อควรระวังและกฎควบคุม Data Leakage (Known Issues & Rules)

1. **ห้ามคำนวณ `free_flow_kmh` หรือสถิติ Traffy ระดับเขตใหม่จากข้อมูลทั้ง 4 เดือน:**
   * `free_flow_kmh` (Percentile 90) และ `train_flood_reports` ในชั้น Gold ถูกคำนวณไว้ **เฉพาะจากช่วง `split = 'train'` (`2022-06-01` ถึง `2022-08-23`) เท่านั้น** ตาม Proposal ข้อ 6.2
2. **ห้ามสุ่มแบ่งข้อมูล (Random Split):**
   * ใช้คอลัมน์ `split` (`train`, `valid`, `test`) ที่แบ่งตามลำดับเวลาไว้ให้แล้วเท่านั้น
3. **ธงตรวจคุณภาพข้อมูล (Quality Flags) ยังไม่ได้ลบแถวออก:**
   * `flag_incomplete_day`, `flag_ratio_gt_1_2` (`ratio > 1.2`) และ `flag_jump_gt_60` (ความเร็วกระโดดเกิน 60 กม./ชม. ใน 15 นาที) ถูกติดธงไว้ให้เพื่อใช้ทำ Error Analysis และตัดสินใจร่วมกันหลังจบ EDA
4. **ข้อจำกัดของ Traffy Fondue:**
   * เป็นข้อมูลเรื่องร้องเรียน ไม่ใช่การวัดความเร็วจริง (28.4% ไม่ระบุหมวด และยอดแจ้งเรื่องจราจรลดลงตามความนิยมแพลตฟอร์มจาก มิ.ย. $\rightarrow$ ก.ย.) จึงใช้เป็นค่าสะสมระดับพื้นที่ (`train_flood_reports`, `train_flood_per_sq_km`) ตามข้อสรุปใน Proposal หน้า 7
5. **Governance (ข้อ 4.1):**
   * ข้อมูลข้อความร้องเรียน ที่อยู่ และรูปภาพของ Traffy Fondue ถูกตัดออกทั้งหมดแล้ว
   * ทุกครั้งที่นำเสนอกราฟหรือ Dashboard ต้องระบุแหล่งที่มา: **มูลนิธิ iTIC (CC-BY 4.0)**, **Open-Meteo (CC-BY 4.0)** และ **NECTEC Traffy Fondue**
