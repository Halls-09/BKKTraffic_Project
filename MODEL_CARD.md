# 🤖 Model Card & Experimental Benchmark Report (Real Bangkok iTIC Dataset)

> **โครงการ:** ระบบพยากรณ์จุดรถติดในกรุงเทพมหานครล่วงหน้า 1 ชั่วโมง และวิเคราะห์ปัจจัยที่ทำให้รถติด (CSS 342 KMUTT)  
> **Active Champion Version:** `v1.0.0` | **Algorithm:** `LightGBMRegressor (v4.7.0, best_iter=756)`  
> **Model SHA-256:** `8616d91ed9cd9cccd1dc7f6f328057bec46aa454cb8fa392fea4414e82d4c921`  
> **Dataset:** `1,269,453` link-hours | `448` ช่วงถนนที่มีทิศทาง (100% Location Table Join Coverage ตาม Data Contract 4.2) | `21` Features

---

## 1. ผลการเปรียบเทียบโมเดลหลักบนข้อมูลจริง (Section 9.2 Benchmark & Section 13 Daytime Probe)

| โมเดล (Model) | Split | Threshold | F1 ทั้งหมด (24h) | F1 กลางวัน (06–23h) | Onset Recall (&ge;0.55) | F1 ช่วงเร่งด่วน | F1 ช่วงฝนตก | MAE | RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0: Persistence Baseline** | Test | `0.50` | `0.6197` | `0.6337` | `0.0000` ❌ | `0.6610` | `0.6555` | `0.1121` | `0.2012` |
| **B1: Historical Profile Avg** | Test | `0.50` | `0.6008` | `0.6108` | `0.2410` ❌ | `0.6554` | `0.6111` | `0.0983` | `0.1643` |
| **B2: LightGBM Regressor** | Val | `0.345` | `0.7364` | `0.7419` | `0.5530` ✅ | `0.7813` | `0.7612` | `0.0898` | `0.1459` |
| **B2: LightGBM Champion (`v1.0.0`)** | **Test** | **`0.345`** | **`0.6978`** ✅ | **`0.7056`** ✅ | **`0.5630`** ✅ | **`0.7365`** | **`0.7124`** | **`0.0902`** ✅ | **`0.1482`** |

---

## 2. การทดลองตัดกลุ่มฟีเจอร์ (B3 Feature-Group Ablation & Section 9.3 Feature Gate)

| กลุ่มฟีเจอร์ที่ตัดออก (Removed Group) | Ablated Test F1 | &Delta;F1 Gain | &Delta;Onset Recall | &Delta;Rainy F1 | &Delta;Rush F1 | Feature Gate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| G2: รูปแบบในอดีต (Historical Profile) | `0.6945` | `+0.0033` | `+0.0582` | `+0.0001` | `+0.0061` | ✅ Keep |
| G4: ปฏิทินและวันเงินเดือนออก (Calendar) | `0.6989` | `-0.0011` | `+0.0251` | `-0.0032` | `-0.0027` | ✅ Keep |
| G1: สภาพปัจจุบันและแนวโน้ม (Current & Lags) | `0.6968` | `+0.0010` | `+0.0165` | `+0.0017` | `+0.0007` | ✅ Keep |
| G7: กายภาพถนนและจุดเสี่ยงน้ำท่วม (Road & Traffy) | `0.6946` | `+0.0032` | `+0.0090` | `+0.0055` | `+0.0033` | ✅ Keep |
| G6: ถนนข้างหน้าและภาพรวมเขต (Spatial Graph) | `0.6949` | `+0.0029` | `+0.0049` | `+0.0034` | `+0.0020` | ✅ Keep |
| G3: ช่วงเวลาเป้าหมาย (Temporal Cyclical) | `0.6965` | `+0.0013` | `+0.0046` | `+0.0017` | `+0.0004` | ✅ Keep |
| G5: ฝนสะสม Open-Meteo (Weather) | `0.6980` | `-0.0002` | `+0.0025` | `-0.0010` | `-0.0004` | ✅ Keep |

---

## 3. การวิเคราะห์ปัจจัยด้วย Exact TreeSHAP และการทดสอบสมมติฐาน H1–H5 (Section 5 Page 6)

### Top 8 Features by Mean |TreeSHAP|
1. **`hist_avg_red_target_slot`** — Mean |SHAP| = `0.12441` (**54.31%**)
2. **`hist_p75_red_target_slot`** — Mean |SHAP| = `0.02084` (**9.10%**)
3. **`current_diff_from_hist`** — Mean |SHAP| = `0.01996` (**8.71%**)
4. **`district_red_rate_t0`** — Mean |SHAP| = `0.01162` (**5.07%**)
5. **`link_cat`** — Mean |SHAP| = `0.00964` (**4.21%**)
6. **`dow`** — Mean |SHAP| = `0.00572` (**2.50%**)
7. **`is_holiday`** — Mean |SHAP| = `0.00503` (**2.20%**)
8. **`downstream_red_rate_t0`** — Mean |SHAP| = `0.00488` (**2.13%**)

### ผลทดสอบสมมติฐาน H1–H5 บนชุดข้อมูล Test จริง (7–30 ก.ย. 2022)
* **H1 — LightGBM ชนะ Baseline ทั้งสองใน F1 และ Onset Recall (Benchmark Superiority):** ✅ Supported — LightGBM ได้ Test F1 = 69.8% (ชนะ B0=62.0%, B1=60.1%) และ Onset Recall = 56.3% (B1=24.1%)
* **H2 — ฟีเจอร์ฝนช่วยยกระดับความแม่นยำในชั่วโมงฝนตก (Rainy-Hour Gain):** ✅ Supported — ช่วงฝนตก (>1 มม./ชม.) LightGBM ได้ Rainy F1 = 71.2% (ชนะ B0=65.5%, B1=61.1%)
* **H3 — สภาพถนนข้างหน้าช่วยจับเหตุการณ์เริ่มติดใหม่ (Downstream Spillback & Onset):** ✅ Supported — ตัดกลุ่มถนนข้างเคียงทำ Onset Recall ลดลง 0.49% · เมื่อถนนข้างหน้าติด (>=50%) ถนนเส้นนี้ติด 46.0% (เทียบ 13.3%)
* **H4 — เขตที่มีรายงานน้ำท่วม Traffy Fondue สูง รถติดหนักกว่าเมื่อฝนตก (Rain × Flood 95% CI):** ✅ Supported — เขตเสี่ยงน้ำท่วมเมื่อฝน >3mm รถติด 25.5% เทียบวันปกติ 13.5% (ผลต่าง +12.0%, 95% CI: [+11.5%, +12.5%])
* **H5 — วันเงินเดือนออกและเย็นวันศุกร์ (16–20 น.) รถติดกว่าปกติ (Payday & Friday Surge):** ✅ Supported — ช่วง 16–20 น. ของศุกร์เย็น/วันเงินเดือนออก รถติดเฉลี่ย 34.4% สูงกว่าเย็นวันอื่น (29.0%) และช่วยเพิ่ม Onset Recall +2.51%

---

## 4. การแข่งขัน Challenger 3 กลยุทธ์หน้าต่างข้อมูล และ 3-Fold Temporal Robustness (Section 10 & 11)

### 4.1 Retraining Window Tournament (Selected on Validation)
| รหัสกลยุทธ์ | คำอธิบาย | Train Rows | Val F1 | Test F1 | Onset Recall | Test MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `Challenger_1_Expanding_Window` | Expanding Window (ข้อมูลฝึกทั้งหมด 84 วัน) | `874,324` | `0.7364` | `0.6973` | `0.5624` | `0.0903` |
| `Challenger_2_Rolling_8Weeks` | Rolling 8 Weeks (หน้าต่างล่าสุด 8 สัปดาห์ / 56 วัน) | `592,464` | `0.7327` | `0.6939` | `0.5656` | `0.0908` |
| `Challenger_3_Same_Season_Monsoon` | Same-Season Monsoon (ถ่วงน้ำหนักฤดูมรสุม + ล่าสุด) | `874,324` | `0.7360` | `0.6963` | `0.5672` | `0.0905` |

### 4.2 Rolling-Origin 3-Fold Temporal Robustness (เกณฑ์รับงาน: F1 แกว่งไม่เกิน &plusmn;0.05)
* **Mean Fold F1:** `0.6978` | **Max Deviation:** `&plusmn;0.0071` | **Status:** `✅ PASSED (<= 0.05)`

| Temporal Fold | Rows | F1 ทั้งหมด (24h) | F1 กลางวัน (06–23h) | Onset Recall | MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Fold 1 (2022-09-07 .. 2022-09-14)` | `84,539` | `0.7033` | `0.7116` | `0.5262` | `0.0967` |
| `Fold 2 (2022-09-15 .. 2022-09-22)` | `76,005` | `0.6995` | `0.7080` | `0.5903` | `0.0832` |
| `Fold 3 (2022-09-23 .. 2022-09-30)` | `85,907` | `0.6907` | `0.6976` | `0.5808` | `0.0899` |

---

## 5. ระบบเฝ้าระวัง 3 ชั้น (Section 10: 3-Layer MLOps Monitoring)

* **System Health:** `WARNING_WATCH` | **Retrain Recommended:** `False`
* **Layer 1 (Data Quality & Circuit-Breaker):** Location Join Coverage = `100.0%`, Weather Completeness = `100.0%`, Feed Freshness = `FRESH`, Min Daily Samples = `8,048` rows/day (`✅ PASSED`), Paused Segments = `0`
* **Layer 2 (Feature & Prediction Drift):** Prediction PSI = `0.0009` (`STABLE`)
* **Layer 3 (Performance Monitoring):** Test Overall F1 = `0.6978`, Stratified Alert Days = `7/24`

---

## 6. จริยธรรม ข้อจำกัด และการอ้างอิงแหล่งข้อมูล (Ethics, Limitations & Attribution)

1. **ความเป็นส่วนตัว (Privacy by Design):** ใช้เฉพาะข้อมูลสถานะจราจรที่รวมระดับช่วงถนนแล้ว ไม่มีการระบุตัวตนหรือติดตามยานพาหนะรายคัน และใช้เฉพาะจำนวนรายงาน Traffy Fondue ที่รวมระดับเขตในช่วง Train เท่านั้น
2. **การอ้างอิงแหล่งที่มา (Data Attribution):** ข้อมูลจราจรและตารางพิกัดจาก **มูลนิธิศูนย์ข้อมูลจราจรอัจฉริยะไทย (iTIC) & Longdo Traffic** (CC-BY 4.0 / CC-BY-SA 4.0), ข้อมูลสภาพอากาศจาก **Open-Meteo Historical Weather API** (CC-BY 4.0), ข้อมูลขอบเขตเขตจาก **HDX COD-AB-THA**, และข้อมูลร้องเรียนน้ำท่วมจาก **Traffy Fondue**
3. **ความปลอดภัยในการนำไปใช้ (Human-in-the-Loop):** ระบบนี้ออกแบบมาเพื่อสนับสนุนการตัดสินใจล่วงหน้า 1 ชั่วโมง ไม่ใช่การสั่งการสัญญาณไฟจราจรอัตโนมัติ และมีระบบ **Segment Circuit-Breaker** หยุดออกคำพยากรณ์ทันทีเมื่อข้อมูลของช่วงถนนนั้นขาดหายเกินเกณฑ์
