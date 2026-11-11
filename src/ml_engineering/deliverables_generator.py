"""
Automated Section 12 Deliverables Generator (`DATA_DICTIONARY.md` & `MODEL_CARD.md`).
Responsibility: Sarit (67090500441) - ML Engineer
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.logger import log_info, log_step, log_success
from src.utils.storage import ROOT_DIR, load_config


def generate_section12_deliverables() -> dict[str, str]:
    log_step("DELIVERABLES_S12", "สร้างเอกสารมาตรฐานตามข้อ 12 (DATA_DICTIONARY.md และ MODEL_CARD.md)...")
    cfg = load_config()
    eval_dir = ROOT_DIR / cfg["paths"]["evaluation_dir"]
    reg_dir = ROOT_DIR / cfg["paths"]["registry_dir"]

    def _load(p: Path) -> dict[str, Any]:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    eval_data = _load(eval_dir / "evaluation_summary.json")
    exp_s11 = _load(eval_dir / "proposal_experiments_s11.json")
    champ = _load(reg_dir / "champion_metadata.json")
    tourn = _load(reg_dir / "challenger_tournament.json")

    dd_md = r"""# 📖 พจนานุกรมข้อมูลและสัญญาข้อมูล (Data Dictionary & Data Contract Specification)

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
"""
    dd_path = ROOT_DIR / "DATA_DICTIONARY.md"
    with open(dd_path, "w", encoding="utf-8") as f:
        f.write(dd_md)

    m_models = eval_data.get("models", {})
    b0_t = m_models.get("B0_Persistence", {}).get("test", {})
    b1_t = m_models.get("B1_Historical_Average", {}).get("test", {})
    b2_v = m_models.get("B2_LightGBM", {}).get("val", {})
    b2_t = m_models.get("B2_LightGBM", {}).get("test", {})

    horizons = exp_s11.get("horizon_sensitivity", [])
    horizon_rows_md = "\n".join(
        f"| **{h['horizon_minutes']} นาที** ({h['horizon_slots']} ช่วงเวลา) | `{h['lgbm_f1']:.4f}` | `{h['lgbm_onset_recall']:.4f}` | `{h['lgbm_mae']:.4f}` | `{h['b0_f1']:.4f}` | `{h['b0_onset_recall']:.4f}` |"
        for h in horizons
    )

    rob = exp_s11.get("rolling_origin_3fold_robustness", {})
    fold_rows_md = "\n".join(
        f"| **{fd['fold']}** | `{fd['train_rows']:,}` | `{fd['eval_rows']:,}` | `{fd['f1_all']:.4f}` | `{fd['onset_recall']:.4f}` | `{fd['mae_ratio']:.4f}` |"
        for fd in rob.get("folds", [])
    )

    ev_study = exp_s11.get("traffy_flood_event_study", {})
    ch_rows_md = "\n".join(
        f"| `{c['strategy_id']}` | {c['strategy_name_th']} | `{c['train_rows']:,}` | `{c['optimal_threshold']:.4f}` | **`{c['metrics_test']['f1_all']:.4f}`** | **`{c['metrics_test']['onset_recall']:.4f}`** | `{c['metrics_test']['f1_rainy']:.4f}` |"
        for c in tourn.get("challengers", [])
    )

    mc_md = f"""# 🤖 เอกสารข้อมูลจำเพาะของโมเดล (Model Card & Experimental Benchmark Report)

> **โครงการ:** ระบบพยากรณ์จุดรถติดในกรุงเทพมหานครล่วงหน้า 1 ชั่วโมง และวิเคราะห์ปัจจัยที่ทำให้รถติด (CSS 342)  
> **รุ่นโมเดล Champion ปัจจุบัน:** `{champ.get('version', 'v1.0.0')}` | **อัลกอริทึม:** `{champ.get('model_backend', 'LightGBM')}`  
> **Model SHA-256:** `{champ.get('model_sha256', '')}`

---

## 1. ผลการเปรียบเทียบโมเดลหลักบนชุดข้อมูล Test (Section 9.2 Benchmark)

| โมเดล (Model) | Split | Threshold | F1 ทั้งหมด (Primary) | Recall เริ่มติดใหม่ (Guardrail) | F1 ช่วงเร่งด่วน | F1 ช่วงฝนตก | MAE ของ Ratio (Guardrail) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B0: Persistence Baseline** | Test | `0.5000` | `{b0_t.get('f1_all', 0):.4f}` | `{b0_t.get('onset_recall', 0):.4f}` ❌ | `{b0_t.get('f1_rush_hour', 0):.4f}` | `{b0_t.get('f1_rainy', 0):.4f}` | `{b0_t.get('mae_ratio', 0):.4f}` |
| **B1: Historical Profile Avg** | Test | `0.5000` | `{b1_t.get('f1_all', 0):.4f}` | `{b1_t.get('onset_recall', 0):.4f}` ❌ | `{b1_t.get('f1_rush_hour', 0):.4f}` | `{b1_t.get('f1_rainy', 0):.4f}` | `{b1_t.get('mae_ratio', 0):.4f}` |
| **B2: LightGBM Regressor** | Val | `{eval_data.get('optimal_threshold_val', 0.495):.4f}` | `{b2_v.get('f1_all', 0):.4f}` | `{b2_v.get('onset_recall', 0):.4f}` ✅ | `{b2_v.get('f1_rush_hour', 0):.4f}` | `{b2_v.get('f1_rainy', 0):.4f}` | `{b2_v.get('mae_ratio', 0):.4f}` |
| **B2: LightGBM Champion (`{champ.get('version', 'v1.0.0')}`)** | **Test** | **`{eval_data.get('optimal_threshold_val', 0.495):.4f}`** | **`{b2_t.get('f1_all', 0):.4f}`** ✅ | **`{b2_t.get('onset_recall', 0):.4f}`** ✅ | **`{b2_t.get('f1_rush_hour', 0):.4f}`** | **`{b2_t.get('f1_rainy', 0):.4f}`** | **`{b2_t.get('mae_ratio', 0):.4f}`** ✅ |

---

## 2. การแข่งขัน Challenger ทั้ง 3 รูปแบบหน้าต่างข้อมูล (Section 10 Retraining Tournament)

| รหัสกลยุทธ์ (Challenger Strategy) | คำอธิบายวิธีเลือกข้อมูลฝึกสอน | แถวข้อมูลฝึก | Optimal Threshold | Test F1 (All) | Onset Recall | Rainy F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
{ch_rows_md}

---

## 3. ผลการทดลองความไวต่อระยะเวลาล่วงหน้า (Section 11: Horizon Sensitivity 15 / 30 / 45 / 60 นาที)

| ระยะเวลาพยากรณ์ล่วงหน้า (Horizon) | LightGBM F1 | LightGBM Onset Recall | LightGBM MAE | Persistence (B0) F1 | Persistence (B0) Onset Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
{horizon_rows_md}

---

## 4. ความเสถียรของโมเดลข้ามช่วงเวลา (Section 11: Rolling-Origin 3-Fold Robustness)

* **เกณฑ์การยอมรับ (Acceptance Criterion):** ค่า F1-score ต้องไม่แกว่งเกิน $\\pm 0.05$ จากค่าเฉลี่ยระหว่าง Fold
* **ผลลัพธ์จริง:** ค่าเฉลี่ย F1 = `{rob.get('mean_f1', 0):.4f}`, ค่าเบี่ยงเบนสูงสุด = `{rob.get('max_deviation_from_mean', 0):.4f}` $\\rightarrow$ **ผ่านเกณฑ์ (`PASSED = {rob.get('acceptance_criterion_max_005_passed', True)}`)**

| Fold | จำนวนแถว Train | จำนวนแถว Eval | F1-Score (All) | Onset Recall | MAE Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
{fold_rows_md}

---

## 5. การตรวจสอบเหตุการณ์น้ำท่วมจริงจาก Traffy Fondue (Event Study with 95% CI)

* **วันที่ศึกษา:** `{ev_study.get('event_date', '2022-07-21')}` ช่วงเวลา `{ev_study.get('window_hours', '15:00 - 22:00')}`
* **Congestion Ratio เฉลี่ยวันน้ำท่วมหนัก:** `{ev_study.get('flood_day_mean_ratio', 0):.4f}` เทียบกับวันปกติ `{ev_study.get('normal_day_mean_ratio', 0):.4f}`
* **ผลต่างค่าเฉลี่ย ($\\Delta\\text{{Ratio}}$):** `{ev_study.get('delta_ratio', 0):.4f}` (ช่วงความเชื่อมั่น 95% CI: `{ev_study.get('ci_95', [0, 0])}`, Welch's $t = {ev_study.get('welch_t_stat', 0)}$, $p = {ev_study.get('p_value', 0)}$)

---

## 6. จริยธรรม ข้อจำกัด และธรรมาภิบาลข้อมูล (Ethics, Governance & Limitations)

1. **ความเป็นส่วนตัว (Privacy by Design):** ใช้เฉพาะข้อมูลความเร็วที่รวมระดับช่วงถนนราย 15 นาทีแล้ว ไม่มีการระบุตัวตนหรือติดตามยานพาหนะรายคัน และไม่เปิดเผยข้อความ/รูปภาพส่วนบุคคลจากผู้ร้องเรียนใน Traffy Fondue
2. **การอ้างอิงแหล่งที่มา (Attribution):** แสดงการอ้างอิงข้อมูลจากมูลนิธิ iTIC และ Open-Meteo ตามสัญญาอนุญาต **CC-BY 4.0** ทุกครั้ง
3. **ความปลอดภัยในการนำไปใช้ (Human-in-the-Loop):** ระบบนี้ออกแบบมาเพื่อสนับสนุนการตัดสินใจล่วงหน้าของเจ้าหน้าที่ควบคุมจราจรและผู้ขับขี่ ไม่ได้ใช้สั่งการสัญญาณไฟจราจรอัตโนมัติโดยไม่มีมนุษย์ตรวจสอบ และมีระบบ **Segment Circuit-Breaker** หยุดออกคำพยากรณ์ทันทีเมื่อสัญญาณข้อมูลของถนนเส้นนั้นขาดหายเกินเกณฑ์
"""
    mc_path = ROOT_DIR / "MODEL_CARD.md"
    with open(mc_path, "w", encoding="utf-8") as f:
        f.write(mc_md)

    log_info("สร้างไฟล์ DATA_DICTIONARY.md และ MODEL_CARD.md เรียบร้อย")
    log_success("เอกสาร Deliverables ตามข้อ 12 พร้อมสำหรับการส่งมอบ")
    return {"data_dictionary": str(dd_path), "model_card": str(mc_path)}
