"""
Clean Core Feature Group Definitions for the Real Bangkok iTIC 1-Hour-Ahead Dataset.
Pruned from 44 raw candidate columns down to 21 non-redundant Core Features (20 numeric + 1 categorical `link_cat`)
spanning all 7 Proposal Feature Groups (Section 8 Page 9 & Section 9.3 Feature Gate).
"""
from __future__ import annotations

FEATURE_GROUPS: dict[str, list[str]] = {
    "group_1_current_lags": [
        "red_rate_t0",
        "yellow_rate_t0",
        "delta_red_1h",
        "red_rate_lag_24h",
    ],
    "group_2_historical_profile": [
        "hist_avg_red_target_slot",
        "hist_p75_red_target_slot",
        "current_diff_from_hist",
    ],
    "group_3_temporal_cyclical": [
        "target_hour",
        "target_hour_sin",
        "target_hour_cos",
    ],
    "group_4_calendar": [
        "dow",
        "is_holiday",
        "is_friday_evening",
        "is_payday",
    ],
    "group_5_weather": [
        "precip_mm_t0",
        "precip_3h_mm_t0",
    ],
    "group_6_spatial_graph": [
        "downstream_red_rate_t0",
        "district_red_rate_t0",
    ],
    "group_7_road_traffy_static": [
        "train_flood_per_sq_km",
        "rain_x_flood_interaction",
    ],
}

GROUP_LABELS_TH: dict[str, str] = {
    "group_1_current_lags": "G1: สภาพปัจจุบันและแนวโน้ม (Current & Lags)",
    "group_2_historical_profile": "G2: รูปแบบในอดีต (Historical Profile)",
    "group_3_temporal_cyclical": "G3: ช่วงเวลาเป้าหมาย (Temporal Cyclical)",
    "group_4_calendar": "G4: ปฏิทินและวันเงินเดือนออก (Calendar)",
    "group_5_weather": "G5: ฝนสะสม Open-Meteo (Weather)",
    "group_6_spatial_graph": "G6: ถนนข้างหน้าและภาพรวมเขต (Spatial Graph)",
    "group_7_road_traffy_static": "G7: กายภาพถนนและจุดเสี่ยงน้ำท่วม (Road & Traffy)",
}

FEATURE_DESCRIPTIONS_TH: dict[str, str] = {
    "red_rate_t0": "สัดส่วนรถติดสีแดง ณ ปัจจุบัน (t0)",
    "yellow_rate_t0": "สัดส่วนรถชะลอตัวสีเหลือง ณ ปัจจุบัน (สัญญาณก่อนติด)",
    "delta_red_1h": "อัตราการเพิ่มขึ้นของรถติดเทียบ 1 ชม. ก่อน (t0 - t-1h)",
    "red_rate_lag_24h": "สัดส่วนรถติด ณ เวลาเดียวกันของเมื่อวาน (t-24h)",
    "hist_avg_red_target_slot": "ค่าเฉลี่ยรถติดในอดีตของถนนเส้นนั้น ณ ชั่วโมงเป้าหมาย (คิดเฉพาะ Train)",
    "hist_p75_red_target_slot": "ค่า Percentile 75 ของรถติดในอดีตของถนนเส้นนั้น ณ ชั่วโมงเป้าหมาย",
    "current_diff_from_hist": "ส่วนต่างรถติดปัจจุบันเทียบกับค่าปกติในอดีต (วัดความผิดปกติ)",
    "target_hour": "ชั่วโมงเป้าหมายที่จะพยากรณ์ (0–23 น.)",
    "target_hour_sin": "ค่า sin ของชั่วโมงเป้าหมาย (วงจรเวลา 24 ชม.)",
    "target_hour_cos": "ค่า cos ของชั่วโมงเป้าหมาย (วงจรเวลา 24 ชม.)",
    "dow": "วันในสัปดาห์ (0=จันทร์ .. 6=อาทิตย์)",
    "is_holiday": "วันหยุดนักขัตฤกษ์ตามประกาศ ธปท. (0/1)",
    "is_friday_evening": "ช่วงเย็นวันศุกร์ 15:00–21:00 น. (0/1 · ทดสอบ H5)",
    "is_payday": "ช่วงวันเงินเดือนออก วันที่ 25–31 และ 1–2 (0/1 · ทดสอบ H5)",
    "precip_mm_t0": "ปริมาณฝนรายชั่วโมง ณ เวลาพยากรณ์ (mm · ทดสอบ H2)",
    "precip_3h_mm_t0": "ปริมาณฝนสะสม 3 ชั่วโมงล่าสุด (mm)",
    "downstream_red_rate_t0": "สัดส่วนรถติดของถนนข้างหน้าที่ต่อจากปลายทาง (Spillback · ทดสอบ H3)",
    "district_red_rate_t0": "ค่าเฉลี่ยรถติดของถนนทุกเส้นในเขตเดียวกัน ณ เวลา t0",
    "train_flood_per_sq_km": "ความหนาแน่นจุดร้องเรียนน้ำท่วมต่อ ตร.กม. จาก Traffy Fondue (คิดเฉพาะ Train)",
    "rain_x_flood_interaction": "ปฏิสัมพันธ์ ฝนสะสม 3 ชม. × ความเสี่ยงน้ำท่วมของเขต (ทดสอบ H4)",
    "link_cat": "รหัสประจำช่วงถนนที่มีทิศทาง (Categorical ID ทั้ง 448 ช่วงถนน)",
}

ALL_FEATURE_COLS: list[str] = [col for cols in FEATURE_GROUPS.values() for col in cols]
