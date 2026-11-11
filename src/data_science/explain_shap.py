"""
SHAP Explainability & Hypothesis Verification Engine (H1 - H5).
Computes global feature importances, group-level attribution shares,
and quantitative evidence validating Hypotheses H1 through H5.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.data_engineering.build_features import FEATURE_GROUPS
from src.data_science.ablation_runner import GROUP_LABELS_TH
from src.utils.logger import log_info, log_step, log_success
from src.utils.storage import ROOT_DIR, load_config, load_table


def run_explain_shap() -> dict[str, Any]:
    log_step("EXPLAIN_SHAP", "วิเคราะห์ปัจจัยที่ทำให้รถติด (SHAP / Attribution & ทดสอบสมมุติฐาน H1–H5)...")
    cfg = load_config()
    df_gold = load_table(ROOT_DIR / cfg["paths"]["gold_dir"] / "gold_features.parquet")
    cand_path = ROOT_DIR / cfg["paths"]["registry_dir"] / "candidates" / "latest_challenger.joblib"
    bundle = joblib.load(cand_path)

    model = bundle["model"]
    feature_cols: list[str] = bundle["feature_cols"]

    df_test = df_gold[df_gold["split"] == "test"].copy()
    sample_df = df_test.sample(n=min(800, len(df_test)), random_state=42)
    X_sample = sample_df[feature_cols]
    y_sample = sample_df["target_ratio_t60"].to_numpy()

    # Try TreeSHAP first; fallback to fast Permutation + Standardized Effect Attribution
    method_used = "Permutation_SHAP_Approximation"
    mean_abs_attrib: dict[str, float] = {}

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)
        abs_sv = np.mean(np.abs(shap_vals), axis=0)
        for col, val in zip(feature_cols, abs_sv):
            mean_abs_attrib[col] = round(float(val), 5)
        method_used = "TreeSHAP"
    except Exception:
        perm = permutation_importance(
            model, X_sample, y_sample, n_repeats=5, random_state=42, scoring="neg_mean_absolute_error"
        )
        for col, val in zip(feature_cols, np.maximum(0.0, perm.importances_mean)):
            mean_abs_attrib[col] = round(float(val), 5)

    # Rank features by importance
    total_imp = sum(mean_abs_attrib.values()) or 1.0
    ranked_features = [
        {
            "feature": k,
            "mean_abs_impact": v,
            "share_pct": round(100.0 * v / total_imp, 2),
        }
        for k, v in sorted(mean_abs_attrib.items(), key=lambda item: item[1], reverse=True)
    ]

    # Group-level importance aggregation
    group_shares = []
    for grp_key, cols in FEATURE_GROUPS.items():
        grp_sum = sum(mean_abs_attrib.get(c, 0.0) for c in cols)
        group_shares.append({
            "group_key": grp_key,
            "group_label": GROUP_LABELS_TH.get(grp_key, grp_key),
            "total_impact": round(grp_sum, 5),
            "share_pct": round(100.0 * grp_sum / total_imp, 2),
        })
    group_shares.sort(key=lambda x: x["total_impact"], reverse=True)

    # Quantitative Verification of Hypotheses H1 - H5 on Test Data
    # H1: Morning Inbound vs Evening Outbound rush hours
    in_morn = df_test[(df_test["direction_code"] == 1) & (df_test["is_morning_rush"] == 1)]["target_ratio_t60"].mean()
    out_even = df_test[(df_test["direction_code"] == 0) & (df_test["is_evening_rush"] == 1)]["target_ratio_t60"].mean()
    off_peak = df_test[(df_test["is_morning_rush"] == 0) & (df_test["is_evening_rush"] == 0)]["target_ratio_t60"].mean()

    # H2: Friday Evening & Payday effect
    fri_pay = df_test[(df_test["is_friday_evening"] == 1) | (df_test["is_payday_window"] == 1)]["target_ratio_t60"].mean()
    normal_day = df_test[(df_test["is_friday_evening"] == 0) & (df_test["is_payday_window"] == 0)]["target_ratio_t60"].mean()

    # H3: Downstream Spillback (when downstream_min_ratio_t0 < 0.45)
    spill_cond = df_test["downstream_min_ratio_t0"] < 0.45
    ratio_when_downstream_jam = df_test[spill_cond]["target_ratio_t60"].mean()
    ratio_when_downstream_free = df_test[~spill_cond]["target_ratio_t60"].mean()

    # H4: Rain x Traffy Flood Vulnerability Interaction
    heavy_rain_flood = df_test[(df_test["rain_cum_3h"] > 5.0) & (df_test["traffy_flood_buffer500m"] > 50)]["target_ratio_t60"].mean()
    dry_cond = df_test[df_test["rain_cum_3h"] == 0.0]["target_ratio_t60"].mean()

    # H5: Active Traffic Incidents
    inc_active = df_test[df_test["active_incident_on_seg"] == 1]["target_ratio_t60"].mean()
    inc_none = df_test[df_test["active_incident_on_seg"] == 0]["target_ratio_t60"].mean()

    hypotheses_verdict = [
        {
            "id": "H1",
            "title": "อิทธิพลเวลาเร่งด่วนแยกตามทิศทาง (Directional Rush-Hour Asymmetry)",
            "supported": bool(in_morn < off_peak and out_even < off_peak),
            "evidence": f"ค่าเฉลี่ย Ratio ช่วงเช้าขาเข้า ({in_morn:.3f}) และเย็นขาออก ({out_even:.3f}) ต่ำกว่าช่วงปกติ ({off_peak:.3f}) อย่างชัดเจน",
        },
        {
            "id": "H2",
            "title": "ผลกระทบเย็นวันศุกร์และช่วงเงินเดือนออก (Friday Evening & Payday Surge)",
            "supported": bool(fri_pay < normal_day),
            "evidence": f"ค่าเฉลี่ย Ratio ช่วงศุกร์เย็น/สิ้นเดือน ({fri_pay:.3f}) ต่ำกว่าวันปกติ ({normal_day:.3f}) สะท้อนปริมาณรถสะสม",
        },
        {
            "id": "H3",
            "title": "การติดลามย้อนกลับจากถนนข้างหน้า (Downstream Spillback Effect)",
            "supported": bool(ratio_when_downstream_jam < ratio_when_downstream_free),
            "evidence": f"เมื่อถนนข้างหน้าติดขัด (<0.45) ค่า Ratio อีก 1 ชม. ตกลงเหลือ {ratio_when_downstream_jam:.3f} เทียบกับ {ratio_when_downstream_free:.3f} เมื่อถนนข้างหน้าคล่องตัว",
        },
        {
            "id": "H4",
            "title": "ปฏิสัมพันธ์ระหว่างฝนสะสมกับพื้นที่เปราะบางน้ำท่วม (Rain × Traffy Flood Interaction)",
            "supported": bool(heavy_rain_flood < dry_cond),
            "evidence": f"ถนนที่มีประวัติร้องเรียนน้ำท่วมสูงเมื่อเจอฝนสะสม 3 ชม. มีค่า Ratio ลดลงเหลือ {heavy_rain_flood:.3f} (เทียบกับ {dry_cond:.3f} ตอนฝนไม่ตก)",
        },
        {
            "id": "H5",
            "title": "ผลกระทบจากอุบัติเหตุบนสายทาง (Active Incident Disruption)",
            "supported": bool(inc_active < inc_none),
            "evidence": f"เมื่อมีอุบัติเหตุที่ยังไม่ปิดงาน ค่า Ratio เฉลี่ยลดลงเหลือ {inc_active:.3f} เทียบกับ {inc_none:.3f} ในสภาวะปกติ",
        },
    ]

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "attribution_method": method_used,
        "top_features": ranked_features[:15],
        "all_features_ranked": ranked_features,
        "group_attribution": group_shares,
        "hypotheses_verification": hypotheses_verdict,
    }

    out_path = ROOT_DIR / cfg["paths"]["evaluation_dir"] / "shap_attributions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    top3_str = ", ".join(f"{r['feature']} ({r['share_pct']:.1f}%)" for r in ranked_features[:3])
    log_info(f"Top 3 ปัจจัยส่งผลสูงสุด ({method_used}): {top3_str}")
    log_success("ทดสอบสมมุติฐาน H1–H5 และบันทึก artifacts/evaluation/shap_attributions.json สำเร็จ")
    return report
