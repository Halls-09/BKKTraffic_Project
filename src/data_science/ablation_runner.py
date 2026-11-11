"""
B3 Ablation Study Runner.
Systematically removes one feature group at a time (Leave-One-Group-Out)
across all 8 Feature Groups to quantify each group's marginal contribution to:
- Overall F1
- Onset Recall
- Rainy F1
- Rush Hour F1
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.data_engineering.build_features import ALL_FEATURE_COLS, FEATURE_GROUPS
from src.data_science.evaluate_metrics import evaluate_predictions
from src.data_science.train_lgbm import _build_regressor
from src.utils.logger import log_info, log_step, log_success
from src.utils.storage import ROOT_DIR, load_config, load_table


GROUP_LABELS_TH = {
    "group_1_current_lags": "G1: สภาพปัจจุบันและ Lag (Current & Lags)",
    "group_2_historical_profile": "G2: รูปแบบในอดีต (Historical Profile)",
    "group_3_temporal_cyclical": "G3: ช่วงเวลา (Temporal Cyclical)",
    "group_4_calendar": "G4: ปฏิทิน (Calendar)",
    "group_5_weather": "G5: ฝนและสภาพอากาศ (Weather)",
    "group_6_spatial_graph": "G6: ถนนข้างเคียง (Spatial Graph Neighbors)",
    "group_7_incidents": "G7: อุบัติเหตุ (Traffic Incidents)",
    "group_8_road_traffy_static": "G8: ลักษณะถนนและพื้นที่เสี่ยงน้ำท่วม (Road & Traffy)",
}


def run_ablation_study() -> dict[str, Any]:
    log_step("ABLATION_B3", "รันการทดลอง B3 (Ablation Study ตัดทีละกลุ่มฟีเจอร์ทั้ง 8 กลุ่ม)...")
    cfg = load_config()
    df_gold = load_table(ROOT_DIR / cfg["paths"]["gold_dir"] / "gold_features.parquet")
    eval_path = ROOT_DIR / cfg["paths"]["evaluation_dir"] / "evaluation_summary.json"

    optimal_th = 0.50
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            optimal_th = float(json.load(f).get("optimal_threshold_val", 0.50))

    df_train = df_gold[df_gold["split"] == "train"].copy()
    df_test = df_gold[df_gold["split"] == "test"].copy()
    y_train = df_train["target_ratio_t60"].to_numpy(dtype=np.float64)

    sample_weights = (
        np.ones(len(df_train), dtype=np.float64)
        + 1.5 * df_train["is_onset_event"].to_numpy()
        + 0.5 * (df_train["rain_mm_t0"].to_numpy() > 1.0)
    )

    hp = dict(cfg["model"]["hyperparameters"])
    hp["n_estimators"] = 180  # Fast ablation trees

    # Full model reference on Test
    full_model, _ = _build_regressor(hp)
    full_model.fit(df_train[ALL_FEATURE_COLS], y_train, sample_weight=sample_weights)
    df_test["pred_full"] = np.clip(full_model.predict(df_test[ALL_FEATURE_COLS]), 0.0, 1.2)
    full_metrics = evaluate_predictions(df_test, "pred_full", threshold=optimal_th)

    ablation_rows = []
    for grp_key, grp_cols in FEATURE_GROUPS.items():
        remaining_cols = [c for c in ALL_FEATURE_COLS if c not in grp_cols]
        ab_model, _ = _build_regressor(hp)
        ab_model.fit(df_train[remaining_cols], y_train, sample_weight=sample_weights)
        pred_col = f"pred_without_{grp_key}"
        df_test[pred_col] = np.clip(ab_model.predict(df_test[remaining_cols]), 0.0, 1.2)
        m = evaluate_predictions(df_test, pred_col, threshold=optimal_th)

        row = {
            "group_key": grp_key,
            "group_label": GROUP_LABELS_TH.get(grp_key, grp_key),
            "removed_features": grp_cols,
            "f1_all": m["f1_all"],
            "delta_f1_all": round(m["f1_all"] - full_metrics["f1_all"], 4),
            "onset_recall": m["onset_recall"],
            "delta_onset_recall": round(m["onset_recall"] - full_metrics["onset_recall"], 4),
            "f1_rainy": m["f1_rainy"],
            "delta_f1_rainy": round(m["f1_rainy"] - full_metrics["f1_rainy"], 4),
            "f1_rush_hour": m["f1_rush_hour"],
            "delta_f1_rush_hour": round(m["f1_rush_hour"] - full_metrics["f1_rush_hour"], 4),
            "mae_ratio": m["mae_ratio"],
        }
        ablation_rows.append(row)

    ablation_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "full_model_test_metrics": full_metrics,
        "ablation_experiments": ablation_rows,
    }

    out_path = ROOT_DIR / cfg["paths"]["evaluation_dir"] / "ablation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ablation_report, f, indent=2, ensure_ascii=False)

    log_info(f"ทดสอบ Ablation ทั้ง {len(ablation_rows)} กลุ่มฟีเจอร์สำเร็จ (Full F1={full_metrics['f1_all']:.4f})")
    log_success("บันทึกผลลัพธ์ลง artifacts/evaluation/ablation_results.json เรียบร้อย")
    return ablation_report
