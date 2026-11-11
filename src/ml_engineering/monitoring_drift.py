"""
Data Drift (PSI) & Concept Drift (Rolling 7-Day Performance) Monitoring Engine.
Responsibility: Sarit (67090500441) - ML Engineer

Features:
1. Population Stability Index (PSI) across key traffic & weather features:
   - PSI < 0.10: STABLE
   - 0.10 <= PSI < 0.25: WARNING_DRIFT
   - PSI >= 0.25: CRITICAL_DRIFT (triggers retraining alert)
2. Daily & Rolling Performance Tracking (F1, Onset Recall, MAE)
3. Stress-Test Simulation ("Monsoon Extreme Rain Shift") to verify drift alerting
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.data_science.evaluate_metrics import evaluate_predictions
from src.utils.logger import log_info, log_step, log_success, log_warning
from src.utils.storage import ROOT_DIR, load_config, load_table


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """
    Computes Population Stability Index (PSI) between reference (Train) and current distributions.
    """
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]
    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.unique(np.percentile(ref_clean, quantiles))
    if len(bin_edges) < 2:
        # Fallback for zero-inflated features like rainfall
        max_val = max(float(np.max(ref_clean)), float(np.max(cur_clean)), 1.0)
        bin_edges = np.linspace(0.0, max_val, bins + 1)
    else:
        bin_edges[0] = min(bin_edges[0], float(np.min(cur_clean))) - 1e-6
        bin_edges[-1] = max(bin_edges[-1], float(np.max(cur_clean))) + 1e-6

    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    cur_counts, _ = np.histogram(cur_clean, bins=bin_edges)

    ref_pct = np.maximum(ref_counts / max(1, len(ref_clean)), eps)
    cur_pct = np.maximum(cur_counts / max(1, len(cur_clean)), eps)

    psi_value = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(max(0.0, psi_value), 4)


def classify_psi(psi_val: float, warn_th: float = 0.10, crit_th: float = 0.25) -> str:
    if psi_val >= crit_th:
        return "CRITICAL_DRIFT"
    if psi_val >= warn_th:
        return "WARNING_DRIFT"
    return "STABLE"


def run_monitoring_drift() -> dict[str, Any]:
    log_step("DRIFT_MONITOR", "คำนวณ Data Drift (PSI) และติดตาม Concept Drift รายวัน (Rolling Performance)...")
    cfg = load_config()
    mon_cfg = cfg["monitoring"]
    df_gold = load_table(ROOT_DIR / cfg["paths"]["gold_dir"] / "gold_features.parquet")

    eval_path = ROOT_DIR / cfg["paths"]["evaluation_dir"] / "evaluation_summary.json"
    optimal_th = 0.50
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            optimal_th = float(json.load(f).get("optimal_threshold_val", 0.50))

    df_train = df_gold[df_gold["split"] == "train"].copy()
    df_test = df_gold[df_gold["split"] == "test"].copy()

    monitored_features = mon_cfg["monitored_features"]
    warn_th = float(mon_cfg["psi_warning_threshold"])
    crit_th = float(mon_cfg["psi_critical_threshold"])
    n_bins = int(mon_cfg["psi_bins"])

    # 1. Normal Test Window PSI vs Train Reference
    feature_drift_normal = []
    for col in monitored_features:
        if col not in df_train.columns:
            continue
        psi_val = compute_psi(
            df_train[col].to_numpy(dtype=np.float64),
            df_test[col].to_numpy(dtype=np.float64),
            bins=n_bins,
        )
        status = classify_psi(psi_val, warn_th, crit_th)
        feature_drift_normal.append({
            "feature": col,
            "psi": psi_val,
            "status": status,
            "ref_mean": round(float(df_train[col].mean()), 4),
            "cur_mean": round(float(df_test[col].mean()), 4),
        })

    # 2. Simulated Extreme Monsoon Regime Shift (Stress-Test Scenario for MLOps Alerting)
    df_stress = df_test.copy()
    df_stress["rain_mm_t0"] = df_stress["rain_mm_t0"] * 3.5 + 4.0
    df_stress["rain_cum_3h"] = df_stress["rain_cum_3h"] * 3.8 + 18.0
    df_stress["ratio_t0"] = np.clip(df_stress["ratio_t0"] - 0.16, 0.10, 1.20)
    df_stress["downstream_min_ratio_t0"] = np.clip(df_stress["downstream_min_ratio_t0"] - 0.18, 0.10, 1.20)

    feature_drift_stress = []
    for col in monitored_features:
        if col not in df_train.columns:
            continue
        psi_val = compute_psi(
            df_train[col].to_numpy(dtype=np.float64),
            df_stress[col].to_numpy(dtype=np.float64),
            bins=n_bins,
        )
        status = classify_psi(psi_val, warn_th, crit_th)
        feature_drift_stress.append({
            "feature": col,
            "psi": psi_val,
            "status": status,
            "ref_mean": round(float(df_train[col].mean()), 4),
            "stress_mean": round(float(df_stress[col].mean()), 4),
        })

    # 3. Daily & Rolling Performance Tracking across Val + Test timeline
    df_eval_timeline = df_gold[df_gold["split"].isin(["val", "test"])].copy()
    df_eval_timeline["date_str"] = pd.to_datetime(df_eval_timeline["ts_t"]).dt.strftime("%Y-%m-%d")

    daily_metrics = []
    for date_str, grp in df_eval_timeline.groupby("date_str"):
        m_b2 = evaluate_predictions(grp, "pred_b2_lgbm", threshold=optimal_th)
        m_b0 = evaluate_predictions(grp, "pred_b0_persistence", threshold=0.50)
        daily_metrics.append({
            "date": date_str,
            "split": str(grp["split"].iloc[0]),
            "num_observations": int(len(grp)),
            "mean_rain_mm": round(float(grp["rain_mm_t0"].mean()), 3),
            "f1_b2": m_b2["f1_all"],
            "f1_b0": m_b0["f1_all"],
            "onset_recall_b2": m_b2["onset_recall"],
            "mae_b2": m_b2["mae_ratio"],
        })

    df_daily = pd.DataFrame(daily_metrics)
    if len(df_daily) > 0:
        df_daily["rolling_3d_f1"] = df_daily["f1_b2"].rolling(3, min_periods=1).mean().round(4)
        df_daily["rolling_3d_onset_recall"] = df_daily["onset_recall_b2"].rolling(3, min_periods=1).mean().round(4)
        daily_metrics = df_daily.to_dict(orient="records")

    any_crit_normal = any(r["status"] == "CRITICAL_DRIFT" for r in feature_drift_normal)
    any_warn_normal = any(r["status"] == "WARNING_DRIFT" for r in feature_drift_normal)
    overall_health = (
        "CRITICAL_RETRAIN_NEEDED"
        if any_crit_normal
        else ("WARNING_MONITOR_CLOSELY" if any_warn_normal else "HEALTHY_STABLE")
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_system_health": overall_health,
        "psi_thresholds": {
            "warning": warn_th,
            "critical": crit_th,
        },
        "current_window_psi": feature_drift_normal,
        "monsoon_stress_test_psi": feature_drift_stress,
        "daily_performance_timeline": daily_metrics,
        "recommended_action": (
            "Normal operation: Model & feature distributions are within acceptable bounds."
            if overall_health == "HEALTHY_STABLE"
            else "Trigger automated retraining pipeline with recent window data."
        ),
    }

    out_path = ROOT_DIR / cfg["paths"]["monitoring_dir"] / "drift_monitoring_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    max_psi_item = max(feature_drift_normal, key=lambda x: x["psi"])
    log_info(
        f"System Drift Status: [{overall_health}] | Max PSI = {max_psi_item['feature']} ({max_psi_item['psi']:.4f}) | "
        f"Tracked {len(daily_metrics)} evaluation days"
    )
    if overall_health != "HEALTHY_STABLE":
        log_warning(f"ตรวจพบการเปลี่ยนแปลงของการกระจายข้อมูล: {overall_health}")
    log_success("บันทึกรายงาน Monitoring ลง artifacts/monitoring/drift_monitoring_report.json สำเร็จ")
    return report
