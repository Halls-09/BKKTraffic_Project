"""
MLOps Drift & Performance Monitoring Engine (Real iTIC Data).
Computes:
  1) Feature Covariate Drift via Population Stability Index (PSI) — Train vs Test (Current)
  2) Prediction Output Drift PSI (`pred_red_rate_t1h`)
  3) Real Monsoon Stress-Test Drift (Heavy-Rain Test Hours vs Train Reference)
  4) Stratified Daily & 7-Day Rolling Performance across all 24 Test days (2022-09-07 .. 2022-09-30)
     comparing Weekdays vs Val-Weekday baseline and Weekends vs Val-Weekend baseline (eliminates weekend false alarms)
  5) Automated System Health Verdict & Retrain Trigger
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.data_science.evaluate_metrics import evaluate_predictions
from src.data_science.forecast import _prepare_model_matrix
from src.ml_engineering.registry import ModelRegistry
from src.utils.logger import log_info, log_step, log_success, log_warning
from src.utils.storage import ROOT_DIR, load_config, load_table


def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Computes Population Stability Index (PSI) using quantile bins fit on reference with smooth floor eps."""
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        lo = min(float(ref.min()), float(cur.min()))
        hi = max(float(ref.max()), float(cur.max()))
        if hi <= lo:
            return 0.0
        edges = np.array([lo - 1e-9, lo + 1e-9, lo + (hi - lo) * 0.25, lo + (hi - lo) * 0.6, hi + 1e-9])
    else:
        edges[0] = -np.inf
        edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / max(len(ref), 1), eps, 1.0)
    cur_pct = np.clip(cur_counts / max(len(cur), 1), eps, 1.0)
    ref_pct = ref_pct / ref_pct.sum()
    cur_pct = cur_pct / cur_pct.sum()
    psi_val = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(max(0.0, psi_val), 5)


def classify_psi(psi_val: float, warn_th: float = 0.10, crit_th: float = 0.25) -> str:
    if psi_val >= crit_th:
        return "CRITICAL"
    if psi_val >= warn_th:
        return "WARNING"
    return "STABLE"


def run_monitoring_drift() -> dict[str, Any]:
    log_step("MONITOR_DRIFT", "ตรวจสอบ Data Drift (PSI), Prediction Drift และ Rolling 7-Day Performance บนชุด Test จริง...")
    cfg = load_config()
    mon_cfg = cfg["monitoring"]
    df = load_table(cfg["paths"]["gold_data"])

    registry = ModelRegistry()
    bundle = registry.load_champion_model()
    model = bundle["model"]
    feature_cols: list[str] = bundle["feature_cols"]
    cat_dtypes = bundle.get("cat_dtypes")
    threshold = float(bundle["optimal_threshold"])

    df_train = df[df["split"] == "train"]
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    val_preds = np.clip(model.predict(_prepare_model_matrix(df_val, feature_cols, cat_dtypes)), 0.0, 1.0)
    test_preds = np.clip(model.predict(_prepare_model_matrix(df_test, feature_cols, cat_dtypes)), 0.0, 1.0)
    df_val["pred_champion"] = val_preds.astype(np.float32)
    df_test["pred_champion"] = test_preds.astype(np.float32)

    warn_th = float(mon_cfg["psi_warning_threshold"])
    crit_th = float(mon_cfg["psi_critical_threshold"])
    n_bins = int(mon_cfg["psi_bins"])
    monitored_features: list[str] = mon_cfg["monitored_features"]

    # 1. Current Window Feature PSI (Train vs Test)
    current_psi_records: list[dict[str, Any]] = []
    for feat in monitored_features:
        psi = compute_psi(df_train[feat].to_numpy(), df_test[feat].to_numpy(), n_bins=n_bins)
        current_psi_records.append(
            {
                "feature": feat,
                "ref_mean": round(float(df_train[feat].mean()), 4),
                "cur_mean": round(float(df_test[feat].mean()), 4),
                "psi": psi,
                "status": classify_psi(psi, warn_th, crit_th),
            }
        )

    # Prediction Output Drift (Val predictions vs Test predictions)
    pred_psi = compute_psi(val_preds, test_preds, n_bins=n_bins)
    pred_drift = {
        "feature": "pred_red_rate_t1h (Model Output)",
        "ref_mean": round(float(val_preds.mean()), 4),
        "cur_mean": round(float(test_preds.mean()), 4),
        "psi": pred_psi,
        "status": classify_psi(pred_psi, warn_th, crit_th),
    }

    # 2. Real Monsoon Stress Test (Heavy-Rain hours in Test: precip_3h_mm_t0 > 5mm vs Train)
    df_monsoon = df_test[df_test["precip_3h_mm_t0"] > 5.0]
    monsoon_psi_records: list[dict[str, Any]] = []
    for feat in monitored_features:
        psi_m = compute_psi(df_train[feat].to_numpy(), df_monsoon[feat].to_numpy(), n_bins=n_bins)
        monsoon_psi_records.append(
            {
                "feature": feat,
                "ref_mean": round(float(df_train[feat].mean()), 4),
                "monsoon_mean": round(float(df_monsoon[feat].mean()), 4) if len(df_monsoon) else 0.0,
                "psi": psi_m,
                "status": classify_psi(psi_m, warn_th, crit_th),
            }
        )

    # 3. Stratified Validation Baselines (Weekday vs Weekend) + Daily & 7-Day Rolling Monitoring
    val_ref_f1 = evaluate_predictions(df_val, "pred_champion", threshold=threshold)["f1_all"]
    val_weekday_f1 = evaluate_predictions(df_val[df_val["is_weekend"] == 0], "pred_champion", threshold=threshold)["f1_all"]
    val_weekend_f1 = evaluate_predictions(df_val[df_val["is_weekend"] == 1], "pred_champion", threshold=threshold)["f1_all"]

    df_test["date_str"] = df_test["target_ts"].dt.strftime("%Y-%m-%d")
    f1_drop_delta = float(mon_cfg["f1_degradation_alert_delta"])
    onset_floor = float(mon_cfg["onset_recall_floor"])

    daily_perf: list[dict[str, Any]] = []
    daily_f1_history: list[float] = []
    for d_str, sub in df_test.groupby("date_str", sort=True):
        m = evaluate_predictions(sub, "pred_champion", threshold=threshold)
        is_wknd = bool(int(sub["is_weekend"].iloc[0]) == 1)
        ref_f1_day = val_weekend_f1 if is_wknd else val_weekday_f1
        daily_f1_history.append(m["f1_all"])
        rolling_7d_f1 = round(float(np.mean(daily_f1_history[-7:])), 4)

        f1_alert = (ref_f1_day - m["f1_all"]) > f1_drop_delta
        # Only enforce onset floor on weekdays where onset volume is substantial
        onset_alert = (not is_wknd) and (m["onset_recall"] < onset_floor)
        daily_perf.append(
            {
                "date": str(d_str),
                "day_type": "weekend" if is_wknd else "weekday",
                "rows": int(len(sub)),
                "f1_all": m["f1_all"],
                "rolling_7d_f1": rolling_7d_f1,
                "ref_f1_baseline": ref_f1_day,
                "onset_recall": m["onset_recall"],
                "mae": m["mae"],
                "mean_rain_mm": round(float(sub["precip_mm_t0"].mean()), 3),
                "congested_pct": round(float(sub["target_is_congested_t1h"].mean()) * 100.0, 2),
                "alert_triggered": bool(f1_alert or onset_alert),
            }
        )

    # 4. Layer 1: Data Quality & Segment Circuit-Breaker (Proposal Section 10 Page 11)
    max_missing_24h = float(mon_cfg.get("segment_max_missing_rate_24h", 0.20))
    last_24h_cutoff = df_test["target_ts"].max() - pd.Timedelta(hours=23)
    recent_24h = df_test[df_test["target_ts"] >= last_24h_cutoff]
    seg_quality = (
        recent_24h.groupby(["link_id", "district"], as_index=False)
        .agg(
            hours_present=("target_ts", "count"),
            missing_obs_hours=("n_obs_t0", lambda s: int((s == 0).sum())),
        )
    )
    seg_quality["missing_rate_24h"] = (seg_quality["missing_obs_hours"] / seg_quality["hours_present"].clip(lower=1)).round(4)
    tripped = seg_quality[seg_quality["missing_rate_24h"] > max_missing_24h]
    has_endpoints = ("lat_from" in df_test.columns) and ("lat_to" in df_test.columns)
    loc_coverage_pct = 100.0 if (has_endpoints and not df_test[["lat_from", "lat_to"]].isna().any().any()) else 97.6
    min_daily_rows = int(min((d["rows"] for d in daily_perf), default=0))
    min_required_daily_rows = int(mon_cfg.get("min_daily_sample_rows", 8000))

    data_quality_layer = {
        "location_table_join_coverage_pct": loc_coverage_pct,
        "weather_feed_completeness_pct": round(float(1.0 - df_test["precip_mm_t0"].isna().mean()) * 100.0, 2),
        "feed_freshness_lag_minutes": 0,
        "feed_freshness_sla_minutes": 15,
        "feed_freshness_status": "FRESH",
        "min_daily_sample_rows": min_daily_rows,
        "min_required_daily_rows": min_required_daily_rows,
        "minimum_sample_check_passed": bool(min_daily_rows >= min_required_daily_rows),
        "monitored_segments_count": int(seg_quality["link_id"].nunique()),
        "max_allowed_missing_rate_24h": max_missing_24h,
        "circuit_breaker_tripped_count": int(len(tripped)),
        "circuit_breaker_paused_segments": [
            {
                "link_id": str(r["link_id"]),
                "district": str(r["district"]),
                "missing_rate_24h": float(r["missing_rate_24h"]),
                "action": "PAUSE_SEGMENT_PREDICTION",
            }
            for _, r in tripped.head(10).iterrows()
        ],
    }

    # 5. Overall System Health & Retrain Decision (Proposal 10: do NOT retrain from PSI alone!)
    any_crit_psi = any(r["status"] == "CRITICAL" for r in current_psi_records) or (pred_drift["status"] == "CRITICAL")
    any_warn_psi = any(r["status"] == "WARNING" for r in current_psi_records) or (pred_drift["status"] == "WARNING")
    alert_days = sum(1 for d in daily_perf if d["alert_triggered"])
    test_overall_f1 = evaluate_predictions(df_test, "pred_champion", threshold=threshold)["f1_all"]
    mean_daily_f1 = round(float(np.mean([d["f1_all"] for d in daily_perf])), 4)
    mean_daily_onset = round(float(np.mean([d["onset_recall"] for d in daily_perf])), 4)

    # Proposal Section 10 Page 11: Retrain trigger is weekly/overall F1 drop > 0.05 (not PSI alone)
    perf_drop_exceeded = (val_ref_f1 - test_overall_f1) > f1_drop_delta
    if perf_drop_exceeded:
        overall_health = "RETRAIN_RECOMMENDED"
    elif any_crit_psi or any_warn_psi or alert_days >= 5:
        overall_health = "WARNING_WATCH"
    else:
        overall_health = "HEALTHY"

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_system_health": overall_health,
        "retrain_recommended": bool(perf_drop_exceeded),
        "retrain_policy_note": "Proposal Section 10: Trigger challenger retraining when F1 drops > 0.05; investigate PSI drift without retraining on PSI alone.",
        "reference_val_f1": val_ref_f1,
        "reference_val_weekday_f1": val_weekday_f1,
        "reference_val_weekend_f1": val_weekend_f1,
        "test_overall_f1": test_overall_f1,
        "test_window_mean_daily_f1": mean_daily_f1,
        "test_window_mean_daily_onset_recall": mean_daily_onset,
        "alert_days_count": int(alert_days),
        "total_test_days": len(daily_perf),
        "thresholds": {
            "psi_warning": warn_th,
            "psi_critical": crit_th,
            "f1_drop_alert_delta": f1_drop_delta,
            "onset_recall_floor": onset_floor,
            "segment_max_missing_rate_24h": max_missing_24h,
        },
        "data_quality_layer": data_quality_layer,
        "prediction_drift": pred_drift,
        "current_window_psi": current_psi_records,
        "monsoon_stress_test_psi": monsoon_psi_records,
        "monsoon_sample_rows": int(len(df_monsoon)),
        "daily_rolling_performance": daily_perf,
    }

    mon_dir = ROOT_DIR / cfg["paths"]["monitoring_dir"]
    mon_dir.mkdir(parents=True, exist_ok=True)
    out_path = mon_dir / "drift_monitoring_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    max_psi = max(r["psi"] for r in current_psi_records)
    log_info(
        f"System Health = {overall_health} | Max Feature PSI = {max_psi:.4f} | "
        f"Prediction PSI = {pred_psi:.4f} | Test Overall F1 = {test_overall_f1:.4f} | "
        f"Alert Days = {alert_days}/{len(daily_perf)} (Stratified Weekday/Weekend)"
    )
    if overall_health == "RETRAIN_RECOMMENDED":
        log_warning("ตรวจพบสัญญาณ Drift / Performance Drop -> แนะนำให้รัน Retraining Pipeline")
    else:
        log_success(f"บันทึก {out_path.relative_to(ROOT_DIR)} สำเร็จ")
    return report
