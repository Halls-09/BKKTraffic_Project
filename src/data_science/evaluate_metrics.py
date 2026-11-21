"""
Evaluation Metrics & Guardrail Threshold Optimization for Real Bangkok iTIC Data.
Target: `target_red_rate_t1h` in [0, 1] (proportion of red/congested snapshots in hour t+1).
Class 1 (Congested) = `red_rate >= threshold` (default 0.50 = congested >= 30 mins in the hour).
Onset Event = `red_rate_t0 < 0.35` AND `target_red_rate_t1h >= 0.50`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)


def evaluate_predictions(
    df: pd.DataFrame,
    pred_col: str,
    threshold: float = 0.50,
    true_congestion_cutoff: float = 0.50,
    onset_freeflow_max: float = 0.35,
) -> dict[str, float]:
    """Computes Primary Metric (F1) and Guardrails (Onset Recall, MAE, RMSE, Rainy F1, Rush F1)."""
    y_true = df["target_red_rate_t1h"].to_numpy(dtype=np.float64)
    y_pred = df[pred_col].to_numpy(dtype=np.float64)

    # Class 1 = Congested (red_rate >= threshold)
    y_true_cls = (y_true >= true_congestion_cutoff).astype(int)
    y_pred_cls = (y_pred >= threshold).astype(int)

    f1_all = f1_score(y_true_cls, y_pred_cls, zero_division=0)
    prec_all = precision_score(y_true_cls, y_pred_cls, zero_division=0)
    rec_all = recall_score(y_true_cls, y_pred_cls, zero_division=0)

    # Guardrail 1: Onset Recall (currently free-flowing < 0.35, congested in 1h >= 0.50)
    onset_mask = (df["red_rate_t0"].to_numpy() < onset_freeflow_max) & (y_true_cls == 1)
    onset_recall = (
        float(recall_score(y_true_cls[onset_mask], y_pred_cls[onset_mask], zero_division=0))
        if np.sum(onset_mask) > 0
        else 0.0
    )

    # Guardrail 2: Regression MAE & RMSE on continuous red_rate [0, 1]
    mae_all = float(mean_absolute_error(y_true, y_pred))
    rmse_all = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Situational F1: Rainy (precip > 1mm or 3h > 5mm), Rush Hour, and Daytime Active Probe (06:00-23:00, Proposal Sec 13)
    rain_mask = (df["precip_mm_t0"].to_numpy() > 1.0) | (df["precip_3h_mm_t0"].to_numpy() > 5.0)
    rush_mask = (df["is_morning_rush_t1h"].to_numpy() == 1) | (df["is_evening_rush_t1h"].to_numpy() == 1)
    if "target_hour" in df.columns:
        th_arr = df["target_hour"].to_numpy()
        daytime_mask = (th_arr >= 6) & (th_arr <= 23)
    else:
        daytime_mask = np.ones(len(df), dtype=bool)

    f1_rain = (
        float(f1_score(y_true_cls[rain_mask], y_pred_cls[rain_mask], zero_division=0))
        if np.sum(rain_mask) > 0
        else 0.0
    )
    f1_rush = (
        float(f1_score(y_true_cls[rush_mask], y_pred_cls[rush_mask], zero_division=0))
        if np.sum(rush_mask) > 0
        else 0.0
    )
    f1_daytime = (
        float(f1_score(y_true_cls[daytime_mask], y_pred_cls[daytime_mask], zero_division=0))
        if np.sum(daytime_mask) > 0
        else float(f1_all)
    )
    mae_daytime = (
        float(mean_absolute_error(y_true[daytime_mask], y_pred[daytime_mask]))
        if np.sum(daytime_mask) > 0
        else float(mae_all)
    )

    return {
        "f1_all": round(float(f1_all), 4),
        "f1_daytime_06_23": round(float(f1_daytime), 4),
        "precision_all": round(float(prec_all), 4),
        "recall_all": round(float(rec_all), 4),
        "onset_recall": round(float(onset_recall), 4),
        "f1_rush_hour": round(float(f1_rush), 4),
        "f1_rainy": round(float(f1_rain), 4),
        "mae": round(float(mae_all), 4),
        "mae_daytime_06_23": round(float(mae_daytime), 4),
        "rmse": round(float(rmse_all), 4),
        "threshold_used": round(float(threshold), 4),
        "onset_support": int(np.sum(onset_mask)),
    }


def tune_congestion_threshold(
    val_df: pd.DataFrame,
    pred_col: str,
    search_min: float = 0.30,
    search_max: float = 0.65,
    num_steps: int = 36,
    min_onset_recall: float = 0.55,
) -> tuple[float, dict[str, float]]:
    """
    Finds the classification threshold on Validation set that maximizes F1_All
    subject to Onset Recall >= min_onset_recall guardrail.
    """
    best_th = 0.50
    best_score = -1.0
    best_metrics: dict[str, float] = {}

    for th in np.linspace(search_min, search_max, num_steps):
        m = evaluate_predictions(val_df, pred_col, threshold=float(th))
        meets_guardrail = m["onset_recall"] >= min_onset_recall
        composite = m["f1_all"] + (0.25 if meets_guardrail else 0.15 * m["onset_recall"])
        if composite > best_score:
            best_score = composite
            best_th = float(th)
            best_metrics = m

    return round(best_th, 4), best_metrics
