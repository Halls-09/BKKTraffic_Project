"""
Evaluation Metrics & Guardrail Threshold Optimization (Section 8 & 9.2 of Proposal).
Computes Primary Metric (F1 All) and Guardrails (Onset Recall, MAE, Rainy F1, Rush Hour F1).
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
    pred_ratio_col: str,
    threshold: float = 0.50,
    true_congestion_cutoff: float = 0.50,
    onset_freeflow_buffer: float = 0.55,
) -> dict[str, float]:
    """
    Computes Primary Metric (F1) and Guardrails (Onset Recall, MAE, Rain/Rush Hour F1).
    """
    y_true_ratio = df["target_ratio_t60"].to_numpy(dtype=np.float64)
    y_pred_ratio = df[pred_ratio_col].to_numpy(dtype=np.float64)

    # Class 1 = Congested (ratio < cutoff)
    y_true_cls = (y_true_ratio < true_congestion_cutoff).astype(int)
    y_pred_cls = (y_pred_ratio < threshold).astype(int)

    # 1. Primary Metric: F1-score on congested class
    f1_all = f1_score(y_true_cls, y_pred_cls, zero_division=0)
    prec_all = precision_score(y_true_cls, y_pred_cls, zero_division=0)
    rec_all = recall_score(y_true_cls, y_pred_cls, zero_division=0)

    # 2. Guardrail 1: Onset Recall (currently free-flowing >= 0.55, congested in 60m < 0.50)
    onset_mask = (df["ratio_t0"].to_numpy() >= onset_freeflow_buffer) & (y_true_cls == 1)
    onset_recall = (
        float(recall_score(y_true_cls[onset_mask], y_pred_cls[onset_mask], zero_division=0))
        if np.sum(onset_mask) > 0
        else 0.0
    )

    # 3. Guardrail 2: MAE & RMSE of continuous congestion ratio
    mae_all = float(mean_absolute_error(y_true_ratio, y_pred_ratio))
    rmse_all = float(np.sqrt(mean_squared_error(y_true_ratio, y_pred_ratio)))

    # 4. Situational F1: Rainy (> 1.0 mm) and Rush Hour
    rain_mask = (df["rain_mm_t0"].to_numpy() > 1.0) | (df["rain_cum_3h"].to_numpy() > 5.0)
    rush_mask = (df["is_morning_rush"].to_numpy() == 1) | (df["is_evening_rush"].to_numpy() == 1)

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

    return {
        "f1_all": round(float(f1_all), 4),
        "precision_all": round(float(prec_all), 4),
        "recall_all": round(float(rec_all), 4),
        "onset_recall": round(float(onset_recall), 4),
        "f1_rush_hour": round(float(f1_rush), 4),
        "f1_rainy": round(float(f1_rain), 4),
        "mae_ratio": round(float(mae_all), 4),
        "rmse_ratio": round(float(rmse_all), 4),
        "threshold_used": round(float(threshold), 4),
        "onset_support": int(np.sum(onset_mask)),
    }


def tune_congestion_threshold(
    val_df: pd.DataFrame,
    pred_ratio_col: str,
    search_min: float = 0.44,
    search_max: float = 0.58,
    num_steps: int = 29,
    min_onset_recall: float = 0.55,
) -> tuple[float, dict[str, float]]:
    """
    Finds the classification threshold on Validation set that maximizes F1_All
    subject to Onset Recall >= min_onset_recall guardrail.
    """
    best_th = 0.50
    best_score = -1.0
    best_metrics: dict[str, float] = {}

    candidates = np.linspace(search_min, search_max, num_steps)
    for th in candidates:
        m = evaluate_predictions(val_df, pred_ratio_col, threshold=float(th))
        # Prioritize candidates meeting the Onset Recall guardrail
        meets_guardrail = m["onset_recall"] >= min_onset_recall
        composite = m["f1_all"] + (0.25 if meets_guardrail else 0.15 * m["onset_recall"])
        if composite > best_score:
            best_score = composite
            best_th = float(th)
            best_metrics = m

    return round(best_th, 4), best_metrics
