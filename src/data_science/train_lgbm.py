"""
B2 Model Training Engine (LightGBM / Histogram Gradient Boosting Regressor).
Trains 1-Hour-Ahead Congestion Ratio predictor, tunes decision threshold on Validation set
with Onset Recall guardrail, compares against Baselines B0 & B1, and saves artifacts.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data_engineering.build_features import ALL_FEATURE_COLS, FEATURE_GROUPS
from src.data_science.baselines import evaluate_baselines
from src.data_science.evaluate_metrics import evaluate_predictions, tune_congestion_threshold
from src.utils.logger import log_info, log_step, log_success
from src.utils.storage import ROOT_DIR, load_config, load_table, save_table


def _build_regressor(hp: dict[str, Any]) -> tuple[Any, str]:
    """Uses LightGBM if installed; otherwise uses scikit-learn HistGradientBoostingRegressor (LightGBM algorithm)."""
    try:
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            n_estimators=hp.get("n_estimators", 350),
            learning_rate=hp.get("learning_rate", 0.05),
            max_depth=hp.get("max_depth", 8),
            num_leaves=hp.get("num_leaves", 63),
            min_child_samples=hp.get("min_child_samples", 30),
            subsample=hp.get("subsample", 0.8),
            colsample_bytree=hp.get("colsample_bytree", 0.8),
            random_state=hp.get("random_state", 42),
            verbose=-1,
        )
        return model, "LightGBMRegressor"
    except ImportError:
        model = HistGradientBoostingRegressor(
            max_iter=hp.get("n_estimators", 350),
            learning_rate=hp.get("learning_rate", 0.05),
            max_depth=hp.get("max_depth", 8),
            max_leaf_nodes=hp.get("num_leaves", 63),
            min_samples_leaf=hp.get("min_child_samples", 30),
            l2_regularization=1.0,
            random_state=hp.get("random_state", 42),
        )
        return model, "HistGradientBoostingRegressor (LightGBM-Tree)"


def run_train_and_evaluate() -> dict[str, Any]:
    log_step("TRAIN_MODEL", "เทรนโมเดล B2 (Gradient Boosted Trees) และปรับจูน Threshold บนชุด Validation...")
    cfg = load_config()
    gold_path = ROOT_DIR / cfg["paths"]["gold_dir"] / "gold_features.parquet"
    df_gold = load_table(gold_path)

    # 1. Evaluate Baselines B0 & B1
    baseline_results = evaluate_baselines(df_gold)

    # 2. Filter enabled feature groups from config
    enabled_cfg = cfg["feature_engineering"]["enabled_groups"]
    feature_cols: list[str] = []
    for grp_name, cols in FEATURE_GROUPS.items():
        if enabled_cfg.get(grp_name, True):
            feature_cols.extend(cols)

    df_train = df_gold[df_gold["split"] == "train"].copy()
    df_val = df_gold[df_gold["split"] == "val"].copy()
    df_test = df_gold[df_gold["split"] == "test"].copy()

    X_train = df_train[feature_cols]
    y_train = df_train["target_ratio_t60"].to_numpy(dtype=np.float64)

    # Upweight onset events & rainy/congested transitions during training to sharpen onset sensitivity
    sample_weights = np.ones(len(df_train), dtype=np.float64)
    sample_weights += 1.5 * df_train["is_onset_event"].to_numpy()
    sample_weights += 0.5 * (df_train["rain_mm_t0"].to_numpy() > 1.0)

    hp = cfg["model"]["hyperparameters"]
    model, backend_name = _build_regressor(hp)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # 3. Generate predictions across all splits
    df_gold["pred_b0_persistence"] = df_gold["ratio_t0"]
    df_gold["pred_b1_historical"] = df_gold["hist_avg_ratio_target_slot"]
    df_gold["pred_b2_lgbm"] = np.clip(model.predict(df_gold[feature_cols]), 0.0, 1.2).round(4)

    # 4. Tune classification threshold on Validation split ONLY
    th_cfg = cfg["model"]["threshold_tuning"]
    optimal_th, val_tuned_metrics = tune_congestion_threshold(
        df_gold[df_gold["split"] == "val"],
        pred_ratio_col="pred_b2_lgbm",
        search_min=float(th_cfg["search_min"]),
        search_max=float(th_cfg["search_max"]),
        num_steps=int(th_cfg["num_steps"]),
        min_onset_recall=float(th_cfg["min_onset_recall_guardrail"]),
    )

    df_gold["pred_is_congested_t60"] = (df_gold["pred_b2_lgbm"] < optimal_th).astype(int)
    df_gold["pred_is_onset_alert"] = (
        (df_gold["ratio_t0"] >= cfg["feature_engineering"]["onset_freeflow_buffer"])
        & (df_gold["pred_is_congested_t60"] == 1)
    ).astype(int)

    # Save enriched predictions back to Gold store for Dashboard & Monitoring
    save_table(df_gold, gold_path, layer="gold")

    # 5. Evaluate B2 across Train, Val, Test
    b2_metrics = {}
    for sp in ["train", "val", "test"]:
        sub = df_gold[df_gold["split"] == sp]
        b2_metrics[sp] = evaluate_predictions(sub, "pred_b2_lgbm", threshold=optimal_th)

    # Save candidate artifact for Model Registry
    eval_dir = ROOT_DIR / cfg["paths"]["evaluation_dir"]
    eval_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = ROOT_DIR / cfg["paths"]["registry_dir"] / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    candidate_model_path = candidate_dir / "latest_challenger.joblib"
    joblib.dump(
        {
            "model": model,
            "backend": backend_name,
            "feature_cols": feature_cols,
            "optimal_threshold": optimal_th,
            "hyperparameters": hp,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        candidate_model_path,
    )

    eval_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_backend": backend_name,
        "num_features": len(feature_cols),
        "feature_columns": feature_cols,
        "optimal_threshold_val": optimal_th,
        "models": {
            "B0_Persistence": baseline_results["B0_Persistence"],
            "B1_Historical_Average": baseline_results["B1_Historical_Average"],
            "B2_LightGBM": b2_metrics,
        },
    }

    with open(eval_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)

    test_m = b2_metrics["test"]
    log_info(
        f"Optimal Threshold (Val) = {optimal_th:.4f} | "
        f"[Test] B2 LightGBM -> F1={test_m['f1_all']:.4f}, Onset Recall={test_m['onset_recall']:.4f}, "
        f"Rainy F1={test_m['f1_rainy']:.4f}, Rush F1={test_m['f1_rush_hour']:.4f}, MAE={test_m['mae_ratio']:.4f}"
    )
    log_success("เทรนและประเมินผลโมเดล B0, B1, B2 เสร็จสมบูรณ์")
    return eval_summary
