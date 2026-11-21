"""
B2 Model Training & 1-Hour-Ahead Forecasting Engine (LightGBM Regressor) on Real iTIC Data.
Trains Congestion Rate predictor (0-1) with pruned active features + native road/district categoricals,
tunes decision threshold on Validation set with Onset Recall guardrail, compares against B0 & B1,
and saves clean model bundle & evaluation metrics.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data_science.baselines import evaluate_baselines
from src.data_science.evaluate_metrics import evaluate_predictions, tune_congestion_threshold
from src.data_science.features import FEATURE_GROUPS
from src.utils.logger import log_info, log_step, log_success
from src.utils.storage import ROOT_DIR, load_config, load_table


def run_train_and_evaluate() -> dict[str, Any]:
    log_step("TRAIN_LGBM", "เทรนโมเดล B2 (LightGBM Regressor 1-Hour Ahead) บนข้อมูล iTIC จริง...")
    cfg = load_config()
    df = load_table(cfg["paths"]["gold_data"])

    # 1. Evaluate Baselines B0 & B1
    baseline_results = evaluate_baselines(df)

    # 2. Select active feature groups (+ optional native LightGBM categorical IDs)
    enabled_cfg = cfg["feature_engineering"]["enabled_groups"]
    feature_cols: list[str] = [
        col
        for grp_name, cols in FEATURE_GROUPS.items()
        if enabled_cfg.get(grp_name, True)
        for col in cols
    ]

    work = df.copy()
    cat_dtypes: dict[str, pd.CategoricalDtype] = {}
    if cfg["feature_engineering"].get("use_categorical_ids", True):
        cats = sorted(work["link_id"].dropna().unique().tolist())
        dtype = pd.CategoricalDtype(categories=cats)
        work["link_cat"] = work["link_id"].astype(dtype)
        cat_dtypes["link_cat"] = dtype
        feature_cols.append("link_cat")

    df_train = work[work["split"] == "train"]
    df_val = work[work["split"] == "val"]

    X_train = df_train[feature_cols]
    y_train = df_train["target_red_rate_t1h"].to_numpy(dtype=np.float32)

    X_val = df_val[feature_cols]
    y_val = df_val["target_red_rate_t1h"].to_numpy(dtype=np.float32)

    # Mild onset & rainy sample weighting to sharpen onset recall without inflating MAE
    sw_cfg = cfg["model"].get("sample_weights", {})
    onset_boost = float(sw_cfg.get("onset_boost", 0.5))
    rainy_boost = float(sw_cfg.get("rainy_boost", 0.25))
    sample_weights = np.ones(len(df_train), dtype=np.float32)
    sample_weights += onset_boost * df_train["is_onset_event"].to_numpy(dtype=np.float32)
    sample_weights += rainy_boost * (df_train["precip_mm_t0"].to_numpy(dtype=np.float32) > 1.0)

    hp = cfg["model"]["hyperparameters"]
    early_stop = int(hp.get("early_stopping_rounds", 50))
    model = lgb.LGBMRegressor(
        n_estimators=int(hp.get("n_estimators", 800)),
        learning_rate=float(hp.get("learning_rate", 0.03)),
        max_depth=int(hp.get("max_depth", 8)),
        num_leaves=int(hp.get("num_leaves", 63)),
        min_child_samples=int(hp.get("min_child_samples", 50)),
        subsample=float(hp.get("subsample", 0.8)),
        subsample_freq=int(hp.get("subsample_freq", 1)),
        colsample_bytree=float(hp.get("colsample_bytree", 0.8)),
        reg_alpha=float(hp.get("reg_alpha", 0.5)),
        reg_lambda=float(hp.get("reg_lambda", 2.0)),
        random_state=int(hp.get("random_state", 42)),
        n_jobs=int(hp.get("n_jobs", -1)),
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_X=X_val,
        eval_y=y_val,
        callbacks=[lgb.early_stopping(stopping_rounds=early_stop, verbose=False)],
    )
    backend_name = f"LightGBMRegressor (v{lgb.__version__}, best_iter={model.best_iteration_})"

    # 3. Generate 1-hour-ahead forecasts across all splits
    work["pred_b2_lgbm"] = np.clip(model.predict(work[feature_cols]), 0.0, 1.0).astype(np.float32)

    # 4. Tune classification threshold on Validation split ONLY
    th_cfg = cfg["model"]["threshold_tuning"]
    optimal_th, _ = tune_congestion_threshold(
        work[work["split"] == "val"],
        pred_col="pred_b2_lgbm",
        search_min=float(th_cfg["search_min"]),
        search_max=float(th_cfg["search_max"]),
        num_steps=int(th_cfg["num_steps"]),
        min_onset_recall=float(th_cfg["min_onset_recall_guardrail"]),
    )

    # 5. Evaluate B2 across Train, Val, Test
    b2_metrics = {
        sp: evaluate_predictions(work[work["split"] == sp], "pred_b2_lgbm", threshold=optimal_th)
        for sp in ("train", "val", "test")
    }

    # 5b. Rolling-Origin 3-Fold Temporal Robustness Check (Proposal Section 11 Page 12: F1 within +-0.05 across 3 folds)
    df_test_eval = work[work["split"] == "test"].sort_values("target_ts")
    unique_test_days = sorted(df_test_eval["target_ts"].dt.strftime("%Y-%m-%d").unique())
    fold_chunks = np.array_split(unique_test_days, 3)
    rolling_folds: list[dict[str, Any]] = []
    for idx, days_chunk in enumerate(fold_chunks, start=1):
        days_list = list(days_chunk)
        sub_fold = df_test_eval[df_test_eval["target_ts"].dt.strftime("%Y-%m-%d").isin(days_list)]
        fm = evaluate_predictions(sub_fold, "pred_b2_lgbm", threshold=optimal_th)
        rolling_folds.append(
            {
                "fold": f"Fold {idx} ({days_list[0]} .. {days_list[-1]})",
                "num_days": len(days_list),
                "rows": int(len(sub_fold)),
                "f1_all": fm["f1_all"],
                "f1_daytime_06_23": fm["f1_daytime_06_23"],
                "onset_recall": fm["onset_recall"],
                "mae": fm["mae"],
            }
        )
    fold_f1s = [f["f1_all"] for f in rolling_folds]
    mean_fold_f1 = float(np.mean(fold_f1s))
    max_dev = round(float(max(abs(x - mean_fold_f1) for x in fold_f1s)), 4)
    robustness_report = {
        "experiment": "Rolling-Origin 3-Fold Temporal Robustness (Proposal Section 11 Page 12)",
        "acceptance_criterion": "F1 deviation <= +-0.05 across folds",
        "mean_fold_f1": round(mean_fold_f1, 4),
        "max_abs_deviation": max_dev,
        "passed": bool(max_dev <= 0.05),
        "folds": rolling_folds,
    }

    # 6. Save model bundle & evaluation summary
    models_dir = ROOT_DIR / cfg["paths"]["models_dir"]
    eval_dir = ROOT_DIR / cfg["paths"]["evaluation_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "b2_lgbm_model.joblib"
    joblib.dump(
        {
            "model": model,
            "backend": backend_name,
            "feature_cols": feature_cols,
            "cat_dtypes": cat_dtypes,
            "optimal_threshold": optimal_th,
            "hyperparameters": hp,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        model_path,
    )

    eval_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": cfg["paths"]["gold_data"],
        "num_rows_total": int(len(df)),
        "num_links": int(df["link_id"].nunique()),
        "split_counts": {k: int(v) for k, v in df["split"].value_counts().to_dict().items()},
        "model_backend": backend_name,
        "num_features": len(feature_cols),
        "feature_columns": feature_cols,
        "optimal_threshold_val": optimal_th,
        "models": {
            "B0_Persistence": baseline_results["B0_Persistence"],
            "B1_Historical_Average": baseline_results["B1_Historical_Average"],
            "B2_LightGBM": b2_metrics,
        },
        "robustness_3fold": robustness_report,
    }

    with open(eval_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)

    test_m = b2_metrics["test"]
    log_info(
        f"Backend = {backend_name} | Optimal Threshold (Val) = {optimal_th:.4f} | "
        f"[Test] B2 LightGBM -> F1={test_m['f1_all']:.4f}, Onset Recall={test_m['onset_recall']:.4f}, "
        f"Rainy F1={test_m['f1_rainy']:.4f}, Rush F1={test_m['f1_rush_hour']:.4f}, MAE={test_m['mae']:.4f}"
    )
    log_success(f"บันทึกโมเดลที่ {model_path.relative_to(ROOT_DIR)} และผลประเมินเรียบร้อย")
    return eval_summary
