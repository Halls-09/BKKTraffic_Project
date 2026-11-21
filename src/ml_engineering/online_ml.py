"""
Online / Incremental Micro-Batch Machine Learning Engine (LightGBM Warm-Start Continuous Learning).
Simulates real-time / daily micro-batch arrival across the September 2022 Test stream:
  1) At each window t, predicts 1-hour-ahead congestion (`t + 60m`) for all 448 directional road segments.
  2) Once ground truth (`target_red_rate_t1h`) arrives 60 minutes later, updates the Online Challenger
     via LightGBM Incremental Boosting (`init_model=current_booster`) on a sliding recent buffer
     with Monsoon & Onset weighting.
  3) Compares Static Batch Champion vs Online Incremental Model on F1, Daytime F1, Onset Recall,
     Rainy F1, MAE, and micro-batch update latency (ms).
"""
from __future__ import annotations

import copy
import json
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data_science.evaluate_metrics import evaluate_predictions
from src.data_science.forecast import _prepare_model_matrix
from src.utils.logger import log_info, log_step, log_success
from src.utils.storage import ROOT_DIR, load_config, load_table


def run_online_ml_simulation() -> dict[str, Any]:
    log_step("ONLINE_ML", "จำลองระบบ Online / Incremental ML (LightGBM Warm-Start Micro-Batch Updates)...")
    cfg = load_config()
    df = load_table(cfg["paths"]["gold_data"])
    model_path = ROOT_DIR / cfg["paths"]["models_dir"] / "b2_lgbm_model.joblib"
    bundle = joblib.load(model_path)

    static_model = bundle["model"]
    feature_cols: list[str] = bundle["feature_cols"]
    cat_dtypes = bundle.get("cat_dtypes")
    threshold = float(bundle["optimal_threshold"])

    df_val = df[df["split"] == "val"].sort_values("target_ts").copy()
    df_test = df[df["split"] == "test"].sort_values("target_ts").copy()

    # Pre-compute Static Batch predictions on Test
    X_test_all = _prepare_model_matrix(df_test, feature_cols, cat_dtypes)
    df_test["pred_static_batch"] = np.clip(static_model.predict(X_test_all), 0.0, 1.0).astype(np.float32)

    # Partition Test into 6 sequential 4-day micro-batch windows (simulating Airflow scheduled online updates)
    df_test["date_str"] = df_test["target_ts"].dt.strftime("%Y-%m-%d")
    unique_days = sorted(df_test["date_str"].unique())
    day_Windows = np.array_split(unique_days, 6)

    online_booster = copy.deepcopy(static_model.booster_)
    recent_buffer_df = df_val.tail(60000).copy()

    online_preds_list: list[pd.Series] = []
    window_logs: list[dict[str, Any]] = []
    update_latencies_ms: list[float] = []

    for w_idx, days_arr in enumerate(day_Windows, start=1):
        days_list = list(days_arr)
        w_df = df_test[df_test["date_str"].isin(days_list)].copy()
        X_w = _prepare_model_matrix(w_df, feature_cols, cat_dtypes)

        # 1. Online Inference BEFORE seeing ground truth of this window (Zero Leakage!)
        pred_online_w = np.clip(online_booster.predict(X_w), 0.0, 1.0).astype(np.float32)
        w_df["pred_online_ml"] = pred_online_w
        online_preds_list.append(pd.Series(pred_online_w, index=w_df.index))

        m_static_w = evaluate_predictions(w_df, "pred_static_batch", threshold=threshold)
        m_online_w = evaluate_predictions(w_df, "pred_online_ml", threshold=threshold)

        # 2. Once ground truth arrives (t + 60m), perform Warm-Start Incremental Update on sliding buffer
        t0 = time.perf_counter()
        recent_buffer_df = pd.concat([recent_buffer_df, w_df], ignore_index=True).tail(90000)
        X_buf = _prepare_model_matrix(recent_buffer_df, feature_cols, cat_dtypes)
        y_buf = recent_buffer_df["target_red_rate_t1h"].to_numpy(dtype=np.float32)

        w_buf = np.ones(len(recent_buffer_df), dtype=np.float32)
        w_buf += 0.55 * recent_buffer_df["is_onset_event"].to_numpy(dtype=np.float32)
        w_buf += 0.35 * (recent_buffer_df["precip_mm_t0"].to_numpy(dtype=np.float32) > 1.0)
        # Decay weight so newest micro-batch has 1.5x influence
        recency = np.linspace(0.75, 1.35, len(recent_buffer_df), dtype=np.float32)
        w_buf *= recency

        inc_reg = lgb.LGBMRegressor(
            n_estimators=12,
            learning_rate=0.012,
            max_depth=8,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42 + w_idx,
            n_jobs=-1,
            verbose=-1,
        )
        inc_reg.fit(X_buf, y_buf, sample_weight=w_buf, init_model=online_booster)
        online_booster = inc_reg.booster_
        dur_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        update_latencies_ms.append(dur_ms)

        window_logs.append(
            {
                "window_id": f"W{w_idx} ({days_list[0][5:]}..{days_list[-1][5:]})",
                "rows": int(len(w_df)),
                "static_f1": m_static_w["f1_all"],
                "online_f1": m_online_w["f1_all"],
                "static_mae": m_static_w["mae"],
                "online_mae": m_online_w["mae"],
                "online_onset_recall": m_online_w["onset_recall"],
                "update_latency_ms": dur_ms,
                "num_trees_total": int(online_booster.num_trees()),
            }
        )

    df_test["pred_online_ml"] = pd.concat(online_preds_list).sort_index()
    static_overall = evaluate_predictions(df_test, "pred_static_batch", threshold=threshold)
    online_overall = evaluate_predictions(df_test, "pred_online_ml", threshold=threshold)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "Airflow Micro-Batch + LightGBM Warm-Start Incremental Boosting (init_model)",
        "online_schedule": "Predict every 15m -> Join actual at t+60m -> Incremental Warm-Start update on 90k sliding window",
        "mean_update_latency_ms": round(float(np.mean(update_latencies_ms)), 1),
        "initial_trees": int(static_model.booster_.num_trees()),
        "final_trees_after_stream": int(online_booster.num_trees()),
        "static_batch_metrics": static_overall,
        "online_incremental_metrics": online_overall,
        "f1_delta_online_vs_static": round(online_overall["f1_all"] - static_overall["f1_all"], 4),
        "mae_delta_online_vs_static": round(online_overall["mae"] - static_overall["mae"], 4),
        "stream_windows": window_logs,
    }

    eval_dir = ROOT_DIR / cfg["paths"]["evaluation_dir"]
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / "online_ml_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log_info(
        f"Static Batch F1={static_overall['f1_all']:.4f} (MAE={static_overall['mae']:.4f}) vs "
        f"Online Incremental F1={online_overall['f1_all']:.4f} (MAE={online_overall['mae']:.4f}, "
        f"Onset={online_overall['onset_recall']:.4f}, Avg Update={report['mean_update_latency_ms']}ms)"
    )
    log_success("บันทึกผลจำลอง Online ML ที่ artifacts/evaluation/online_ml_report.json สำเร็จ")
    return report
