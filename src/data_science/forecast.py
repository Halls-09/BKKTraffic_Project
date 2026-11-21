"""
1-Hour-Ahead Traffic Congestion Forecasting & Inference Module.
Provides:
  - `forecast_next_hour(df_features, model_path)`: Reusable inference function for ML Engineer / Serving.
  - `run_forecast_report()`: Generates Test-set 1-hour-ahead forecasts, hourly error analysis,
    top congested road hotspots, and saves `artifacts/evaluation/forecast_summary.json`
    + `artifacts/evaluation/sample_test_forecasts.csv`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.utils.logger import log_info, log_step, log_success
from src.utils.storage import ROOT_DIR, load_config, load_table


def _prepare_model_matrix(df: pd.DataFrame, feature_cols: list[str], cat_dtypes: dict[str, Any] | None = None) -> pd.DataFrame:
    """Builds model input matrix, restoring exact categorical dtypes if used during training."""
    work = df.copy()
    if "link_cat" in feature_cols and "link_cat" not in work.columns:
        work["link_cat"] = work["link_id"]
    if "dcode_cat" in feature_cols and "dcode_cat" not in work.columns:
        work["dcode_cat"] = work["dcode"]

    X = work[feature_cols].copy()
    if cat_dtypes:
        for col, dtype in cat_dtypes.items():
            if col in X.columns:
                X[col] = pd.Categorical(X[col], categories=dtype.categories)
    return X


def forecast_next_hour(
    df_features: pd.DataFrame,
    model_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Predicts 1-hour-ahead congestion rate (`pred_red_rate_t1h`), binary congestion alert
    (`pred_is_congested_t1h`), and congestion onset warning (`pred_is_onset_alert`).
    """
    cfg = load_config()
    if model_path is None:
        model_path = ROOT_DIR / cfg["paths"]["models_dir"] / "b2_lgbm_model.joblib"

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols: list[str] = bundle["feature_cols"]
    threshold: float = float(bundle["optimal_threshold"])
    cat_dtypes: dict[str, Any] | None = bundle.get("cat_dtypes")
    onset_max = float(cfg["dataset"]["onset_freeflow_max"])

    X = _prepare_model_matrix(df_features, feature_cols, cat_dtypes)
    preds = np.clip(model.predict(X), 0.0, 1.0).astype(np.float32)

    out = df_features.copy()
    out["pred_red_rate_t1h"] = np.round(preds, 4)
    out["pred_is_congested_t1h"] = (preds >= threshold).astype(np.int8)
    out["pred_is_onset_alert"] = (
        (out["red_rate_t0"].to_numpy() < onset_max) & (out["pred_is_congested_t1h"].to_numpy() == 1)
    ).astype(np.int8)
    out["congestion_risk_level"] = pd.cut(
        out["pred_red_rate_t1h"],
        bins=[-0.01, 0.25, threshold, 0.70, 1.01],
        labels=["LOW", "MODERATE", "HIGH", "SEVERE"],
    ).astype(str)
    return out


def run_forecast_report() -> dict[str, Any]:
    log_step("FORECAST", "สร้างผลพยากรณ์ล่วงหน้า 1 ชั่วโมง (Test Set), วิเคราะห์รายชั่วโมง และจัดอันดับจุดรถติดสูงสุด...")
    cfg = load_config()
    df = load_table(cfg["paths"]["gold_data"])
    df_test = df[df["split"] == "test"].copy()

    df_pred = forecast_next_hour(df_test)

    # 1. Hourly forecast vs actual & MAE comparison (B2 LightGBM vs B0 Persistence)
    df_pred["abs_err_lgbm"] = np.abs(df_pred["target_red_rate_t1h"] - df_pred["pred_red_rate_t1h"])
    df_pred["abs_err_b0"] = np.abs(df_pred["target_red_rate_t1h"] - df_pred["red_rate_t0"])

    hourly = (
        df_pred.groupby("target_hour", as_index=False)
        .agg(
            actual_red_rate=("target_red_rate_t1h", "mean"),
            pred_red_rate=("pred_red_rate_t1h", "mean"),
            mae_lgbm=("abs_err_lgbm", "mean"),
            mae_b0=("abs_err_b0", "mean"),
            actual_congested_pct=("target_is_congested_t1h", "mean"),
            onset_alerts=("pred_is_onset_alert", "sum"),
        )
    )
    hourly_records = [
        {
            "hour": int(r["target_hour"]),
            "hour_label": f"{int(r['target_hour']):02d}:00",
            "actual_red_rate": round(float(r["actual_red_rate"]), 4),
            "pred_red_rate": round(float(r["pred_red_rate"]), 4),
            "mae_lgbm": round(float(r["mae_lgbm"]), 4),
            "mae_b0": round(float(r["mae_b0"]), 4),
            "actual_congested_pct": round(float(r["actual_congested_pct"]) * 100.0, 2),
            "onset_alerts": int(r["onset_alerts"]),
        }
        for _, r in hourly.iterrows()
    ]

    # 2. Top 15 congested road links (Hotspots on Test set)
    hotspots_df = (
        df_pred.groupby(["link_id", "district", "district_en"], as_index=False)
        .agg(
            road_name=("segment_label" if "segment_label" in df_pred.columns else "road_name_th", "first"),
            location_name=("first_name_th", "first"),
            mean_actual_red=("target_red_rate_t1h", "mean"),
            mean_pred_red=("pred_red_rate_t1h", "mean"),
            congested_hours_pct=("target_is_congested_t1h", "mean"),
            mae=("abs_err_lgbm", "mean"),
        )
        .sort_values("mean_actual_red", ascending=False)
        .head(15)
    )
    top_hotspots = [
        {
            "link_id": str(r["link_id"]),
            "district": str(r["district"]),
            "district_en": str(r["district_en"]),
            "road_name": str(r["road_name"]) if pd.notna(r["road_name"]) else str(r["location_name"]),
            "mean_actual_red": round(float(r["mean_actual_red"]), 4),
            "mean_pred_red": round(float(r["mean_pred_red"]), 4),
            "congested_hours_pct": round(float(r["congested_hours_pct"]) * 100.0, 1),
            "mae": round(float(r["mae"]), 4),
        }
        for _, r in hotspots_df.iterrows()
    ]

    # 3. Top 10 districts by predicted evening-rush congestion
    evening_df = df_pred[df_pred["is_evening_rush_t1h"] == 1]
    top_districts_df = (
        evening_df.groupby(["district", "district_en"], as_index=False)
        .agg(
            evening_actual_red=("target_red_rate_t1h", "mean"),
            evening_pred_red=("pred_red_rate_t1h", "mean"),
            mae=("abs_err_lgbm", "mean"),
        )
        .sort_values("evening_actual_red", ascending=False)
        .head(10)
    )
    top_districts = [
        {
            "district": str(r["district"]),
            "district_en": str(r["district_en"]),
            "evening_actual_red": round(float(r["evening_actual_red"]), 4),
            "evening_pred_red": round(float(r["evening_pred_red"]), 4),
            "mae": round(float(r["mae"]), 4),
        }
        for _, r in top_districts_df.iterrows()
    ]

    # Save forecast summary JSON + latest-day sample CSV for inspection / MLE handoff
    eval_dir = ROOT_DIR / cfg["paths"]["evaluation_dir"]
    eval_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "test_rows_forecasted": int(len(df_pred)),
        "total_onset_alerts_predicted": int(df_pred["pred_is_onset_alert"].sum()),
        "hourly_forecast_analysis": hourly_records,
        "top_15_congested_hotspots": top_hotspots,
        "top_10_evening_rush_districts": top_districts,
    }
    with open(eval_dir / "forecast_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Export clean forecast sample (last 24 hours of Test split = 12,000 link-hour predictions)
    last_day = df_pred["target_ts"].max() - pd.Timedelta(hours=23)
    sample_cols = [
        "link_id", "district", "district_en", "road_name_th", "first_name_th",
        "ts_hour", "target_ts", "red_rate_t0", "target_red_rate_t1h",
        "pred_red_rate_t1h", "pred_is_congested_t1h", "pred_is_onset_alert", "congestion_risk_level",
    ]
    sample_out = df_pred[df_pred["target_ts"] >= last_day][sample_cols].sort_values(
        ["target_ts", "pred_red_rate_t1h"], ascending=[True, False]
    )
    sample_csv_path = eval_dir / "latest_24h_forecasts.csv"
    sample_out.to_csv(sample_csv_path, index=False, encoding="utf-8-sig")

    log_info(
        f"พยากรณ์ชุด Test ครบ {len(df_pred):,} แถว | แจ้งเตือน Onset Alert = {summary['total_onset_alerts_predicted']:,} ครั้ง"
    )
    log_success(f"บันทึก forecast_summary.json และ {sample_csv_path.relative_to(ROOT_DIR)} สำเร็จ")
    return summary
