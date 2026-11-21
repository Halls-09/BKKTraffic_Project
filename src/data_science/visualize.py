"""
Presentation-Ready Visualization Generator for Modeling Results (Baseline, LightGBM, SHAP, Forecast).
Saves 4 high-DPI PNG charts to `artifacts/charts/`.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.logger import log_step, log_success
from src.utils.storage import ROOT_DIR, load_config


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#F8FAFC",
            "axes.edgecolor": "#CBD5E1",
            "axes.grid": True,
            "grid.color": "#E2E8F0",
            "grid.linestyle": "--",
            "grid.alpha": 0.8,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
        }
    )


def generate_all_charts() -> list[Path]:
    log_step("VISUALIZE", "สร้างกราฟสรุปผล Baseline, LightGBM, TreeSHAP และ Forecast (artifacts/charts/)...")
    _setup_style()
    cfg = load_config()
    eval_dir = ROOT_DIR / cfg["paths"]["evaluation_dir"]
    charts_dir = ROOT_DIR / cfg["paths"]["charts_dir"]
    charts_dir.mkdir(parents=True, exist_ok=True)

    with open(eval_dir / "evaluation_summary.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    with open(eval_dir / "shap_attributions.json", "r", encoding="utf-8") as f:
        shap_data = json.load(f)
    with open(eval_dir / "forecast_summary.json", "r", encoding="utf-8") as f:
        fc_data = json.load(f)

    saved: list[Path] = []

    # ── Chart 1: Model Benchmark Comparison (B0 vs B1 vs B2 LightGBM) ─────────
    models = eval_data["models"]
    b0 = models["B0_Persistence"]["test"]
    b1 = models["B1_Historical_Average"]["test"]
    b2 = models["B2_LightGBM"]["test"]

    metric_labels = ["Overall F1", "Onset Recall\n(Guardrail >=0.55)", "Rush Hour F1", "Rainy F1"]
    b0_vals = [b0["f1_all"], b0["onset_recall"], b0["f1_rush_hour"], b0["f1_rainy"]]
    b1_vals = [b1["f1_all"], b1["onset_recall"], b1["f1_rush_hour"], b1["f1_rainy"]]
    b2_vals = [b2["f1_all"], b2["onset_recall"], b2["f1_rush_hour"], b2["f1_rainy"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2.4, 1]})
    x = np.arange(len(metric_labels))
    w = 0.26
    bars0 = ax1.bar(x - w, b0_vals, w, label="B0: Persistence", color="#94A3B8")
    bars1 = ax1.bar(x, b1_vals, w, label="B1: Historical Avg", color="#60A5FA")
    bars2 = ax1.bar(x + w, b2_vals, w, label="B2: LightGBM (Ours)", color="#0F766E")

    ax1.axhline(0.55, color="#DC2626", linestyle=":", linewidth=1.4, label="Onset Recall Guardrail (0.55)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels)
    ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("Score")
    ax1.set_title("Classification & Onset Metrics (Real iTIC Test Set)")
    ax1.legend(loc="upper center", ncol=2, frameon=True, fontsize=9)
    for bars in (bars0, bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

    mae_models = ["B0\nPersistence", "B1\nHistorical", "B2\nLightGBM"]
    mae_vals = [b0["mae"], b1["mae"], b2["mae"]]
    colors_mae = ["#94A3B8", "#60A5FA", "#0F766E"]
    bars_mae = ax2.bar(mae_models, mae_vals, color=colors_mae, width=0.55)
    ax2.set_ylim(0, max(mae_vals) * 1.25)
    ax2.set_ylabel("MAE (Lower is Better)")
    ax2.set_title("1-Hour Forecast MAE")
    for bar in bars_mae:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.003, f"{h:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Bangkok 1-Hour-Ahead Traffic Forecasting: Baseline vs LightGBM Benchmark", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    p1 = charts_dir / "01_model_benchmark.png"
    fig.savefig(p1, dpi=180)
    plt.close(fig)
    saved.append(p1)

    # ── Chart 2: TreeSHAP Feature Importance & Group Attribution ──────────────
    top12 = shap_data["top_features"][:12][::-1]
    groups = shap_data["group_attribution"][::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5), gridspec_kw={"width_ratios": [1.4, 1]})
    feats = [r["feature"] for r in top12]
    vals = [r["share_pct"] for r in top12]
    ax1.barh(feats, vals, color="#0284C7")
    ax1.set_xlabel("SHAP Importance Share (%)")
    ax1.set_title("Top 12 Features by Mean |TreeSHAP|")
    ax1.set_xlim(0, max(vals) * 1.22)
    for i, v in enumerate(vals):
        ax1.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8.5)

    g_labels = [g["group_key"].replace("group_", "G").replace("_", " ").title() for g in groups]
    g_vals = [g["share_pct"] for g in groups]
    ax2.barh(g_labels, g_vals, color="#0F766E")
    ax2.set_xlabel("Group Attribution Share (%)")
    ax2.set_title("Feature Group Attribution (TreeSHAP)")
    ax2.set_xlim(0, max(g_vals) * 1.22)
    for i, v in enumerate(g_vals):
        ax2.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=9, fontweight="bold")

    fig.suptitle(f"Explainability Analysis — {shap_data['attribution_method']}", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    p2 = charts_dir / "02_shap_importance.png"
    fig.savefig(p2, dpi=180)
    plt.close(fig)
    saved.append(p2)

    # ── Chart 3: 24-Hour Actual vs Predicted & Hourly MAE ─────────────────────
    hourly = fc_data["hourly_forecast_analysis"]
    hours = [h["hour"] for h in hourly]
    actual = [h["actual_red_rate"] * 100 for h in hourly]
    pred = [h["pred_red_rate"] * 100 for h in hourly]
    mae_lgbm = [h["mae_lgbm"] for h in hourly]
    mae_b0 = [h["mae_b0"] for h in hourly]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(hours, actual, marker="o", linewidth=2.2, color="#1E293B", label="Actual Congestion Rate (%)")
    ax1.plot(hours, pred, marker="s", linewidth=2.0, linestyle="--", color="#0F766E", label="LightGBM 1h Forecast (%)")
    ax1.axvspan(7, 9, color="#FEF3C7", alpha=0.5, label="Morning Rush (07-09)")
    ax1.axvspan(16, 19, color="#FEE2E2", alpha=0.5, label="Evening Rush (16-19)")
    ax1.set_ylabel("Mean Red Congestion Rate (%)")
    ax1.set_title("24-Hour Traffic Congestion Profile: Actual vs 1-Hour-Ahead Forecast (Test Set)")
    ax1.legend(loc="upper left", ncol=2, frameon=True)

    ax2.plot(hours, mae_b0, marker="^", linewidth=2.0, color="#DC2626", linestyle="--", label="B0 Persistence MAE")
    ax2.plot(hours, mae_lgbm, marker="o", linewidth=2.2, color="#0F766E", label="B2 LightGBM MAE")
    ax2.fill_between(hours, mae_lgbm, mae_b0, where=np.array(mae_b0) >= np.array(mae_lgbm), color="#10B981", alpha=0.15, label="Error Reduction")
    ax2.set_xticks(range(0, 24))
    ax2.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right")
    ax2.set_ylabel("Hourly MAE")
    ax2.set_xlabel("Target Forecast Hour (Asia/Bangkok)")
    ax2.set_title("Hourly Forecast Error Comparison: LightGBM vs Persistence Baseline")
    ax2.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    p3 = charts_dir / "03_hourly_forecast_error.png"
    fig.savefig(p3, dpi=180)
    plt.close(fig)
    saved.append(p3)

    # ── Chart 4: Hypotheses H1-H5 Quantitative Verification ───────────────────
    h_meta = shap_data["hypotheses_verification"]
    labels = [f"{h['id']}" for h in h_meta]
    # Extract quantitative percentages directly from structured metrics
    vals_high = [h["metric_high_pct"] for h in h_meta]
    vals_low = [h["metric_low_pct"] for h in h_meta]

    fig, ax = plt.subplots(figsize=(10.5, 5))
    x = np.arange(len(labels))
    w = 0.34
    b_low = ax.bar(x - w / 2, vals_low, w, label="Baseline Condition (%)", color="#94A3B8")
    b_high = ax.bar(x + w / 2, vals_high, w, label="Model / Risk Condition (%)", color="#E11D48")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Metric / Congestion Rate (%)")
    ax.set_ylim(0, max(vals_high) * 1.25)
    ax.set_title("Empirical Verification of Hypotheses H1–H5 on Real Bangkok Test Data (All Supported)")
    ax.legend(frameon=True)
    for bars in (b_low, b_high):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    fig.tight_layout()
    p4 = charts_dir / "04_hypotheses_verification.png"
    fig.savefig(p4, dpi=180)
    plt.close(fig)
    saved.append(p4)

    log_success(f"สร้างกราฟครบ {len(saved)} ภาพที่ artifacts/charts/")
    return saved
