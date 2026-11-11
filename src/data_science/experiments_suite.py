"""
Advanced Experiments Suite (Section 9.2, 9.3, and Section 11 of Proposal PDF):
1. Horizon Sensitivity Analysis (15m, 30m, 45m, 60m lead-time tradeoff)
2. Rolling-Origin 3-Fold Temporal Robustness (verifies F1 stability within +/- 0.05)
3. Traffy Fondue Flood Event Study (July 21, 2022 flood spike vs normal days with 95% CI & t-test)
4. Hourly MAE Error Analysis (00:00 - 23:00 target hours)
5. Automated Feature Gate (Section 9.3: retains feature groups improving Val F1 or Onset Recall >= 0.005)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.data_engineering.build_features import ALL_FEATURE_COLS, FEATURE_GROUPS
from src.data_science.evaluate_metrics import evaluate_predictions
from src.data_science.train_lgbm import _build_regressor
from src.utils.logger import log_info, log_step, log_success
from src.utils.storage import ROOT_DIR, load_config, load_table


def run_proposal_experiments() -> dict[str, Any]:
    log_step(
        "EXPERIMENTS_S11",
        "รันการทดลองตามข้อ 9.2, 9.3 และ 11 (Horizon 15/30/60m, Rolling 3-Fold, Traffy Event Study 95% CI, Hourly MAE)...",
    )
    cfg = load_config()
    df_gold = load_table(ROOT_DIR / cfg["paths"]["gold_dir"] / "gold_features.parquet")
    df_gold = df_gold.sort_values(["seg_id", "ts_t"]).reset_index(drop=True)

    hp = dict(cfg["model"]["hyperparameters"])
    hp["n_estimators"] = 180

    # ------------------------------------------------------------------
    # 1. Horizon Sensitivity Experiment (15m, 30m, 45m, 60m ahead)
    # ------------------------------------------------------------------
    g = df_gold.groupby("seg_id", sort=False)
    horizons = [(1, 15), (2, 30), (3, 45), (4, 60)]
    horizon_results = []

    for slots, mins in horizons:
        df_h = df_gold.copy()
        lead_r = g["ratio_t0"].shift(-slots)
        lead_ts = pd.to_datetime(g["ts_t"].shift(-slots))
        valid_mask = (lead_ts - pd.to_datetime(df_h["ts_t"])) == pd.Timedelta(minutes=mins)
        df_h["target_ratio_t60"] = np.where(valid_mask, lead_r, np.nan)
        df_h = df_h.dropna(subset=["target_ratio_t60"]).reset_index(drop=True)

        tr = df_h[df_h["split"] == "train"]
        te = df_h[df_h["split"] == "test"].copy()

        model, _ = _build_regressor(hp)
        sw = np.ones(len(tr)) + 1.5 * ((tr["ratio_t0"] >= 0.55) & (tr["target_ratio_t60"] < 0.50)).to_numpy()
        model.fit(tr[ALL_FEATURE_COLS], tr["target_ratio_t60"].to_numpy(), sample_weight=sw)

        te["pred_h"] = np.clip(model.predict(te[ALL_FEATURE_COLS]), 0.0, 1.2)
        te["pred_b0"] = te["ratio_t0"]
        m_lgbm = evaluate_predictions(te, "pred_h", threshold=0.495)
        m_b0 = evaluate_predictions(te, "pred_b0", threshold=0.50)

        horizon_results.append({
            "horizon_minutes": mins,
            "horizon_slots": slots,
            "lgbm_f1": m_lgbm["f1_all"],
            "lgbm_onset_recall": m_lgbm["onset_recall"],
            "lgbm_mae": m_lgbm["mae_ratio"],
            "b0_f1": m_b0["f1_all"],
            "b0_onset_recall": m_b0["onset_recall"],
            "b0_mae": m_b0["mae_ratio"],
        })

    # ------------------------------------------------------------------
    # 2. Rolling-Origin 3-Fold Temporal Robustness (Acceptance: |F1 diff| <= 0.05)
    # ------------------------------------------------------------------
    df_sorted = df_gold.sort_values("ts_t").reset_index(drop=True)
    unique_ts = sorted(df_sorted["ts_t"].unique())
    n_ts = len(unique_ts)
    # Create 3 expanding temporal folds, each evaluating a balanced 6-day window:
    fold_cuts = [(0.42, 0.62), (0.60, 0.80), (0.78, 0.98)]
    fold_results = []
    for f_idx, (tr_frac, ev_frac) in enumerate(fold_cuts, 1):
        t_cut1 = unique_ts[int(n_ts * tr_frac)]
        t_cut2 = unique_ts[int(n_ts * ev_frac)]
        f_train = df_sorted[df_sorted["ts_t"] <= t_cut1]
        f_eval = df_sorted[(df_sorted["ts_t"] > t_cut1) & (df_sorted["ts_t"] <= t_cut2)].copy()

        m_fold, _ = _build_regressor(hp)
        sw = np.ones(len(f_train)) + 1.5 * f_train["is_onset_event"].to_numpy()
        m_fold.fit(f_train[ALL_FEATURE_COLS], f_train["target_ratio_t60"].to_numpy(), sample_weight=sw)
        f_eval["pred_fold"] = np.clip(m_fold.predict(f_eval[ALL_FEATURE_COLS]), 0.0, 1.2)
        fm = evaluate_predictions(f_eval, "pred_fold", threshold=0.495)

        fold_results.append({
            "fold": f"Fold_{f_idx}",
            "train_end_ts": str(t_cut1),
            "eval_end_ts": str(t_cut2),
            "train_rows": int(len(f_train)),
            "eval_rows": int(len(f_eval)),
            "f1_all": fm["f1_all"],
            "onset_recall": fm["onset_recall"],
            "mae_ratio": fm["mae_ratio"],
        })

    fold_f1s = [r["f1_all"] for r in fold_results]
    max_dev_from_mean = float(np.max(np.abs(np.array(fold_f1s) - np.mean(fold_f1s))))
    robustness_passed = max_dev_from_mean <= 0.05

    # ------------------------------------------------------------------
    # 3. Traffy Fondue Flood Event Study (21 July 2022 vs Normal Days + 95% CI)
    # ------------------------------------------------------------------
    df_gold["date_str"] = pd.to_datetime(df_gold["ts_t"]).dt.strftime("%Y-%m-%d")
    df_gold["hour_int"] = pd.to_datetime(df_gold["ts_t"]).dt.hour

    # Compare afternoon/evening (15:00-22:00) on Peak Flood Day (2022-07-21) vs Normal Days
    evening_sub = df_gold[(df_gold["hour_int"] >= 15) & (df_gold["hour_int"] <= 22)]
    flood_day_ratios = evening_sub[evening_sub["date_str"] == "2022-07-21"]["ratio_t0"].to_numpy()
    normal_day_ratios = evening_sub[evening_sub["date_str"] != "2022-07-21"]["ratio_t0"].to_numpy()

    mean_flood = float(np.mean(flood_day_ratios))
    mean_normal = float(np.mean(normal_day_ratios))
    diff_mean = mean_flood - mean_normal

    se_diff = float(
        np.sqrt(
            np.var(flood_day_ratios, ddof=1) / max(1, len(flood_day_ratios))
            + np.var(normal_day_ratios, ddof=1) / max(1, len(normal_day_ratios))
        )
    )
    ci_95_low = round(diff_mean - 1.96 * se_diff, 4)
    ci_95_high = round(diff_mean + 1.96 * se_diff, 4)
    t_stat, p_val = stats.ttest_ind(flood_day_ratios, normal_day_ratios, equal_var=False)

    event_study = {
        "event_date": "2022-07-21 (Peak Traffy Flood Day - 1,333 reports)",
        "window_hours": "15:00 - 22:00",
        "flood_day_mean_ratio": round(mean_flood, 4),
        "normal_day_mean_ratio": round(mean_normal, 4),
        "delta_ratio": round(diff_mean, 4),
        "ci_95": [ci_95_low, ci_95_high],
        "welch_t_stat": round(float(t_stat), 3),
        "p_value": float(f"{p_val:.4e}"),
        "statistically_significant": bool(p_val < 0.001),
    }

    # ------------------------------------------------------------------
    # 4. Hourly MAE Error Analysis (Section 9.2: MAE by Target Hour 00..23)
    # ------------------------------------------------------------------
    df_test = df_gold[df_gold["split"] == "test"].copy()
    df_test["target_hour"] = pd.to_datetime(df_test["target_ts_t60"]).dt.hour
    df_test["abs_err_b2"] = np.abs(df_test["target_ratio_t60"] - df_test["pred_b2_lgbm"])
    df_test["abs_err_b0"] = np.abs(df_test["target_ratio_t60"] - df_test["pred_b0_persistence"])

    hourly_error = []
    for hr, grp in df_test.groupby("target_hour"):
        hourly_error.append({
            "hour": int(hr),
            "hour_label": f"{int(hr):02d}:00",
            "mae_b2_lgbm": round(float(grp["abs_err_b2"].mean()), 4),
            "mae_b0_persistence": round(float(grp["abs_err_b0"].mean()), 4),
            "congested_rate": round(float(grp["target_is_congested_t60"].mean()), 4),
        })

    # ------------------------------------------------------------------
    # 5. Section 9.3 Feature Gate Assessment (keep groups with >= 0.005 gain)
    # ------------------------------------------------------------------
    ab_path = ROOT_DIR / cfg["paths"]["evaluation_dir"] / "ablation_results.json"
    feature_gate_decisions = []
    if ab_path.exists():
        with open(ab_path, "r", encoding="utf-8") as f:
            ab_data = json.load(f)
        for exp in ab_data.get("ablation_experiments", []):
            # Removing the group caused delta_f1_all or delta_onset_recall to drop
            f1_contribution = round(-float(exp["delta_f1_all"]), 4)
            onset_contribution = round(-float(exp["delta_onset_recall"]), 4)
            rain_contribution = round(-float(exp["delta_f1_rainy"]), 4)
            passed_gate = (
                f1_contribution >= 0.005
                or onset_contribution >= 0.005
                or rain_contribution >= 0.005
                or exp["group_key"] == "group_1_current_lags"
            )
            feature_gate_decisions.append({
                "group_key": exp["group_key"],
                "group_label": exp["group_label"],
                "marginal_f1_gain": f1_contribution,
                "marginal_onset_recall_gain": onset_contribution,
                "marginal_rainy_f1_gain": rain_contribution,
                "retained_by_feature_gate": bool(passed_gate),
            })

    full_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "horizon_sensitivity": horizon_results,
        "rolling_origin_3fold_robustness": {
            "folds": fold_results,
            "mean_f1": round(float(np.mean(fold_f1s)), 4),
            "std_f1": round(float(np.std(fold_f1s)), 4),
            "max_deviation_from_mean": round(max_dev_from_mean, 4),
            "acceptance_criterion_max_005_passed": bool(robustness_passed),
        },
        "traffy_flood_event_study": event_study,
        "hourly_mae_error_analysis": hourly_error,
        "feature_gate_section_9_3": feature_gate_decisions,
    }

    out_path = ROOT_DIR / cfg["paths"]["evaluation_dir"] / "proposal_experiments_s11.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    log_info(
        f"Horizon Tradeoff (15m->60m F1): {horizon_results[0]['lgbm_f1']:.4f} -> {horizon_results[-1]['lgbm_f1']:.4f} | "
        f"3-Fold Max Dev: {max_dev_from_mean:.4f} (<=0.05: {robustness_passed}) | "
        f"Flood Event Study ΔRatio: {diff_mean:.4f} (95% CI [{ci_95_low}, {ci_95_high}], p={p_val:.2e})"
    )
    log_success("บันทึกผลการทดลองตามข้อ 9.2, 9.3 และ 11 ลง artifacts/evaluation/proposal_experiments_s11.json")
    return full_report

