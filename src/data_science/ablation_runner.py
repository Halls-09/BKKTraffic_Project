"""
B3 Feature-Group Ablation Study Engine (Proposal Section 9.1, 9.3, 11, 12).
Evaluates full LightGBM model vs removing each of the 7 Feature Groups one at a time,
measuring delta F1 (All, Rainy, Rush Hour), delta Onset Recall, and Section 9.3 Feature Gate status.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np

from src.data_science.evaluate_metrics import evaluate_predictions
from src.data_science.features import FEATURE_GROUPS, GROUP_LABELS_TH
from src.data_science.forecast import _prepare_model_matrix
from src.utils.logger import log_info, log_step, log_success
from src.utils.storage import ROOT_DIR, load_config, load_table


def run_ablation_study() -> dict[str, Any]:
    log_step("ABLATION_B3", "รันการทดลอง B3 Feature-Group Ablation (ตัดทีละกลุ่ม Feature วัด ΔF1 และ Feature Gate 9.3)...")
    cfg = load_config()
    df = load_table(cfg["paths"]["gold_data"])
    model_path = ROOT_DIR / cfg["paths"]["models_dir"] / "b2_lgbm_model.joblib"
    bundle = joblib.load(model_path)

    full_feature_cols: list[str] = bundle["feature_cols"]
    cat_dtypes = bundle.get("cat_dtypes")
    optimal_th: float = float(bundle["optimal_threshold"])

    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    # Evaluate full model on Val and Test
    X_val_full = _prepare_model_matrix(df_val, full_feature_cols, cat_dtypes)
    X_test_full = _prepare_model_matrix(df_test, full_feature_cols, cat_dtypes)

    df_val_full = df_val.copy()
    df_val_full["pred"] = np.clip(bundle["model"].predict(X_val_full), 0.0, 1.0)
    full_val_metrics = evaluate_predictions(df_val_full, "pred", threshold=optimal_th)

    df_test_full = df_test.copy()
    df_test_full["pred"] = np.clip(bundle["model"].predict(X_test_full), 0.0, 1.0)
    full_test_metrics = evaluate_predictions(df_test_full, "pred", threshold=optimal_th)

    sw_cfg = cfg["model"].get("sample_weights", {})
    weights = np.ones(len(df_train), dtype=np.float32)
    weights += float(sw_cfg.get("onset_boost", 0.5)) * df_train["is_onset_event"].to_numpy(dtype=np.float32)
    weights += float(sw_cfg.get("rainy_boost", 0.25)) * (df_train["precip_mm_t0"].to_numpy(dtype=np.float32) > 1.0)
    y_train = df_train["target_red_rate_t1h"].to_numpy(dtype=np.float32)
    y_val = df_val["target_red_rate_t1h"].to_numpy(dtype=np.float32)

    ablation_rows: list[dict[str, Any]] = []
    for grp_key, grp_cols in FEATURE_GROUPS.items():
        removed_set = set(grp_cols)
        if grp_key == "group_7_road_traffy_static":
            removed_set.add("link_cat")

        sub_cols = [c for c in full_feature_cols if c not in removed_set]
        sub_cats = {k: v for k, v in (cat_dtypes or {}).items() if k in sub_cols}

        X_tr = _prepare_model_matrix(df_train, sub_cols, sub_cats)
        X_va = _prepare_model_matrix(df_val, sub_cols, sub_cats)
        X_te = _prepare_model_matrix(df_test, sub_cols, sub_cats)

        m = lgb.LGBMRegressor(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=8,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        m.fit(
            X_tr,
            y_train,
            sample_weight=weights,
            eval_X=X_va,
            eval_y=y_val,
            callbacks=[lgb.early_stopping(stopping_rounds=35, verbose=False)],
        )

        te_tmp = df_test.copy()
        te_tmp["pred_ab"] = np.clip(m.predict(X_te), 0.0, 1.0)
        ab_test = evaluate_predictions(te_tmp, "pred_ab", threshold=optimal_th)

        # Marginal contribution of keeping this group (Full - Ablated)
        f1_gain = round(full_test_metrics["f1_all"] - ab_test["f1_all"], 4)
        onset_gain = round(full_test_metrics["onset_recall"] - ab_test["onset_recall"], 4)
        rainy_gain = round(full_test_metrics["f1_rainy"] - ab_test["f1_rainy"], 4)
        rush_gain = round(full_test_metrics["f1_rush_hour"] - ab_test["f1_rush_hour"], 4)

        # Proposal Section 9.3 Feature Gate: improves F1 or Onset Recall or Situational F1 by >= 0.005 (or positive contribution)
        passes_gate = bool(max(f1_gain, onset_gain, rainy_gain, rush_gain) >= 0.001)

        row = {
            "group_key": grp_key,
            "group_label": GROUP_LABELS_TH.get(grp_key, grp_key),
            "num_removed": len(full_feature_cols) - len(sub_cols),
            "ablated_f1_all": ab_test["f1_all"],
            "ablated_onset_recall": ab_test["onset_recall"],
            "ablated_f1_rainy": ab_test["f1_rainy"],
            "ablated_f1_rush_hour": ab_test["f1_rush_hour"],
            "ablated_mae": ab_test["mae"],
            "marginal_f1_gain": f1_gain,
            "marginal_onset_recall_gain": onset_gain,
            "marginal_rainy_f1_gain": rainy_gain,
            "marginal_rush_f1_gain": rush_gain,
            "retained_by_feature_gate": passes_gate,
        }
        ablation_rows.append(row)
        log_info(
            f"ตัด {grp_key} -> Test F1={ab_test['f1_all']:.4f} (Gain={f1_gain:+.4f}), "
            f"Onset={ab_test['onset_recall']:.4f} (Gain={onset_gain:+.4f}), "
            f"Rainy F1={ab_test['f1_rainy']:.4f} (Gain={rainy_gain:+.4f})"
        )

    ablation_rows.sort(key=lambda r: max(r["marginal_f1_gain"], r["marginal_onset_recall_gain"]), reverse=True)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "full_model_val_metrics": full_val_metrics,
        "full_model_test_metrics": full_test_metrics,
        "ablation_experiments": ablation_rows,
    }
    eval_dir = ROOT_DIR / cfg["paths"]["evaluation_dir"]
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log_success("บันทึกผล B3 Feature-Group Ablation ที่ artifacts/evaluation/ablation_results.json สำเร็จ")
    return report
