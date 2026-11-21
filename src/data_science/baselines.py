"""
Baseline Models (B0: Persistence & B1: Historical Profile Average) on Real iTIC Data.
Demonstrates why Persistence fails 100% on Congestion Onset (Onset Recall = 0.0).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.data_science.evaluate_metrics import evaluate_predictions
from src.utils.logger import log_info, log_step


def evaluate_baselines(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    log_step("BASELINES", "ประเมินโมเดลฐาน B0 (Persistence) และ B1 (Historical Profile Average)...")
    work = df.copy()

    # B0: Persistence (predicts red_rate at t+1h equals current red_rate_t0)
    work["pred_b0_persistence"] = work["red_rate_t0"]

    # B1: Historical Profile Average (Train-only historical mean for link x target_hour x is_weekend)
    work["pred_b1_historical"] = work["hist_avg_red_target_slot"]

    results: dict[str, dict[str, Any]] = {"B0_Persistence": {}, "B1_Historical_Average": {}}
    for split_name in ("train", "val", "test"):
        sub = work[work["split"] == split_name]
        if len(sub) == 0:
            continue
        results["B0_Persistence"][split_name] = evaluate_predictions(sub, "pred_b0_persistence", threshold=0.50)
        results["B1_Historical_Average"][split_name] = evaluate_predictions(sub, "pred_b1_historical", threshold=0.50)

    b0_test = results["B0_Persistence"].get("test", {})
    b1_test = results["B1_Historical_Average"].get("test", {})
    log_info(
        f"[Test] B0 Persistence -> F1={b0_test.get('f1_all', 0):.4f}, "
        f"Onset Recall={b0_test.get('onset_recall', 0):.4f}, MAE={b0_test.get('mae', 0):.4f}"
    )
    log_info(
        f"[Test] B1 Historical  -> F1={b1_test.get('f1_all', 0):.4f}, "
        f"Onset Recall={b1_test.get('onset_recall', 0):.4f}, MAE={b1_test.get('mae', 0):.4f}"
    )
    return results
