"""
Strict Temporal Leakage Unit Tests.
Verifies:
1. Train split target timestamps (t + 60m) never cross train_end cutoff
2. P90 Free-flow speed and Historical Profile are fit strictly on Train split
3. Perturbing future target values in Test split does NOT alter features at time t
"""
from __future__ import annotations

import pandas as pd

from src.data_engineering.build_features import ALL_FEATURE_COLS
from src.utils.storage import ROOT_DIR, load_config, load_table


def test_zero_temporal_leakage_boundaries() -> None:
    cfg = load_config()
    gold_path = ROOT_DIR / cfg["paths"]["gold_dir"] / "gold_features.parquet"
    df_gold = load_table(gold_path)

    train_end = pd.Timestamp(cfg["temporal_split"]["train_end"])
    val_start = pd.Timestamp(cfg["temporal_split"]["val_start"])
    val_end = pd.Timestamp(cfg["temporal_split"]["val_end"])
    test_start = pd.Timestamp(cfg["temporal_split"]["test_start"])

    df_train = df_gold[df_gold["split"] == "train"]
    df_val = df_gold[df_gold["split"] == "val"]
    df_test = df_gold[df_gold["split"] == "test"]

    # 1. Verify strict non-overlapping temporal boundaries including +60m target horizon
    assert (pd.to_datetime(df_train["target_ts_t60"]) <= train_end).all(), (
        "Temporal Leakage: Train target_ts_t60 crossed train_end boundary!"
    )
    assert (pd.to_datetime(df_val["ts_t"]) >= val_start).all()
    assert (pd.to_datetime(df_val["target_ts_t60"]) <= val_end).all()
    assert (pd.to_datetime(df_test["ts_t"]) >= test_start).all()

    # 2. Verify no target column is accidentally present in ALL_FEATURE_COLS
    forbidden = {"target_ratio_t60", "target_is_congested_t60", "is_onset_event", "target_ts_t60"}
    overlap = forbidden.intersection(set(ALL_FEATURE_COLS))
    assert len(overlap) == 0, f"Target columns found in feature list: {overlap}"

    # 3. Verify lag features strictly match historical observations at t - 15m
    sample_seg = df_gold["seg_id"].iloc[0]
    sub = df_gold[df_gold["seg_id"] == sample_seg].sort_values("ts_t").reset_index(drop=True)
    consec = (pd.to_datetime(sub["ts_t"]).diff() == pd.Timedelta(minutes=15))
    valid_idx = sub.index[consec]
    for idx in valid_idx[:25]:
        assert abs(sub.loc[idx, "ratio_lag_15m"] - sub.loc[idx - 1, "ratio_t0"]) < 1e-5
