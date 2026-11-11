"""
Unit Tests for Quality Gate 4.2 & Data Contract Enforcement.
Verifies that out-of-bounds speeds (>150 km/h), zero probe counts, and extreme jumps
are properly caught, flagged, and quarantined.
"""
from __future__ import annotations

import pandas as pd

from src.data_engineering.quality_gate import validate_and_partition_traffic
from src.utils.storage import ROOT_DIR, load_config, load_table


def test_quality_gate_quarantines_outliers() -> None:
    cfg = load_config()
    bronze_dir = ROOT_DIR / cfg["paths"]["bronze_dir"]
    df_segs = load_table(bronze_dir / "bronze_itic_segments.parquet")
    df_traffic = load_table(bronze_dir / "bronze_itic_traffic_15m.parquet")

    df_valid, df_quarantine, report = validate_and_partition_traffic(df_traffic, df_segs)

    # 1. Verify no row in valid_df exceeds 150 km/h or has < 1 probe
    assert (df_valid["speed_median_kmh"] <= 150.0).all(), "Speed > 150 km/h leaked into Silver!"
    assert (df_valid["speed_median_kmh"] >= 0.0).all(), "Negative speed leaked into Silver!"
    assert (df_valid["probe_count"] >= 1).all(), "Zero probe count leaked into Silver!"

    # 2. Verify injected outliers were caught in quarantine
    assert len(df_quarantine) >= 20, "Quarantine table did not catch injected sensor outliers!"
    assert "quarantine_reason" in df_quarantine.columns

    # 3. Verify completeness passes contract (>= 80% daily, >= 90% daytime)
    assert report["overall_completeness"] >= 0.80
    assert report["daytime_completeness"] >= 0.90
