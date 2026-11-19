"""Automated pytest suite for Thanakorn's Data Engineering Deliverables (Handoff 1).

Tests:
1. Data Contract & Data Dictionary completeness (Proposal Section 12).
2. Manifest SHA-256 checksum integrity (Proposal Section 0 & 7).
3. Quality Gates (Schema, Completeness, Uniqueness, Plausibility, Join Coverage - Section 4.2).
4. Governance (No PII / comment / photo in processed data - Section 4.1).
5. Leakage Controls (Train-only static features, strictly ordered temporal split, available_time - Section 6.2).
"""
from __future__ import annotations

import yaml
import pandas as pd

from src.common import ROOT, path, verify_manifest
from src.quality.gates import verify_all_processed_tables


def test_data_contract_and_dictionary_exist():
    contract_file = ROOT / "contracts" / "data_contract.yaml"
    dict_file = ROOT / "docs" / "data_dictionary.md"
    assert contract_file.exists(), "Missing contracts/data_contract.yaml"
    assert dict_file.exists(), "Missing docs/data_dictionary.md"

    contract = yaml.safe_load(contract_file.read_text(encoding="utf-8"))
    for ds_name in [
        "weather_hourly",
        "calendar_daily",
        "dim_district",
        "traffy_district_hourly",
        "traffy_district_daily",
        "traffy_district_static_train",
        "district_hourly_panel",
    ]:
        assert ds_name in contract["datasets"], f"Missing {ds_name} in data_contract.yaml"
        ds = contract["datasets"][ds_name]
        for req_key in ["owner", "license", "grain", "event_time_col", "available_time_col"]:
            assert req_key in ds, f"Missing {req_key} for {ds_name}"


def test_raw_manifest_checksums():
    problems = verify_manifest()
    assert problems == [], f"Raw manifest checksum mismatch: {problems}"


def test_all_processed_csv_and_parquet_exist():
    proc = path("processed")
    expected = [
        "weather_hourly",
        "calendar_daily",
        "dim_district",
        "traffy_district_hourly",
        "traffy_district_daily",
        "traffy_district_static_train",
        "district_hourly_panel",
    ]
    for name in expected:
        assert (proc / f"{name}.parquet").exists(), f"Missing {name}.parquet"
        assert (proc / "csv" / f"{name}.csv").exists(), f"Missing csv/{name}.csv"


def test_quality_and_leakage_gates_pass():
    summary = verify_all_processed_tables(run_id="pytest")
    assert summary["pass"] is True


def test_proposal_exact_audit_numbers():
    proc = path("processed")
    panel = pd.read_parquet(proc / "district_hourly_panel.parquet")
    assert len(panel) == 50 * 2928  # 146,400 rows
    assert int(panel["n_total"].sum()) == 151248
    assert int(panel["n_traffic"].sum()) == 6008
    assert int(panel["n_flood"].sum()) == 15024
    assert int(panel["n_road"].sum()) == 36191
