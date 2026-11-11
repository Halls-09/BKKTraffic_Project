"""
Unit Tests for MLOps Model Registry & Drift Monitoring (Sarit - 67090500441).
Verifies:
1. Champion model registration, SHA256 lineage tracking, and promotion gates
2. PSI calculation accuracy (identical distributions -> PSI = 0.0; shifted -> PSI > 0.25)
"""
from __future__ import annotations

import json
import numpy as np

from src.ml_engineering.monitoring_drift import classify_psi, compute_psi
from src.utils.storage import ROOT_DIR, load_config


def test_psi_drift_mathematics() -> None:
    rng = np.random.default_rng(42)
    ref = rng.normal(0.65, 0.12, size=2000)
    same = ref.copy()
    shifted = rng.normal(0.35, 0.15, size=2000)

    psi_zero = compute_psi(ref, same, bins=10)
    psi_high = compute_psi(ref, shifted, bins=10)

    assert psi_zero == 0.0, f"Expected PSI=0.0 for identical arrays, got {psi_zero}"
    assert classify_psi(psi_zero) == "STABLE"
    assert psi_high >= 0.25, f"Expected PSI >= 0.25 for severely shifted array, got {psi_high}"
    assert classify_psi(psi_high) == "CRITICAL_DRIFT"


def test_model_registry_champion_gates() -> None:
    cfg = load_config()
    champ_path = ROOT_DIR / cfg["paths"]["registry_dir"] / "champion_metadata.json"
    assert champ_path.exists(), "Champion metadata not found in registry!"

    with open(champ_path, "r", encoding="utf-8") as f:
        champ = json.load(f)

    assert champ["stage"] == "Champion"
    assert champ["promotion_decision"]["promoted_to_champion"] is True
    assert len(champ["model_sha256"]) == 64
    assert champ["metrics"]["test"]["onset_recall"] >= cfg["registry"]["promotion_rules"]["min_onset_recall"]
