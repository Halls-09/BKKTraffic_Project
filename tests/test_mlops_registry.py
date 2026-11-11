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


def test_env_and_api_endpoints() -> None:
    from src.ml_engineering.api_server import PredictSegmentRequest, create_api_app
    from src.utils.env_config import get_api_keys_status, mask_secret

    # 1. Verify secret masking never exposes raw key
    raw_secret = "sk-live-9876543210abcdef"
    masked = mask_secret(raw_secret)
    assert raw_secret not in masked
    assert masked.startswith("sk-l") and masked.endswith("ef")

    # 2. Verify .env status structure
    env_stat = get_api_keys_status()
    assert env_stat["env_file_exists"] is True
    assert env_stat["total_keys_tracked"] >= 8

    # 3. Verify FastAPI route handlers directly (zero network socket required)
    app = create_api_app()
    routes = {r.path: r.endpoint for r in app.routes if hasattr(r, "path") and hasattr(r, "endpoint")}
    assert "/api/v1/health" in routes
    assert "/api/v1/config/keys-status" in routes
    assert "/api/v1/predict" in routes

    health_res = routes["/api/v1/health"]()
    assert health_res["status"] == "ONLINE", f"Unexpected status: {health_res}"
    assert "v1.0" in str(health_res["champion_version"]), f"Unexpected champion_version: {health_res}"

    # Normal prediction (sensor_age_minutes <= 15)
    normal_req = PredictSegmentRequest(
        seg_id="SEG_SUK_01_IN",
        ratio_t0=0.56,
        ratio_lag_15m=0.68,
        downstream_min_ratio_t0=0.20,
        rain_mm_t0=30.0,
        rain_cum_3h=50.0,
        traffy_flood_buffer500m=28,
        active_incident=1,
        hist_slot_median_ratio=0.45,
        free_flow_speed_kmh=48.0,
        sensor_age_minutes=2.0,
    )
    pred_normal = routes["/api/v1/predict"](normal_req)
    assert pred_normal["circuit_breaker_triggered"] is False, f"Unexpected normal pred: {pred_normal}"
    assert pred_normal["pred_congested"] == 1, f"Expected congested=1, got {pred_normal}"
    assert pred_normal["pred_onset_alert"] == 1, f"Expected onset=1, got {pred_normal}"

    # Stale sensor (> 15 min) triggers Layer-1 Circuit-Breaker fallback to B1
    stale_req = PredictSegmentRequest(
        seg_id="SEG_SUK_01_IN",
        sensor_age_minutes=22.0,
        hist_slot_median_ratio=0.58,
    )
    pred_fallback = routes["/api/v1/predict"](stale_req)
    assert pred_fallback["circuit_breaker_triggered"] is True, f"Expected circuit breaker, got {pred_fallback}"
    assert pred_fallback["model_used"] == "B1_Historical_Slot_Fallback", f"Got {pred_fallback}"
    assert pred_fallback["pred_ratio_t60"] == 0.58, f"Got {pred_fallback}"
