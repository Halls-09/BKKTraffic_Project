"""Automated verification tests for the Bangkok Traffic MLOps Platform (Real iTIC Data)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from src.ml_engineering.api_server import (
    PredictRequest,
    create_api_app,
    handle_health,
    handle_list_links,
    handle_predict,
)
from src.ml_engineering.monitoring_drift import classify_psi, compute_psi
from src.ml_engineering.registry import ModelRegistry
from src.utils.storage import ROOT_DIR, load_config, load_table


def test_zero_temporal_leakage_real_data() -> None:
    cfg = load_config()
    df = load_table(cfg["paths"]["gold_data"])
    train_max = df[df["split"] == "train"]["target_ts"].max()
    val_min = df[df["split"] == "val"]["target_ts"].min()
    val_max = df[df["split"] == "val"]["target_ts"].max()
    test_min = df[df["split"] == "test"]["target_ts"].min()

    assert train_max < val_min, f"Train/Val overlap: {train_max} >= {val_min}"
    assert val_max < test_min, f"Val/Test overlap: {val_max} >= {test_min}"
    assert ((df["target_ts"] - df["ts_hour"]).dt.total_seconds() == 3600).all()

    # Verify Data Contract 4.2: 100% Location Join Coverage, no self-loops, no fake (13.75, 100.5) pins
    assert (df["from_loc"] != df["to_loc"]).all(), "Self-loop links found!"
    for col in ("lat", "lon", "lat_from", "lon_from", "lat_to", "lon_to", "segment_label"):
        assert col in df.columns and not df[col].isna().any(), f"Missing geometry column: {col}"
    links = df.drop_duplicates("link_id")
    fake_pins = ((links["lat"] == 13.75) & (links["lon"] == 100.5)).sum()
    assert fake_pins == 0, f"Found {fake_pins} fake default (13.75, 100.5) coordinates!"
    dup_coords = links.duplicated(["lat", "lon"]).sum()
    assert dup_coords == 0, f"Found {dup_coords} overlapping duplicate marker coordinates!"


def test_psi_drift_math() -> None:
    rng = np.random.default_rng(42)
    ref = rng.normal(0.2, 0.1, size=5000)
    same = rng.normal(0.2, 0.1, size=5000)
    shifted = rng.normal(0.6, 0.1, size=5000)

    psi_same = compute_psi(ref, same)
    psi_shifted = compute_psi(ref, shifted)

    assert psi_same < 0.05 and classify_psi(psi_same) == "STABLE"
    assert psi_shifted > 0.25 and classify_psi(psi_shifted) == "CRITICAL"


def test_registry_gates_and_rollback() -> None:
    cfg = load_config()
    cand = ROOT_DIR / cfg["paths"]["models_dir"] / "b2_lgbm_model.joblib"
    eval_summary = json.loads((ROOT_DIR / cfg["paths"]["evaluation_dir"] / "evaluation_summary.json").read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmp:
        reg = ModelRegistry(registry_dir=Path(tmp))
        m0 = reg.register_and_evaluate_challenger(cand, eval_summary)
        assert m0["promotion_decision"]["promoted_to_champion"] is True
        assert m0["version"] == "v1.0.0"

        m1 = reg.register_and_evaluate_challenger(cand, eval_summary)
        assert m1["version"] == "v1.0.1"

        rb = reg.rollback_to_version("v1.0.0", reason="Unit test rollback verification")
        assert rb["status"] == "SUCCESS"
        assert rb["target_version"] == "v1.0.0"
        assert reg._load_index()["active_champion"] == "v1.0.0"


def test_fastapi_endpoints() -> None:
    app = create_api_app()
    paths = {getattr(r, "path", "") for r in app.routes}
    for required in (
        "/",
        "/api/v1/health",
        "/api/v1/weather",
        "/api/v1/links",
        "/api/v1/predict",
        "/api/v1/config/map-keys",
        "/api/v1/mlops/registry",
        "/api/v1/mlops/monitoring",
        "/api/v1/mlops/lineage",
        "/api/v1/mlops/rollback",
    ):
        assert required in paths, f"Missing FastAPI route: {required}"

    health = handle_health()
    assert health["status"] == "ok" and health["active_champion"] is not None
    assert "longdo_map_configured" in health and "google_maps_configured" in health

    links_res = handle_list_links(limit=5)
    assert links_res["count"] == 5
    first_link = links_res["links"][0]["link_id"]

    pred = handle_predict(PredictRequest(link_id=first_link, precip_3h_mm_t0=25.0, downstream_red_rate_t0=0.8))
    assert 0.0 <= pred["pred_red_rate_t1h"] <= 1.0
    assert pred["congestion_risk_level"] in ("LOW", "MODERATE", "HIGH", "SEVERE")


def test_proposal_acceptance_gates_and_monitoring() -> None:
    cfg = load_config()
    eval_dir = ROOT_DIR / cfg["paths"]["evaluation_dir"]
    mon_dir = ROOT_DIR / cfg["paths"]["monitoring_dir"]

    ev = json.loads((eval_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    sh = json.loads((eval_dir / "shap_attributions.json").read_text(encoding="utf-8"))
    mn = json.loads((mon_dir / "drift_monitoring_report.json").read_text(encoding="utf-8"))

    # 1. Section 11 Page 12: Rolling-origin 3-fold robustness (F1 deviation <= +-0.05)
    rob = ev.get("robustness_3fold", {})
    assert rob.get("passed") is True, f"3-fold robustness failed: {rob}"
    assert float(rob.get("max_abs_deviation", 1.0)) <= 0.05
    assert len(rob.get("folds", [])) == 3

    # 2. Section 13 Page 13: Daytime active probe (06:00-23:00) evaluation present
    b2_test = ev["models"]["B2_LightGBM"]["test"]
    assert "f1_daytime_06_23" in b2_test and b2_test["f1_daytime_06_23"] >= b2_test["f1_all"]

    # 3. Section 5 Page 6: All 5 Hypotheses (H1-H5) verified and supported
    hyps = sh.get("hypotheses_verification", [])
    assert [h["id"] for h in hyps] == ["H1", "H2", "H3", "H4", "H5"]
    assert all(h["supported"] is True for h in hyps)

    # 4. Section 7 & 10: Layer 1 Data Quality (Join Coverage 100%, Freshness, Min Sample Check)
    dq = mn.get("data_quality_layer", {})
    assert dq.get("location_table_join_coverage_pct") == 100.0
    assert dq.get("feed_freshness_status") == "FRESH"
    assert dq.get("minimum_sample_check_passed") is True

    # 5. Data Contract 4.2 Adapter, Airflow DAG Specs & Online Incremental ML
    from dags.bkk_traffic_mlops_dag import AIRFLOW_DAG_SPECS
    assert len(AIRFLOW_DAG_SPECS) == 3

    dc_path = mon_dir / "data_contract_audit.json"
    om_path = eval_dir / "online_ml_report.json"
    assert dc_path.exists() and om_path.exists()
    dc = json.loads(dc_path.read_text(encoding="utf-8"))
    om = json.loads(om_path.read_text(encoding="utf-8"))
    assert dc.get("all_gates_passed") is True and len(dc.get("gates", [])) == 5
    assert om.get("final_trees_after_stream", 0) > om.get("initial_trees", 0)
    assert len(om.get("stream_windows", [])) == 6
