"""
FastAPI Serving & MLOps Control Plane API (Real iTIC Champion Model + Longdo & Google Maps Config).
Uses mtime-keyed model caching & cached snapshot telemetry for <10ms response latency.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.data_science.forecast import _prepare_model_matrix
from src.ml_engineering.app_dashboard import (
    build_dashboard_payload,
    generate_standalone_dashboard_html,
    get_map_api_keys,
    save_map_api_keys,
)
from src.ml_engineering.registry import ModelRegistry
from src.utils.logger import log_info
from src.utils.storage import ROOT_DIR, load_config, load_table


class PredictRequest(BaseModel):
    link_id: str = Field(..., description="Road link ID (from_loc_to_loc)")
    precip_3h_mm_t0: float | None = Field(default=None, ge=0.0, le=200.0)
    downstream_red_rate_t0: float | None = Field(default=None, ge=0.0, le=1.0)
    red_rate_t0: float | None = Field(default=None, ge=0.0, le=1.0)


class RollbackRequest(BaseModel):
    target_version: str = Field(..., description="Registered version to roll back to, e.g. v1.0.0")
    reason: str = Field(default="Manual rollback via MLOps API")


class MapKeysRequest(BaseModel):
    longdo_map_api_key: str | None = Field(default=None, description="Longdo Map JS API key")
    google_maps_api_key: str | None = Field(default=None, description="Google Maps JS API key")


@lru_cache(maxsize=1)
def _load_latest_link_features() -> pd.DataFrame:
    cfg = load_config()
    df = load_table(cfg["paths"]["gold_data"])
    df_test = df[df["split"] == "test"]
    peak = df_test[(df_test["target_hour"] == 18) & (df_test["is_weekend"] == 0)]
    snap_ts = peak["target_ts"].max() if len(peak) else df_test["target_ts"].max()
    return df_test[df_test["target_ts"] == snap_ts].drop_duplicates("link_id").set_index("link_id")


@lru_cache(maxsize=2)
def _load_cached_model_bundle(model_path_str: str, mtime_ns: int) -> dict[str, Any]:
    _ = mtime_ns
    bundle = joblib.load(model_path_str)
    if hasattr(bundle.get("model"), "set_params"):
        try:
            bundle["model"].set_params(n_jobs=1)
        except Exception:
            pass
    return bundle


def _get_active_model_bundle() -> dict[str, Any]:
    cfg = load_config()
    reg = ModelRegistry()
    target = reg.champion_model_path
    if not target.exists():
        target = ROOT_DIR / cfg["paths"]["models_dir"] / "b2_lgbm_model.joblib"
    return _load_cached_model_bundle(str(target), target.stat().st_mtime_ns)


def handle_health() -> dict[str, Any]:
    cfg = load_config()
    reg = ModelRegistry()
    champ_meta = {}
    if reg.champion_meta_path.exists():
        champ_meta = json.loads(reg.champion_meta_path.read_text(encoding="utf-8"))
    keys = get_map_api_keys()
    return {
        "status": "ok",
        "project": cfg["project"]["name"],
        "active_champion": champ_meta.get("version"),
        "model_backend": champ_meta.get("model_backend"),
        "optimal_threshold": champ_meta.get("optimal_threshold"),
        "longdo_map_configured": bool(keys["longdo_map_api_key"]),
        "google_maps_configured": bool(keys["google_maps_api_key"]),
    }


def handle_list_links(district: str | None = None, risk: str | None = None, limit: int = 100) -> dict[str, Any]:
    payload = build_dashboard_payload(use_cached_snapshot=True)
    links = payload["map_links"]
    if district:
        links = [l for l in links if l["district"] == district or l["district_en"].lower() == district.lower()]
    if risk:
        links = [l for l in links if l["risk"].upper() == risk.upper()]
    return {"count": len(links[:limit]), "total_matching": len(links), "links": links[:limit]}


def handle_predict(req: PredictRequest) -> dict[str, Any]:
    cfg = load_config()
    latest_df = _load_latest_link_features()
    if req.link_id not in latest_df.index:
        raise HTTPException(status_code=404, detail=f"link_id '{req.link_id}' not found in active 500 links")

    row_df = latest_df.loc[[req.link_id]].reset_index().copy()
    if req.precip_3h_mm_t0 is not None:
        row_df["precip_3h_mm_t0"] = float(req.precip_3h_mm_t0)
        row_df["precip_mm_t0"] = round(float(req.precip_3h_mm_t0) / 3.0, 2)
        row_df["rain_x_flood_interaction"] = float(req.precip_3h_mm_t0) * np.log1p(
            max(0.0, float(row_df["train_flood_per_sq_km"].iloc[0]))
        )
    if req.downstream_red_rate_t0 is not None:
        row_df["downstream_red_rate_t0"] = float(req.downstream_red_rate_t0)
        row_df["spatial_gradient_t0"] = float(req.downstream_red_rate_t0) - float(row_df["red_rate_t0"].iloc[0])
    if req.red_rate_t0 is not None:
        row_df["red_rate_t0"] = float(req.red_rate_t0)

    bundle = _get_active_model_bundle()
    X = _prepare_model_matrix(row_df, bundle["feature_cols"], bundle.get("cat_dtypes"))
    pred_rate = float(np.clip(bundle["model"].predict(X)[0], 0.0, 1.0))
    th = float(bundle["optimal_threshold"])
    is_cong = int(pred_rate >= th)
    onset_max = float(cfg["dataset"]["onset_freeflow_max"])
    is_onset = int(float(row_df["red_rate_t0"].iloc[0]) < onset_max and is_cong == 1)
    risk = "SEVERE" if pred_rate >= 0.70 else ("HIGH" if pred_rate >= th else ("MODERATE" if pred_rate >= 0.25 else "LOW"))

    return {
        "link_id": req.link_id,
        "district": str(row_df["district"].iloc[0]),
        "pred_red_rate_t1h": round(pred_rate, 4),
        "threshold_used": th,
        "pred_is_congested_t1h": is_cong,
        "pred_is_onset_alert": is_onset,
        "congestion_risk_level": risk,
        "model_backend": bundle.get("backend"),
    }


def create_api_app() -> FastAPI:
    cfg = load_config()
    app = FastAPI(
        title="Bangkok Traffic MLOps & Dual Map Dashboard API",
        version=cfg["project"]["version"],
    )

    @app.get("/", response_class=HTMLResponse, tags=["Dashboard UI"])
    def serve_dashboard_ui() -> str:
        html_path = ROOT_DIR / cfg["paths"]["dashboard_html"]
        if not html_path.exists():
            generate_standalone_dashboard_html()
        return html_path.read_text(encoding="utf-8")

    @app.get("/api/payload", tags=["Dashboard UI"])
    def get_dashboard_payload() -> dict[str, Any]:
        return build_dashboard_payload(use_cached_snapshot=True)

    @app.get("/api/v1/health", tags=["System"])
    def health_check() -> dict[str, Any]:
        return handle_health()

    @app.get("/api/v1/weather", tags=["Weather"])
    def get_weather_summary() -> dict[str, Any]:
        payload = build_dashboard_payload(use_cached_snapshot=True)
        return payload.get("weather_summary", {})

    @app.post("/api/v1/config/map-keys", tags=["Map APIs"])
    def update_map_keys(req: MapKeysRequest) -> dict[str, Any]:
        keys = save_map_api_keys(
            longdo_map_api_key=req.longdo_map_api_key,
            google_maps_api_key=req.google_maps_api_key,
        )
        generate_standalone_dashboard_html()
        return {
            "status": "saved",
            "longdo_map_configured": bool(keys["longdo_map_api_key"]),
            "google_maps_configured": bool(keys["google_maps_api_key"]),
        }

    @app.get("/api/v1/links", tags=["Traffic Telemetry"])
    def list_links(
        district: str | None = Query(default=None),
        risk: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        return handle_list_links(district=district, risk=risk, limit=limit)

    @app.post("/api/v1/predict", tags=["1-Hour-Ahead Inference"])
    def predict_one_hour_ahead(req: PredictRequest) -> dict[str, Any]:
        return handle_predict(req)

    @app.get("/api/v1/mlops/registry", tags=["MLOps Control Plane"])
    def get_mlops_registry() -> dict[str, Any]:
        reg = ModelRegistry()
        return {
            "champion": json.loads(reg.champion_meta_path.read_text(encoding="utf-8")) if reg.champion_meta_path.exists() else {},
            "index": reg._load_index(),
            "tournament": json.loads(reg.tournament_path.read_text(encoding="utf-8")) if reg.tournament_path.exists() else {},
            "rollback_audit_log": json.loads(reg.rollback_log_path.read_text(encoding="utf-8")) if reg.rollback_log_path.exists() else {},
        }

    @app.get("/api/v1/mlops/monitoring", tags=["MLOps Control Plane"])
    def get_mlops_monitoring() -> dict[str, Any]:
        mon_path = ROOT_DIR / cfg["paths"]["monitoring_dir"] / "drift_monitoring_report.json"
        if not mon_path.exists():
            raise HTTPException(status_code=404, detail="Drift monitoring report not generated yet")
        return json.loads(mon_path.read_text(encoding="utf-8"))

    @app.get("/api/v1/mlops/lineage", tags=["MLOps Control Plane"])
    def get_mlops_lineage() -> dict[str, Any]:
        lin_dir = ROOT_DIR / cfg["paths"]["lineage_dir"]
        latest = json.loads((lin_dir / "latest_run.json").read_text(encoding="utf-8")) if (lin_dir / "latest_run.json").exists() else {}
        history = json.loads((lin_dir / "pipeline_run_history.json").read_text(encoding="utf-8")) if (lin_dir / "pipeline_run_history.json").exists() else []
        return {"latest_run": latest, "history": history}

    @app.post("/api/v1/mlops/rollback", tags=["MLOps Control Plane"])
    def trigger_rollback(req: RollbackRequest) -> dict[str, Any]:
        reg = ModelRegistry()
        try:
            event = reg.rollback_to_version(req.target_version, reason=req.reason)
            generate_standalone_dashboard_html()
            return event
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def serve_dashboard(host: str = "127.0.0.1", port: int = 8050) -> None:
    app = create_api_app()
    log_info(f"เปิด Dashboard (Longdo + Google Maps) + FastAPI ที่ http://{host}:{port} (API Docs: http://{host}:{port}/docs)")
    uvicorn.run(app, host=host, port=port)
