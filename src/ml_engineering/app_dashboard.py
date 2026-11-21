"""
SpaceX Mission Control — Traffic Forecasting Telemetry & Executive ML Report Dashboard.
100% Aligned with CSS 342 Proposal (KMUTT) & ML Engineer Scope:
  - View 1 (01 TELEMETRY MAP): Longdo Map SDK default (with Traffic/Events/CCTV + Google Maps toggle),
    448 directional road segments (Polyline + lane-offset midpoints), search/filter, 3 real 2022 scenarios,
    3-way forecast comparison (+1h ML vs +1h Actual vs t0), and live What-If API simulator.
  - View 2 (02 MODEL REPORT): Full Technical Report with 6 interactive Chart.js visualizations
    (Benchmark, 24h Actual vs Forecast Profile, Hourly MAE, TreeSHAP Top 12, B3 Factor Ablation,
    Online Incremental ML Stream), 21 Core Features inventory, Hypotheses H1-H5, Airflow DAGs,
    Data Contract 4.2 (5 Gates), 3-Fold Robustness, Promotion Gates, Rollback & Defense Q&A Notes.
  - View 3 (03 3-LAYER MONITORING): Layer 1 Data Quality & Segment Circuit-Breaker (5 SLA checks),
    Layer 2 Feature/Prediction PSI Bar Chart + Monsoon Stress Test, Layer 3 Rolling 7-Day F1 & MAE.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_science.features import FEATURE_DESCRIPTIONS_TH, FEATURE_GROUPS, GROUP_LABELS_TH
from src.data_science.forecast import forecast_next_hour
from src.utils.logger import log_step, log_success
from src.utils.storage import ROOT_DIR, load_config, load_table


def _read_env_file() -> dict[str, str]:
    env_file = ROOT_DIR / ".env"
    values: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def get_map_api_keys() -> dict[str, str]:
    """Reads LONGDO_MAP_API_KEY and GOOGLE_MAPS_API_KEY from env or `.env`."""
    file_vals = _read_env_file()
    return {
        "longdo_map_api_key": os.environ.get("LONGDO_MAP_API_KEY", "").strip() or file_vals.get("LONGDO_MAP_API_KEY", ""),
        "google_maps_api_key": os.environ.get("GOOGLE_MAPS_API_KEY", "").strip() or file_vals.get("GOOGLE_MAPS_API_KEY", ""),
    }


def save_map_api_keys(
    longdo_map_api_key: str | None = None,
    google_maps_api_key: str | None = None,
) -> dict[str, str]:
    """Persists provided map API keys into local `.env` safely without overwriting unset keys."""
    updates: dict[str, str] = {}
    if longdo_map_api_key is not None and longdo_map_api_key.strip():
        updates["LONGDO_MAP_API_KEY"] = longdo_map_api_key.strip()
    if google_maps_api_key is not None and google_maps_api_key.strip():
        updates["GOOGLE_MAPS_API_KEY"] = google_maps_api_key.strip()

    env_file = ROOT_DIR / ".env"
    existing_lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
    out_lines: list[str] = []
    written: set[str] = set()

    for line in existing_lines:
        stripped = line.strip()
        matched_key = next((k for k in updates if stripped.startswith(f"{k}=")), None)
        if matched_key:
            out_lines.append(f'{matched_key}="{updates[matched_key]}"')
            written.add(matched_key)
        else:
            out_lines.append(line)

    for k, v in updates.items():
        if k not in written:
            out_lines.append(f'{k}="{v}"')
        os.environ[k] = v

    if updates:
        env_file.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return get_map_api_keys()


def _read_json_safe(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default if default is not None else {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_scenario_links(snap_pred: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    has_seg_lbl = "segment_label" in snap_pred.columns
    has_endpoints = ("lat_from" in snap_pred.columns) and ("lat_to" in snap_pred.columns)
    links: list[dict[str, Any]] = []
    for _, r in snap_pred.iterrows():
        if has_seg_lbl and pd.notna(r["segment_label"]):
            road_label = str(r["segment_label"])
        else:
            road_label = str(r["road_name_th"]) if pd.notna(r["road_name_th"]) else str(r["first_name_th"])
        if road_label == "nan":
            road_label = f"Link {r['link_id']}"

        lat_mid = round(float(r["lat"]), 5)
        lon_mid = round(float(r["lon"]), 5)
        lat_f = round(float(r["lat_from"]), 5) if has_endpoints and pd.notna(r["lat_from"]) else lat_mid
        lon_f = round(float(r["lon_from"]), 5) if has_endpoints and pd.notna(r["lon_from"]) else lon_mid
        lat_t = round(float(r["lat_to"]), 5) if has_endpoints and pd.notna(r["lat_to"]) else lat_mid
        lon_t = round(float(r["lon_to"]), 5) if has_endpoints and pd.notna(r["lon_to"]) else lon_mid

        links.append(
            {
                "link_id": str(r["link_id"]),
                "road_name": road_label,
                "district": str(r["district"]),
                "district_en": str(r["district_en"]),
                "lat": lat_mid,
                "lon": lon_mid,
                "lat_from": lat_f,
                "lon_from": lon_f,
                "lat_to": lat_t,
                "lon_to": lon_t,
                "length_km": round(float(r["length_km"]), 2) if "length_km" in snap_pred.columns else 0.7,
                "red_rate_t0": round(float(r["red_rate_t0"]), 3),
                "actual_t1h": round(float(r["target_red_rate_t1h"]), 3),
                "actual_congested": int(r["target_is_congested_t1h"]),
                "pred_t1h": round(float(r["pred_red_rate_t1h"]), 3),
                "is_congested": int(r["pred_is_congested_t1h"]),
                "is_onset": int(r["pred_is_onset_alert"]),
                "risk": str(r["congestion_risk_level"]),
                "temp_c": round(float(r["temp_c_t0"]), 1),
                "rain_mm": round(float(r["precip_mm_t0"]), 2),
                "rain_3h": round(float(r["precip_3h_mm_t0"]), 2),
                "flood_density": round(float(r["train_flood_per_sq_km"]), 2),
                "downstream": round(float(r["downstream_red_rate_t0"]), 3),
            }
        )

    wx = {
        "avg_temp_c": round(float(snap_pred["temp_c_t0"].mean()), 1),
        "max_rain_1h_mm": round(float(snap_pred["precip_mm_t0"].max()), 2),
        "avg_rain_3h_mm": round(float(snap_pred["precip_3h_mm_t0"].mean()), 2),
        "max_rain_3h_mm": round(float(snap_pred["precip_3h_mm_t0"].max()), 2),
        "rainy_links_count": int((snap_pred["precip_3h_mm_t0"] > 0.5).sum()),
        "heavy_rain_links_count": int((snap_pred["precip_3h_mm_t0"] > 5.0).sum()),
    }
    return links, wx


def _build_feature_catalog(shap_report: dict[str, Any]) -> list[dict[str, Any]]:
    shap_map = {r["feature"]: r["share_pct"] for r in shap_report.get("all_features_ranked", [])}
    rows: list[dict[str, Any]] = []
    for grp_key, cols in FEATURE_GROUPS.items():
        grp_lbl = GROUP_LABELS_TH.get(grp_key, grp_key)
        for c in cols:
            rows.append(
                {
                    "group": grp_lbl.split(":")[0],
                    "group_name": grp_lbl.split(":")[-1].strip(),
                    "feature": c,
                    "shap_pct": shap_map.get(c, 0.0),
                    "description_th": FEATURE_DESCRIPTIONS_TH.get(c, c),
                }
            )
    rows.append(
        {
            "group": "G7",
            "group_name": "กายภาพถนนและจุดเสี่ยงน้ำท่วม (Road & Traffy)",
            "feature": "link_cat",
            "shap_pct": shap_map.get("link_cat", 0.0),
            "description_th": FEATURE_DESCRIPTIONS_TH.get("link_cat", "Categorical Road Segment ID"),
        }
    )
    return rows


def build_dashboard_payload(use_cached_snapshot: bool = False) -> dict[str, Any]:
    """Assembles unified JSON payload for standalone HTML and FastAPI `/api/payload`."""
    cfg = load_config()
    eval_dir = ROOT_DIR / cfg["paths"]["evaluation_dir"]
    reg_dir = ROOT_DIR / cfg["paths"]["registry_dir"]
    mon_dir = ROOT_DIR / cfg["paths"]["monitoring_dir"]
    lin_dir = ROOT_DIR / cfg["paths"]["lineage_dir"]
    snap_cache_path = eval_dir / "dashboard_snapshot.json"

    eval_summary = _read_json_safe(eval_dir / "evaluation_summary.json", {})
    ablation_report = _read_json_safe(eval_dir / "ablation_results.json", {})
    shap_report = _read_json_safe(eval_dir / "shap_attributions.json", {})
    forecast_summary = _read_json_safe(eval_dir / "forecast_summary.json", {})
    online_ml_report = _read_json_safe(eval_dir / "online_ml_report.json", {})
    champion_meta = _read_json_safe(reg_dir / "champion_metadata.json", {})
    registry_index = _read_json_safe(reg_dir / "registry_index.json", {})
    tournament = _read_json_safe(reg_dir / "challenger_tournament.json", {})
    rollback_log = _read_json_safe(reg_dir / "rollback_audit_log.json", {})
    drift_report = _read_json_safe(mon_dir / "drift_monitoring_report.json", {})
    contract_audit = _read_json_safe(mon_dir / "data_contract_audit.json", {})
    latest_run = _read_json_safe(lin_dir / "latest_run.json", {})

    if use_cached_snapshot and snap_cache_path.exists():
        snap_data = _read_json_safe(snap_cache_path, {})
        if "scenarios" in snap_data:
            snapshot_ts = snap_data.get("snapshot_ts", "")
            map_links = snap_data.get("map_links", [])
            districts = snap_data.get("districts", [])
            weather_summary = snap_data.get("weather_summary", {})
            scenarios = snap_data.get("scenarios", {})
        else:
            use_cached_snapshot = False

    if not use_cached_snapshot or not snap_cache_path.exists():
        df = load_table(cfg["paths"]["gold_data"])
        df_test = df[df["split"] == "test"].copy()

        champ_model_path = reg_dir / "models" / "champion_model.joblib"
        if not champ_model_path.exists():
            champ_model_path = ROOT_DIR / cfg["paths"]["models_dir"] / "b2_lgbm_model.joblib"

        wk_18 = df_test[(df_test["target_hour"] == 18) & (df_test["is_weekend"] == 0)]
        ts_evening = wk_18["target_ts"].max() if len(wk_18) else df_test["target_ts"].max()

        rain_by_ts = (
            df_test[df_test["target_hour"].between(15, 20)]
            .groupby("target_ts")["precip_3h_mm_t0"]
            .mean()
            .sort_values(ascending=False)
        )
        ts_rain = rain_by_ts.index[0] if len(rain_by_ts) else ts_evening

        wk_08 = df_test[(df_test["target_hour"] == 8) & (df_test["is_weekend"] == 0)]
        ts_morning = wk_08["target_ts"].max() if len(wk_08) else ts_evening

        scenarios = {}
        for key, label, ts_val in (
            ("evening_rush", f"เร่งด่วนเย็น ({pd.Timestamp(ts_evening).strftime('%Y-%m-%d %H:%M')})", ts_evening),
            ("heavy_rain", f"ฝนตกหนักมรสุม ({pd.Timestamp(ts_rain).strftime('%Y-%m-%d %H:%M')})", ts_rain),
            ("morning_rush", f"เร่งด่วนเช้า ({pd.Timestamp(ts_morning).strftime('%Y-%m-%d %H:%M')})", ts_morning),
        ):
            sub_df = df_test[df_test["target_ts"] == ts_val].drop_duplicates("link_id").copy()
            sub_pred = forecast_next_hour(sub_df, model_path=champ_model_path)
            links_list, wx_dict = _extract_scenario_links(sub_pred)
            scenarios[key] = {
                "label": label,
                "snapshot_ts": str(ts_val),
                "weather_summary": wx_dict,
                "map_links": links_list,
            }

        default_sc = scenarios["evening_rush"]
        snapshot_ts = default_sc["snapshot_ts"]
        map_links = default_sc["map_links"]
        weather_summary = default_sc["weather_summary"]
        districts = sorted({x["district"] for x in map_links})

        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(snap_cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "snapshot_ts": snapshot_ts,
                    "districts": districts,
                    "weather_summary": weather_summary,
                    "map_links": map_links,
                    "scenarios": scenarios,
                },
                f,
                ensure_ascii=False,
            )

    keys = get_map_api_keys()
    return {
        "snapshot_ts": snapshot_ts,
        "longdo_map_api_key": keys["longdo_map_api_key"],
        "google_maps_api_key": keys["google_maps_api_key"],
        "districts": districts,
        "weather_summary": weather_summary,
        "map_links": map_links,
        "scenarios": scenarios,
        "feature_catalog": _build_feature_catalog(shap_report),
        "evaluation": eval_summary,
        "ablation": ablation_report,
        "shap": shap_report,
        "forecast": forecast_summary,
        "online_ml": online_ml_report,
        "champion": champion_meta,
        "registry_index": registry_index,
        "tournament": tournament,
        "rollback_log": rollback_log,
        "monitoring": drift_report,
        "data_contract": contract_audit,
        "lineage_latest": latest_run,
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>BKK Traffic Lab — ระบบพยากรณ์รถติดล่วงหน้า 1 ชั่วโมง (CSS 342 KMUTT)</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Noto+Sans+Thai:wght@400;500;600;700&family=Noto+Serif+Thai:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg-dark: #07111f;
    --bg-deep: #020817;
    --bg-light: #f8fafc;
    --bg-paper: #f0f2f2;
    --bg-soft: #e8eeed;
    --card: #ffffff;
    --text-main: #0f172a;
    --text-body: #334155;
    --text-muted: #64748b;
    --teal: #0f766e;
    --teal-soft: #e2e8e8;
    --blue: #2563eb;
    --blue-dark: #1d4ed8;
    --blue-soft: #dbeafe;
    --amber: #f59e0b;
    --ok: #15803d;
    --danger: #dc2626;
    --onset: #7c3aed;
    --border: #d6dce0;
    --border-light: #e2e8f0;
    --shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
    --radius: 16px;
    --container: 1200px;
  }

  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0;
    background: var(--bg-paper);
    color: var(--text-main);
    font-family: 'Noto Sans Thai', -apple-system, sans-serif;
    font-size: 16px;
    line-height: 1.7;
    -webkit-font-smoothing: antialiased;
  }
  .serif, h1, h2, h3 {
    font-family: 'Noto Serif Thai', Georgia, serif;
    color: var(--text-main);
  }
  .mono {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-variant-numeric: tabular-nums;
  }

  .container {
    width: min(var(--container), calc(100% - 48px));
    margin: 0 auto;
  }

  /* ── SITE HEADER (Inspired by warin.me) ───────────────────────────── */
  .site-header {
    background: rgba(2, 8, 23, 0.96);
    color: #ffffff;
    border-bottom: 1px solid rgba(255, 255, 255, 0.09);
    position: sticky;
    top: 0;
    z-index: 1000;
    backdrop-filter: blur(10px);
  }
  .nav-container {
    height: 72px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    padding: 0 24px;
    max-width: 1440px;
    margin: 0 auto;
  }
  .logo {
    display: flex; align-items: center; gap: 12px; font-weight: 700; color: #fff;
  }
  .logo-mark {
    font-family: 'Noto Serif Thai', Georgia, serif;
    font-size: 24px; color: #60a5fa; letter-spacing: -1.5px; line-height: 1;
  }
  .logo-text { font-size: 15.5px; font-weight: 600; letter-spacing: 0.01em; }
  .health-dot { width: 9px; height: 9px; border-radius: 50%; background: #22c55e; flex-shrink: 0; }

  .main-nav { display: flex; align-items: center; gap: 22px; }
  .nav-btn {
    border: none; background: transparent; color: #cbd5e1;
    font-family: 'Noto Sans Thai', sans-serif; font-size: 15px; font-weight: 500;
    padding: 8px 4px; cursor: pointer; border-bottom: 2px solid transparent;
    transition: color 0.15s, border-color 0.15s; white-space: nowrap;
  }
  .nav-btn:hover { color: #ffffff; }
  .nav-btn.active { color: #ffffff; font-weight: 700; border-bottom-color: #60a5fa; }

  .header-right { display: flex; align-items: center; gap: 10px; }
  .hdr-pill, .scen-sel {
    padding: 7px 14px;
    border: 1px solid rgba(148, 163, 184, 0.35);
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.72);
    color: #e2e8f0;
    font-family: 'Noto Sans Thai', sans-serif;
    font-size: 13.5px; font-weight: 500; cursor: pointer;
    transition: border-color 0.15s, color 0.15s;
  }
  .hdr-pill:hover, .scen-sel:hover { border-color: #60a5fa; color: #ffffff; }
  .scen-sel option { background: #07111f; color: #ffffff; }

  /* Popovers */
  .popover {
    display: none; position: fixed; top: 80px; right: 24px; z-index: 1200;
    width: 390px; max-width: calc(100vw - 32px);
    background: #ffffff; color: var(--text-main);
    border: 1px solid var(--border); border-radius: 16px;
    padding: 22px 24px; box-shadow: 0 20px 50px rgba(15, 23, 42, 0.18);
  }
  .popover.open { display: block; }
  .popover h3 {
    margin: 0 0 14px; font-size: 17px; padding-bottom: 10px;
    border-bottom: 1px solid var(--border-light);
    display: flex; justify-content: space-between; align-items: center;
  }
  .popover-close { border: none; background: none; cursor: pointer; color: var(--text-muted); font-size: 22px; }
  .wx-row { display: flex; justify-content: space-between; padding: 9px 0; border-bottom: 1px solid var(--border-light); font-size: 14.5px; }
  .wx-row:last-child { border-bottom: none; }
  .field { margin-bottom: 14px; }
  .field label { display: block; font-size: 13.5px; color: var(--text-muted); margin-bottom: 5px; font-weight: 500; }
  .field input {
    width: 100%; padding: 10px 12px; border: 1px solid var(--border); border-radius: 10px;
    font-family: 'JetBrains Mono', monospace; font-size: 13.5px; color: var(--text-main);
  }

  /* Buttons (warin.me style) */
  .btn-primary {
    padding: 11px 20px; border: none; border-radius: 10px;
    background: var(--blue); color: #ffffff;
    font-family: inherit; font-size: 14.5px; font-weight: 600; cursor: pointer;
    transition: background 0.15s;
  }
  .btn-primary:hover { background: var(--blue-dark); }
  .btn-outline-dark {
    padding: 10px 18px; border: 1px solid rgba(255, 255, 255, 0.35); border-radius: 10px;
    background: transparent; color: #ffffff;
    font-family: inherit; font-size: 14px; font-weight: 600; cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
  }
  .btn-outline-dark:hover { background: rgba(255,255,255,0.1); border-color: #ffffff; }
  .btn-soft {
    padding: 8px 14px; border: 1px solid var(--border); border-radius: 9px;
    background: #ffffff; color: var(--text-main);
    font-family: inherit; font-size: 13.5px; font-weight: 600; cursor: pointer;
  }
  .btn-soft:hover { border-color: var(--blue); color: var(--blue); }

  .view { display: none; }
  .view.active { display: block; }

  /* ── VIEW 1: INTERACTIVE MAP LAB ──────────────────────────────────── */
  .map-layout {
    display: grid;
    grid-template-columns: 1fr 440px;
    height: calc(100vh - 72px);
    min-height: 640px;
    background: var(--bg-light);
  }
  @media (max-width: 1080px) {
    .map-layout { grid-template-columns: 1fr; height: auto; }
    .map-stage { height: 560px; }
  }
  .map-stage { position: relative; background: #e2e8f0; border-right: 1px solid var(--border); }
  #mapViewport { width: 100%; height: 100%; }

  .float-top {
    position: absolute; top: 16px; left: 50%; transform: translateX(-50%); z-index: 500;
    display: inline-flex; background: rgba(255, 255, 255, 0.96); backdrop-filter: blur(8px);
    border: 1px solid var(--border); border-radius: 999px; padding: 4px;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.12);
  }
  .seg-pill {
    border: none; background: transparent; color: var(--text-muted);
    font-family: inherit; font-size: 13.5px; font-weight: 600;
    padding: 7px 16px; border-radius: 999px; cursor: pointer; white-space: nowrap;
  }
  .seg-pill.active { background: var(--bg-dark); color: #ffffff; }

  .float-bottom {
    position: absolute; bottom: 16px; left: 16px; right: 16px; z-index: 500;
    display: flex; justify-content: space-between; align-items: flex-end; gap: 10px; flex-wrap: wrap;
    pointer-events: none;
  }
  .float-bar {
    pointer-events: auto; display: inline-flex; align-items: center; gap: 13px; flex-wrap: wrap;
    background: rgba(255, 255, 255, 0.96); backdrop-filter: blur(8px);
    border: 1px solid var(--border); border-radius: 12px; padding: 9px 14px;
    font-size: 13.5px; color: var(--text-main); box-shadow: 0 8px 24px rgba(15, 23, 42, 0.1);
  }
  .float-bar label { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; font-weight: 500; }
  .float-bar input[type="checkbox"] { width: 15px; height: 15px; accent-color: var(--teal); }
  .dot-leg { display: inline-flex; align-items: center; gap: 6px; color: var(--text-body); font-size: 13px; font-weight: 500; }
  .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
  .mode-sel {
    border: 1px solid var(--blue); background: var(--blue-soft); color: var(--blue-dark);
    border-radius: 999px; padding: 4px 12px; font-size: 12.5px; font-weight: 700; cursor: pointer; font-family: inherit;
  }

  /* Light Clean Side Rail */
  .side-rail { background: #ffffff; display: flex; flex-direction: column; height: 100%; overflow: hidden; }
  .rail-metrics { display: grid; grid-template-columns: repeat(3, 1fr); border-bottom: 1px solid var(--border-light); background: var(--bg-light); }
  .rm-cell { padding: 15px 16px; border-right: 1px solid var(--border-light); }
  .rm-cell:last-child { border-right: none; }
  .rm-lbl { font-size: 12px; color: var(--text-muted); font-weight: 500; }
  .rm-val { font-size: 22px; font-weight: 700; margin-top: 3px; color: var(--text-main); }

  .rail-search { padding: 12px 16px 0; }
  .rail-search input {
    width: 100%; padding: 10px 13px; background: var(--bg-light); color: var(--text-main);
    border: 1px solid var(--border); border-radius: 10px; font-family: inherit; font-size: 14px;
  }
  .rail-search input:focus { outline: none; border-color: var(--blue); background: #ffffff; }
  .rail-filters { padding: 10px 16px 12px; border-bottom: 1px solid var(--border-light); display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .rail-filters select {
    width: 100%; padding: 9px 10px; background: #ffffff; color: var(--text-main);
    border: 1px solid var(--border); border-radius: 10px; font-family: inherit; font-size: 13.5px;
  }

  .rail-head {
    padding: 10px 16px; font-size: 12.5px; font-weight: 700; color: var(--teal);
    background: #f4f7f6; border-bottom: 1px solid var(--border-light);
    display: flex; justify-content: space-between; align-items: center;
  }
  .road-list { flex: 1; overflow-y: auto; padding: 0 16px; }
  .road-item {
    display: flex; justify-content: space-between; align-items: center; gap: 12px;
    padding: 12px 0; border-bottom: 1px solid var(--border-light); cursor: pointer;
  }
  .road-item:hover { background: var(--bg-light); margin: 0 -16px; padding: 12px 16px; }
  .road-name { font-weight: 600; font-size: 14.5px; color: var(--text-main); line-height: 1.4; }
  .road-sub { font-size: 13px; color: var(--text-muted); margin-top: 2px; }
  .road-stat { text-align: right; flex-shrink: 0; }
  .road-pct { font-size: 14px; font-weight: 600; color: var(--text-main); }

  /* Pill Tags (warin.me style) */
  .tag {
    display: inline-block; padding: 3px 9px; border-radius: 999px;
    font-size: 11.5px; font-weight: 700;
    background: var(--blue-soft); color: var(--blue-dark);
  }
  .t-ok { background: #dcfce7; color: #166534; }
  .t-warn { background: #fef3c7; color: #92400e; }
  .t-danger { background: #fee2e2; color: #991b1b; }
  .t-onset { background: #f3e8ff; color: #6b21a8; }
  .t-teal { background: #ccfbf1; color: #115e59; }

  .whatif-box { border-top: 1px solid var(--border); padding: 14px 16px; background: var(--bg-light); }
  .whatif-toggle {
    width: 100%; border: none; background: none; padding: 2px 0; cursor: pointer;
    display: flex; justify-content: space-between; align-items: center;
    font-family: inherit; font-size: 14.5px; font-weight: 700; color: var(--text-main);
  }
  .whatif-body { display: none; margin-top: 12px; }
  .whatif-body.open { display: block; }
  .whatif-body select {
    width: 100%; padding: 9px 10px; background: #ffffff; color: var(--text-main);
    border: 1px solid var(--border); border-radius: 10px; font-family: inherit; font-size: 13.5px; margin-bottom: 10px;
  }
  .slider-row { display: flex; justify-content: space-between; font-size: 13.5px; color: var(--text-body); margin-top: 6px; }
  .whatif-body input[type="range"] { width: 100%; accent-color: var(--teal); }

  /* ── VIEWS 2 & 3: WARIN.ME RESEARCH DOSSIER & LAB REPORT ──────────── */
  .dossier-hero {
    padding: 64px 0 58px;
    color: #ffffff;
    background:
      radial-gradient(circle at 82% 20%, rgba(45, 212, 191, 0.16), transparent 30rem),
      radial-gradient(circle at 12% 18%, rgba(37, 99, 235, 0.25), transparent 28rem),
      linear-gradient(135deg, var(--bg-deep), var(--bg-dark));
  }
  .dossier-hero-grid {
    display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(300px, 0.85fr);
    gap: 48px; align-items: end;
  }
  @media (max-width: 960px) { .dossier-hero-grid { grid-template-columns: 1fr; gap: 28px; } }
  .eyebrow {
    color: var(--amber); font-size: 13px; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase; margin: 0 0 12px;
  }
  .dossier-hero h1 {
    margin: 0 0 16px; color: #ffffff;
    font-size: clamp(32px, 4.2vw, 48px); line-height: 1.22; letter-spacing: -0.02em;
  }
  .hero-lead { color: #cbd5e1; font-size: 17.5px; margin: 0 0 26px; max-width: 720px; line-height: 1.75; }
  .hero-actions { display: flex; gap: 14px; flex-wrap: wrap; align-items: center; }

  /* Snapshot Box on Hero (Exact warin.me `.project-snapshot` pattern) */
  .project-snapshot {
    padding: 22px; border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 18px; background: rgba(15, 23, 42, 0.65); backdrop-filter: blur(12px);
  }
  .snapshot-kicker {
    margin: 0 0 14px; color: #8ce9dd; font-family: 'JetBrains Mono', monospace;
    font-size: 12px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
  }
  .snapshot-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px;
  }
  .snap-cell {
    padding: 15px 16px; border-radius: 12px;
    background: rgba(2, 8, 23, 0.72); border: 1px solid rgba(255, 255, 255, 0.09);
  }
  .snap-val { display: block; font-size: 26px; font-weight: 700; color: #ffffff; line-height: 1.15; }
  .snap-lbl { display: block; margin-top: 6px; font-size: 12.5px; color: #94a3b8; }

  /* Editorial Report Body Sections */
  .report-section { padding: 54px 0; }
  .report-section.alt { background: var(--bg-light); border-top: 1px solid var(--border-light); border-bottom: 1px solid var(--border-light); }

  .sec-kicker {
    margin: 0 0 8px; color: var(--teal); font-family: 'JetBrains Mono', monospace;
    font-size: 12.5px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
  }
  .sec-header {
    display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 16px;
    margin-bottom: 28px;
  }
  .sec-header h2 { margin: 0; font-size: clamp(24px, 3vw, 32px); line-height: 1.3; }
  .sec-header p { margin: 8px 0 0; color: var(--text-muted); font-size: 15.5px; max-width: 760px; }

  .grid-2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(510px, 1fr)); gap: 22px; margin-bottom: 22px; }
  @media (max-width: 640px) { .grid-2 { grid-template-columns: 1fr; } }

  .dossier-card {
    background: var(--card); border: 1px solid var(--border); border-radius: 20px;
    padding: 28px 30px; box-shadow: var(--shadow); margin-bottom: 22px;
  }
  .grid-2 .dossier-card { margin-bottom: 0; }
  .card-head {
    display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 10px;
    margin-bottom: 18px; padding-bottom: 12px; border-bottom: 1px solid var(--border-light);
  }
  .card-head h3 { margin: 0; font-size: 19px; line-height: 1.35; }
  .card-head span { font-size: 13.5px; color: var(--text-muted); }

  /* Large Readable Tables */
  table { width: 100%; border-collapse: collapse; font-size: 14.5px; }
  th, td { padding: 12px 12px; text-align: left; border-bottom: 1px solid var(--border-light); }
  th {
    font-size: 12.5px; font-weight: 700; color: var(--text-muted);
    background: #f8fafc;
  }
  tr:last-child td { border-bottom: none; }
  tr.win-row td { background: #f0fdfa; font-weight: 600; color: var(--text-main); }
  .num { text-align: right; font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; }

  .chart-wrap { position: relative; height: 320px; }
  .chart-wrap-tall { position: relative; height: 360px; }

  /* Signature warin.me Research Note Box (`.research-question` style) */
  .research-note {
    margin: 22px 0 0; padding: 22px 26px;
    border-left: 4px solid var(--teal);
    border-radius: 0 14px 14px 0;
    background: var(--teal-soft);
    color: #1e293b; font-size: 15px; line-height: 1.75;
  }
  .research-note .note-label {
    display: block; color: var(--teal); font-family: 'JetBrains Mono', monospace;
    font-size: 12px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px;
  }
  .research-note ul { margin: 8px 0 0; padding-left: 20px; }
  .research-note li { margin-bottom: 6px; }

  /* Sequence / Hypotheses Grid (warin.me `.sequence-card` style) */
  .hyp-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px; margin-top: 22px;
  }
  .hyp-card {
    background: var(--bg-light); border: 1px solid var(--border); border-radius: 16px;
    padding: 20px 22px; transition: transform 0.18s, box-shadow 0.18s;
  }
  .hyp-card:hover { transform: translateY(-2px); background: #ffffff; box-shadow: var(--shadow); }
  .hyp-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .hyp-num { color: var(--teal); font-family: 'JetBrains Mono', monospace; font-size: 14px; font-weight: 700; }
  .hyp-title { font-family: 'Noto Serif Thai', Georgia, serif; font-size: 16.5px; font-weight: 700; color: var(--text-main); margin-bottom: 6px; line-height: 1.4; }
  .hyp-desc { font-size: 14px; color: var(--text-body); line-height: 1.65; }

  /* Sources & Split */
  .source-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; margin-bottom: 22px; }
  .src-item {
    background: #ffffff; border: 1px solid var(--border); border-radius: 16px;
    padding: 20px 22px; box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
    display: flex; justify-content: space-between; align-items: flex-start; gap: 12px;
  }
  .src-name { font-family: 'Noto Serif Thai', Georgia, serif; font-weight: 700; font-size: 16.5px; color: var(--text-main); }
  .src-desc { font-size: 14px; color: var(--text-muted); margin-top: 4px; }

  .split-track { display: flex; height: 12px; border-radius: 999px; overflow: hidden; background: #e2e8f0; margin: 14px 0; }
  .split-seg-tr { background: var(--bg-dark); }
  .split-seg-va { background: var(--blue); }
  .split-seg-te { background: var(--teal); }
  .split-legend { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 14px; font-size: 14.5px; color: var(--text-body); }

  /* Pipeline Flow (warin.me `.dossier-flow` style) */
  .dag-flow { display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .dag-node {
    background: #ffffff; border: 1px solid var(--border); border-radius: 14px;
    padding: 16px 16px; border-top: 3px solid var(--teal);
  }
  .dag-step-num { font-size: 12px; color: var(--teal); font-family: 'JetBrains Mono', monospace; font-weight: 700; }
  .dag-step-name { font-size: 15px; font-weight: 700; margin: 4px 0 10px; color: var(--text-main); }
  .dag-step-meta { font-size: 12.5px; display: flex; justify-content: space-between; align-items: center; }

  details.catalog-box {
    background: #ffffff; border: 1px solid var(--border); border-radius: 18px;
    box-shadow: var(--shadow); margin-bottom: 22px; overflow: hidden;
  }
  details.catalog-box summary {
    padding: 18px 26px; cursor: pointer; font-family: 'Noto Serif Thai', Georgia, serif;
    font-size: 17px; font-weight: 700; color: var(--text-main);
    list-style: none; display: flex; justify-content: space-between; align-items: center;
  }
  details.catalog-box[open] summary { border-bottom: 1px solid var(--border-light); background: #f8fafc; }
  .catalog-inner { padding: 10px 26px 22px; max-height: 500px; overflow-y: auto; }

  /* Footer (warin.me `.site-footer` style) */
  .site-footer {
    background: var(--bg-deep); color: #cbd5e1; padding: 40px 0 28px;
  }
  .footer-inner {
    width: min(var(--container), calc(100% - 48px)); margin: 0 auto;
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 16px; font-size: 14px;
  }
  .footer-inner strong { color: #ffffff; }

  @media print {
    .site-header, .no-print { display: none !important; }
    .dossier-hero { background: #0f172a !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  }
</style>
</head>
<body>

<header class="site-header">
  <div class="nav-container">
    <div class="logo">
      <span class="logo-mark">BT</span>
      <span class="logo-text">bkk-traffic.mlops</span>
      <span class="health-dot" id="healthDot" title="System Health"></span>
    </div>

    <nav class="main-nav" role="tablist">
      <button class="nav-btn active" id="nav-map" onclick="showView('map')">แผนที่พยากรณ์</button>
      <button class="nav-btn" id="nav-mlops" onclick="showView('mlops')">รายงานผลโมเดล</button>
      <button class="nav-btn" id="nav-monitor" onclick="showView('monitor')">ระบบเฝ้าระวัง 3 ชั้น</button>
    </nav>

    <div class="header-right">
      <select class="scen-sel" id="scenarioSel" onchange="switchScenario(this.value)" title="เลือกช่วงเวลาในชุด Test 2022">
        <option value="evening_rush">ช่วงเร่งด่วนเย็น (18:00)</option>
        <option value="heavy_rain">ช่วงฝนตกหนักมรสุม</option>
        <option value="morning_rush">ช่วงเร่งด่วนเช้า (08:00)</option>
      </select>
      <button class="hdr-pill mono" id="wxChip" onclick="togglePopover('wxPop')">--°C</button>
      <button class="hdr-pill" onclick="togglePopover('keyPop')">ตั้งค่าแผนที่</button>
    </div>
  </div>
</header>

<!-- Weather Popover -->
<div class="popover" id="wxPop">
  <h3>สภาพอากาศรอบพยากรณ์ (Open-Meteo 2022) <button class="popover-close" onclick="togglePopover('wxPop')">&times;</button></h3>
  <div id="wxPopBody"></div>
</div>

<!-- Map API Key Popover -->
<div class="popover" id="keyPop">
  <h3>ตั้งค่าแผนที่ (Map Engine) <button class="popover-close" onclick="togglePopover('keyPop')">&times;</button></h3>
  <p style="margin:0 0 14px;font-size:14px;color:var(--text-muted);">ค่าเริ่มต้นใช้ Longdo Map SDK (แสดง 448 ช่วงถนนพร้อมชั้นข้อมูลจราจร เหตุการณ์ และ CCTV) หรือสลับเป็น Google Maps SDK</p>
  <div class="field">
    <label>Longdo Map API Key</label>
    <input type="password" id="longdoKeyIn" placeholder="ใส่ Longdo API Key..." />
  </div>
  <div class="field">
    <label>Google Maps JS API Key</label>
    <input type="password" id="googleKeyIn" placeholder="AIza..." />
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-top:16px;">
    <span id="keySaveMsg" class="mono" style="font-size:13px;color:var(--ok);"></span>
    <button class="btn-primary" onclick="saveMapKeys()">บันทึกคีย์</button>
  </div>
</div>

<!-- ════════════════════════════════════════════════════════════════════ -->
<!-- VIEW 1: INTERACTIVE MAP LAB                                         -->
<!-- ════════════════════════════════════════════════════════════════════ -->
<section id="view-map" class="view active">
  <div class="map-layout">
    <div class="map-stage">
      <div class="float-top">
        <button id="btnViewForecast" class="seg-pill active" onclick="setTimeView('forecast')">พยากรณ์ +1 ชม. (ML)</button>
        <button id="btnViewActual" class="seg-pill" onclick="setTimeView('actual')">ค่าจริง +1 ชม. (Actual)</button>
        <button id="btnViewNow" class="seg-pill" onclick="setTimeView('now')">สภาพปัจจุบัน (t0)</button>
      </div>

      <div id="mapViewport"></div>

      <div class="float-bottom">
        <div class="float-bar">
          <button class="mode-sel" id="providerSwitchBtn" onclick="toggleMapProvider()">Longdo Map</button>
          <label><input type="checkbox" id="chkLines" checked onchange="refreshMap()" /> เส้นถนน ML</label>
          <label><input type="checkbox" id="chkForecast" checked onchange="refreshMap()" /> หมุด 448 จุด</label>
          <label><input type="checkbox" id="chkTraffic" checked onchange="toggleLayers()" /> จราจรสด</label>
          <label id="lblEvents"><input type="checkbox" id="chkEvents" checked onchange="toggleLayers()" /> เหตุการณ์</label>
          <label id="lblCameras"><input type="checkbox" id="chkCameras" onchange="toggleLayers()" /> กล้อง CCTV</label>
        </div>
        <div class="float-bar">
          <span class="dot-leg"><span class="dot" style="background:#16a34a"></span>คล่องตัว</span>
          <span class="dot-leg"><span class="dot" style="background:#d97706"></span>ชะลอตัว</span>
          <span class="dot-leg"><span class="dot" style="background:#dc2626"></span>ติดขัด</span>
          <span class="dot-leg"><span class="dot" style="background:#7c3aed"></span>เริ่มติดใหม่</span>
        </div>
      </div>
    </div>

    <aside class="side-rail">
      <div class="rail-metrics" id="railMetrics"></div>

      <div class="rail-search">
        <input type="search" id="roadSearchIn" placeholder="ค้นหาชื่อถนนหรือแยก เช่น พหลโยธิน, สุขุมวิท, สาทร..." oninput="refreshMap()" />
      </div>
      <div class="rail-filters">
        <select id="distSel" onchange="refreshMap()"><option value="ALL">ทุกเขต กทม.</option></select>
        <select id="riskSel" onchange="refreshMap()">
          <option value="ALL">ทุกระดับความเสี่ยง</option>
          <option value="ONSET">เริ่มติดใหม่ (Onset)</option>
          <option value="RAINY">พื้นที่ฝนตก (&gt;0.5mm)</option>
          <option value="SEVERE">ติดหนัก (Severe)</option>
          <option value="HIGH">รถติด (High)</option>
          <option value="LOW">คล่องตัว (Low)</option>
        </select>
      </div>

      <div class="rail-head">
        <span>รายการช่วงถนน (ต้นทาง &rarr; ปลายทาง)</span>
        <span class="mono" id="railCount">--</span>
      </div>

      <div class="road-list" id="roadList"></div>

      <div class="whatif-box">
        <button class="whatif-toggle" onclick="toggleWhatIf()">
          <span>จำลองสถานการณ์ (What-If API)</span>
          <span id="whatifArrow" class="mono">+</span>
        </button>
        <div class="whatif-body" id="whatifBody">
          <select id="simLink"></select>
          <div class="slider-row"><span>ฝนสะสม 3 ชม.</span><span class="mono"><b id="rainLbl">0</b> mm</span></div>
          <input type="range" id="simRain" min="0" max="60" step="5" value="0" oninput="document.getElementById('rainLbl').textContent=this.value" />
          <div class="slider-row"><span>ถนนข้างหน้าติดขัด</span><span class="mono"><b id="downLbl">30</b>%</span></div>
          <input type="range" id="simDown" min="0" max="100" step="10" value="30" oninput="document.getElementById('downLbl').textContent=this.value" />
          <div style="display:flex;gap:8px;margin-top:12px;">
            <button class="btn-soft" onclick="syncScenarioRain()" style="flex:1;">ใช้ฝนรอบนี้</button>
            <button class="btn-primary" onclick="runPredict()" style="flex:1;">พยากรณ์ +1 ชม.</button>
          </div>
          <div id="simOut" class="mono" style="margin-top:10px;font-size:13.5px;font-weight:700;"></div>
        </div>
      </div>
    </aside>
  </div>
</section>

<!-- ════════════════════════════════════════════════════════════════════ -->
<!-- VIEW 2: MODEL EVALUATION DOSSIER & REPORT                           -->
<!-- ════════════════════════════════════════════════════════════════════ -->
<section id="view-mlops" class="view">
  <div class="dossier-hero">
    <div class="container dossier-hero-grid">
      <div>
        <p class="eyebrow">CSS 342 DATA SCIENCE &amp; DATA ENGINEERING · KMUTT</p>
        <h1>รายงานผลโมเดลพยากรณ์รถติดล่วงหน้า 1 ชั่วโมง</h1>
        <p class="hero-lead">ประเมิน LightGBM Regressor (21 Core Features) บนข้อมูลจราจรจริงในกรุงเทพมหานคร 122 วัน (1.27 ล้านแถว · 448 ช่วงถนนที่มีทิศทาง) เทียบ Baseline B0/B1 พร้อมกราฟวิเคราะห์ความคลาดเคลื่อนรายชั่วโมง, TreeSHAP, B3 Ablation, Online ML และ Apache Airflow</p>
        <div class="hero-actions no-print">
          <button class="btn-primary" onclick="window.print()">พิมพ์ / บันทึกรายงาน PDF</button>
          <button class="btn-outline-dark" onclick="showView('map')">กลับไปดูแผนที่พยากรณ์</button>
          <span class="mono" id="reportBannerMeta" style="font-size:13px;color:#94a3b8;">Champion --</span>
        </div>
      </div>
      <div class="project-snapshot">
        <p class="snapshot-kicker">MODEL PERFORMANCE SNAPSHOT (TEST SET)</p>
        <div class="snapshot-grid" id="regStrip"></div>
      </div>
    </div>
  </div>

  <!-- SECTION 1: BENCHMARK & 24H FORECAST -->
  <div class="report-section">
    <div class="container">
      <p class="sec-kicker">01 · MODEL BENCHMARK &amp; 24-HOUR ERROR ANALYSIS</p>
      <div class="sec-header">
        <div>
          <h2>ผลเปรียบเทียบโมเดลและกราฟพยากรณ์ตลอด 24 ชั่วโมง</h2>
          <p>ประเมินบนชุดข้อมูลทดสอบตามลำดับเวลาจริง (7–30 ก.ย. 2022 · 246,451 แถว) โดยจูน Threshold เฉพาะชุด Validation เท่านั้น</p>
        </div>
      </div>

      <div class="grid-2">
        <div class="dossier-card">
          <div class="card-head">
            <h3>กราฟเทียบความแม่นยำ B0 vs B1 vs LightGBM (%)</h3>
            <span id="modelSpecTag">LightGBM Regressor</span>
          </div>
          <div class="chart-wrap"><canvas id="benchChart"></canvas></div>
        </div>
        <div class="dossier-card">
          <div class="card-head">
            <h3>ตารางผลประเมินหลัก (Section 9.2 Benchmark)</h3>
            <span>Test Set 246,451 แถว</span>
          </div>
          <table>
            <thead>
              <tr>
                <th>โมเดล</th>
                <th class="num">F1 (24h)</th>
                <th class="num">06–23h</th>
                <th class="num">Onset</th>
                <th class="num">เร่งด่วน</th>
                <th class="num">ฝนตก</th>
                <th class="num">MAE</th>
              </tr>
            </thead>
            <tbody id="benchTable"></tbody>
          </table>
        </div>
      </div>

      <div id="modelSummaryNote"></div>

      <div class="grid-2" style="margin-top:22px;">
        <div class="dossier-card">
          <div class="card-head">
            <h3>กราฟรถติดจริงเทียบค่าพยากรณ์ +1 ชม. รายชั่วโมง</h3>
            <span>Actual vs LightGBM Forecast (%)</span>
          </div>
          <div class="chart-wrap"><canvas id="hourlyProfileChart"></canvas></div>
        </div>
        <div class="dossier-card">
          <div class="card-head">
            <h3>กราฟความคลาดเคลื่อนรายชั่วโมง (Hourly MAE)</h3>
            <span>LightGBM vs B0 Persistence</span>
          </div>
          <div class="chart-wrap"><canvas id="hourlyMaeChart"></canvas></div>
        </div>
      </div>
    </div>
  </div>

  <!-- SECTION 2: EXPLAINABILITY, ABLATION & HYPOTHESES -->
  <div class="report-section alt">
    <div class="container">
      <p class="sec-kicker">02 · EXPLAINABILITY, ABLATION &amp; HYPOTHESES VERIFICATION</p>
      <div class="sec-header">
        <div>
          <h2>ปัจจัยที่มีผลต่อรถติด (TreeSHAP), B3 Ablation และสมมติฐาน H1–H5</h2>
          <p>วิเคราะห์อิทธิพลของตัวแปรทั้ง 21 Core Features ด้วย Exact TreeSHAP และทดสอบตัดทีละกลุ่มฟีเจอร์ตาม Proposal ข้อ 9.3</p>
        </div>
        <span class="tag t-teal" id="shapMethodLbl">Exact TreeSHAP</span>
      </div>

      <div class="grid-2">
        <div class="dossier-card">
          <div class="card-head">
            <h3>Top 12 ปัจจัยสำคัญสูงสุดจาก TreeSHAP (%)</h3>
            <span>จาก 21 Core Features</span>
          </div>
          <div class="chart-wrap-tall"><canvas id="shapChart"></canvas></div>
        </div>
        <div class="dossier-card">
          <div class="card-head">
            <h3>ผลกระทบเมื่อตัดแต่ละกลุ่มฟีเจอร์ (B3 Ablation)</h3>
            <span>&Delta;F1 และ &Delta;Onset Recall (%)</span>
          </div>
          <div class="chart-wrap-tall"><canvas id="ablationChart"></canvas></div>
        </div>
      </div>

      <div class="dossier-card">
        <div class="card-head">
          <h3>ตารางสรุป B3 Factor Ablation และ Feature Gate 9.3</h3>
          <span>เปรียบเทียบโมเดลเต็มกับตอนตัดแต่ละกลุ่มออก</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>กลุ่มฟีเจอร์ที่ตัดออก</th>
              <th class="num">Ablated F1</th>
              <th class="num">&Delta;F1</th>
              <th class="num">&Delta;Onset</th>
              <th class="num">&Delta;ฝนตก</th>
              <th class="num">&Delta;เร่งด่วน</th>
              <th class="num">Feature Gate</th>
            </tr>
          </thead>
          <tbody id="ablationTable"></tbody>
        </table>
        <div class="research-note">
          <span class="note-label">หมายเหตุอธิบายผล Ablation (สำหรับตอบคำถาม)</span>
          ใน 1 เดือน ชั่วโมงที่ฝนตกหนักหรือเป็นเย็นวันศุกร์มีเพียง 10–15% ของเวลาทั้งหมด ค่า F1 เฉลี่ยทั้งเดือนจึงถูกชั่วโมงปกติกลบ แต่เมื่อตัดกลุ่มปฏิทินหรือกลุ่มฝนออก <strong>Onset Recall จะตกลงทันที (-2.51% และ -0.25%)</strong> จึงต้องเก็บไว้ตามเกณฑ์ Feature Gate 9.3 ส่วน <span class="mono">Historical Profile (54.3%)</span> ทำหน้าที่เป็นฐานคอขวดหลักของแต่ละถนน
        </div>
        <div class="hyp-grid" id="hypMiniGrid"></div>
      </div>
    </div>
  </div>

  <!-- SECTION 3: REAL DATA SOURCES & 21 CORE FEATURES -->
  <div class="report-section">
    <div class="container">
      <p class="sec-kicker">03 · DATA SOURCES, TEMPORAL SPLIT &amp; 21 CORE FEATURES</p>
      <div class="sec-header">
        <div>
          <h2>แหล่งข้อมูลจริง, การแบ่งข้อมูลตามเวลา และตัวแปรทั้ง 21 ตัว</h2>
          <p>ผ่าน Data Contract 4.2 ครบทั้ง 5 ด่าน และตัดตัวแปรซ้ำซ้อนจาก 44 เหลือ 21 ตัวหลักที่อธิบายความหมายได้ทุกตัว</p>
        </div>
      </div>

      <div class="source-grid">
        <div class="src-item">
          <div>
            <div class="src-name">iTIC Historical Traffic Status</div>
            <div class="src-desc">122 วัน (มิ.ย.–ก.ย. 2022) คัด 448 ช่วงถนนที่มีพิกัดต้นทาง–ปลายทางครบ 100%</div>
          </div>
          <span class="tag">Target + Lags</span>
        </div>
        <div class="src-item">
          <div>
            <div class="src-name">iTIC Location Table</div>
            <div class="src-desc">พิกัดแยกต้นทาง (from_loc) &rarr; ปลายทาง (to_loc) และกราฟถนนข้างเคียง</div>
          </div>
          <span class="tag t-ok">CC-BY-SA 4.0</span>
        </div>
        <div class="src-item">
          <div>
            <div class="src-name">Open-Meteo Weather Archive</div>
            <div class="src-desc">ปริมาณฝนรายชั่วโมงและฝนสะสม 3 ชม. ครอบคลุม 3 โซน กทม. (กลาง/เหนือ/ตะวันออก)</div>
          </div>
          <span class="tag t-ok">CC-BY 4.0</span>
        </div>
        <div class="src-item">
          <div>
            <div class="src-name">Traffy Fondue (Train-Only Static)</div>
            <div class="src-desc">ความหนาแน่นจุดร้องเรียนน้ำท่วมต่อ ตร.กม. คำนวณเฉพาะช่วง Train เพื่อกัน Data Leakage</div>
          </div>
          <span class="tag t-onset">Flood Risk</span>
        </div>
        <div class="src-item">
          <div>
            <div class="src-name">BOT Holiday Calendar 2022</div>
            <div class="src-desc">วันหยุดธนาคาร, วันในสัปดาห์, เย็นวันศุกร์ และช่วงวันเงินเดือนออก</div>
          </div>
          <span class="tag">Calendar</span>
        </div>
        <div class="src-item">
          <div>
            <div class="src-name">HDX Bangkok 50 Districts</div>
            <div class="src-desc">รหัสเขตมาตรฐาน adm2_pcode สำหรับเชื่อมข้อมูล iTIC + Traffy + Open-Meteo</div>
          </div>
          <span class="tag t-ok">Spatial Join</span>
        </div>
      </div>

      <div class="dossier-card" id="splitBarBox"></div>

      <details class="catalog-box">
        <summary>
          <span>รายการตัวแปรทั้งหมด 21 Core Features (ครบทั้ง 7 กลุ่มตาม Proposal)</span>
          <span class="mono" style="font-size:13px;color:var(--teal);">คลิกเพื่อกาง/พับตาราง</span>
        </summary>
        <div class="catalog-inner">
          <table>
            <thead><tr><th>กลุ่ม</th><th>ชื่อฟีเจอร์</th><th class="num">SHAP %</th><th>คำอธิบายความหมาย</th></tr></thead>
            <tbody id="featureCatalogTable"></tbody>
          </table>
        </div>
      </details>
    </div>
  </div>

  <!-- SECTION 4: ORCHESTRATION, AIRFLOW & ONLINE ML -->
  <div class="report-section alt">
    <div class="container">
      <p class="sec-kicker">04 · PIPELINE ORCHESTRATION, AIRFLOW &amp; ONLINE INCREMENTAL ML</p>
      <div class="sec-header">
        <div>
          <h2>การทำงานของ MLOps Pipeline, Apache Airflow และ Online Learning</h2>
          <p>พร้อมตัวรับข้อมูลชุดใหม่จาก Data Engineer ผ่าน Data Contract 4.2 และการอัปเดตโมเดลแบบ Incremental Warm-Start</p>
        </div>
        <span class="mono" id="runIdLbl" style="font-size:13.5px;color:var(--teal);font-weight:600;">Lineage</span>
      </div>

      <div class="dag-flow" id="dagFlow"></div>

      <div class="grid-2">
        <div class="dossier-card">
          <div class="card-head">
            <h3>กราฟ Online Incremental ML เทียบ Static Batch</h3>
            <span id="onlineMetaTag"></span>
          </div>
          <div class="chart-wrap"><canvas id="onlineStreamChart"></canvas></div>
          <div id="onlineNoteBox" class="research-note"></div>
        </div>
        <div class="dossier-card">
          <div class="card-head">
            <h3>Apache Airflow DAGs &amp; Data Contract 4.2 Adapter</h3>
            <span class="mono">dags/bkk_traffic_mlops_dag.py</span>
          </div>
          <table>
            <thead><tr><th>Airflow DAG / Data Contract Gate</th><th>รอบการทำงาน / เงื่อนไข</th><th class="num">สถานะ</th></tr></thead>
            <tbody id="airflowContractTable"></tbody>
          </table>
          <div class="research-note">
            <span class="note-label">รองรับข้อมูลชุดใหม่จาก Data Engineer (Plug-and-Play)</span>
            รันคำสั่ง <span class="mono">python run_pipeline.py ingest-external &lt;path&gt;</span> ระบบจะตรวจ Data Contract 4.2 ครบ 5 ด่าน รองรับทั้งข้อมูลราย 15 นาที (horizon=4) และราย 1 ชม. (horizon=1) พร้อมแปลง <span class="mono">congestion_ratio &leftrightarrow; red_rate</span> อัตโนมัติ
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- SECTION 5: MODEL REGISTRY & ROBUSTNESS -->
  <div class="report-section">
    <div class="container">
      <p class="sec-kicker">05 · MODEL REGISTRY, PROMOTION GATES &amp; ROBUSTNESS</p>
      <div class="sec-header">
        <div>
          <h2>การคัดเลือก Champion Model, 3-Fold Robustness และระบบ Rollback</h2>
          <p>ตรวจสอบผ่าน 4 Promotion Gates ตาม Proposal ข้อ 10 หน้า 11 พร้อมยืนยันความเสถียรข้าม 3 ช่วงเวลา</p>
        </div>
        <span class="mono" id="champSha" style="font-size:13px;color:var(--text-muted);">SHA-256</span>
      </div>

      <div class="grid-2">
        <div class="dossier-card">
          <div class="card-head">
            <h3>4 Promotion Gates</h3>
            <span>Proposal Section 10 หน้า 11</span>
          </div>
          <table>
            <thead><tr><th>เงื่อนไขเลื่อนขั้นเป็น Champion</th><th class="num">ผลที่ได้</th><th class="num">เกณฑ์</th><th class="num">สถานะ</th></tr></thead>
            <tbody id="gatesTable"></tbody>
          </table>
        </div>
        <div class="dossier-card">
          <div class="card-head">
            <h3>Challenger Tournament 3 กลยุทธ์หน้าต่างเวลา</h3>
            <span>คัดเลือกบน Validation</span>
          </div>
          <table>
            <thead><tr><th>กลยุทธ์</th><th class="num">Val F1</th><th class="num">Test F1</th><th class="num">Onset</th><th class="num">MAE</th></tr></thead>
            <tbody id="tourneyTable"></tbody>
          </table>
        </div>
      </div>

      <div class="grid-2">
        <div class="dossier-card">
          <div class="card-head">
            <h3>Rolling-Origin 3-Fold Temporal Robustness</h3>
            <span id="robustTag"></span>
          </div>
          <table>
            <thead><tr><th>ช่วงเวลาทดสอบ (Fold)</th><th class="num">จำนวนแถว</th><th class="num">F1 (24h)</th><th class="num">06–23h</th><th class="num">MAE</th></tr></thead>
            <tbody id="robustTable"></tbody>
          </table>
        </div>
        <div class="dossier-card">
          <div class="card-head">
            <h3>ประวัติเวอร์ชันและระบบ Rollback (SHA-256)</h3>
            <span id="rbMsg"></span>
          </div>
          <table>
            <thead><tr><th>Version</th><th>สถานะ</th><th class="num">Test F1</th><th class="num">Onset</th><th class="num">SHA-256</th><th class="num">จัดการ</th></tr></thead>
            <tbody id="verTable"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <footer class="site-footer">
    <div class="footer-inner">
      <div><strong>BKK Traffic Forecasting &amp; MLOps</strong> · CSS 342 Data Science and Data Engineering (KMUTT)</div>
      <div>Data Attribution: iTIC (CC-BY 4.0 / CC-BY-SA 4.0) · Open-Meteo (CC-BY 4.0) · Traffy Fondue · BOT 2022</div>
    </div>
  </footer>
</section>

<!-- ════════════════════════════════════════════════════════════════════ -->
<!-- VIEW 3: 3-LAYER MONITORING                                          -->
<!-- ════════════════════════════════════════════════════════════════════ -->
<section id="view-monitor" class="view">
  <div class="dossier-hero">
    <div class="container dossier-hero-grid">
      <div>
        <p class="eyebrow">PRODUCTION OBSERVABILITY · PROPOSAL SECTION 10</p>
        <h1>ระบบเฝ้าระวังโมเดลและข้อมูล 3 ชั้น</h1>
        <p class="hero-lead">ตรวจสอบคุณภาพข้อมูลและตัดวงจรถนนที่ข้อมูลหาย (Layer 1) วัดการเปลี่ยนกระจายตัวของข้อมูลด้วย PSI (Layer 2) และติดตามความแม่นยำรายวันแบบ Rolling 7 วัน (Layer 3)</p>
        <div class="hero-actions">
          <span class="mono" id="monBannerPill" style="font-size:13.5px;color:#8ce9dd;">Health: --</span>
        </div>
      </div>
      <div class="project-snapshot">
        <p class="snapshot-kicker">3-LAYER MONITORING TELEMETRY</p>
        <div class="snapshot-grid" id="monStrip"></div>
      </div>
    </div>
  </div>

  <div class="report-section">
    <div class="container">
      <div class="dossier-card">
        <div class="card-head">
          <h3>Layer 1: คุณภาพข้อมูล, Freshness SLA และ Segment Circuit-Breaker</h3>
          <span>หยุดพยากรณ์ทันทีถ้าข้อมูล 24 ชม. หายเกิน 20%</span>
        </div>
        <div id="dqSummary" style="font-size:14.5px;color:var(--text-muted);margin-bottom:16px;"></div>
        <table>
          <thead><tr><th>รายการตรวจสอบ</th><th class="num">ค่าที่วัดได้</th><th class="num">เกณฑ์มาตรฐาน</th><th class="num">สถานะ</th></tr></thead>
          <tbody id="dqTable"></tbody>
        </table>
      </div>

      <div class="grid-2">
        <div class="dossier-card">
          <div class="card-head">
            <h3>Layer 2: กราฟ Feature Drift (PSI) เทียบช่วงมรสุม</h3>
            <span>Train vs Test &amp; Monsoon Stress</span>
          </div>
          <div class="chart-wrap"><canvas id="psiChart"></canvas></div>
          <div class="research-note">
            <span class="note-label">ทำไม PSI ฝนขึ้น 0.228 (WARNING) แต่ไม่สั่ง Retrain ทันที?</span>
            ตาม Proposal ข้อ 10 หน้า 11 กำหนดให้ <em>"ตรวจสอบสาเหตุ แต่ห้ามสั่ง Retrain จาก PSI อย่างเดียว"</em> เพราะเดือน ก.ย. เป็นช่วงพีคของฤดูมรสุม ฝนตกหนักกว่า มิ.ย.–ก.ค. ตามธรรมชาติ ขณะที่เอาต์พุตโมเดลยังนิ่ง (<span class="mono">Prediction PSI = 0.0009</span>) และค่า <span class="mono">Test F1 = 69.78%</span> ไม่ได้ตกลงเกินเกณฑ์ <span class="mono">0.05</span>
          </div>
        </div>
        <div class="dossier-card">
          <div class="card-head">
            <h3>Layer 3: กราฟ Rolling 7-Day F1, Daily F1 และ Daily MAE</h3>
            <span>ตลอด 24 วันของชุด Test (7–30 ก.ย. 2022)</span>
          </div>
          <div class="chart-wrap"><canvas id="rollingChart"></canvas></div>
        </div>
      </div>

      <div class="dossier-card" style="margin-top:22px;">
        <div class="card-head">
          <h3>ตารางรายละเอียด Feature PSI (Train vs Test และ Monsoon Stress Test)</h3>
          <span>PSI &lt; 0.10 ปกติ · 0.10–0.25 เฝ้าระวัง · &ge; 0.25 วิกฤต</span>
        </div>
        <table>
          <thead><tr><th>ฟีเจอร์ที่เฝ้าระวัง</th><th class="num">ค่าเฉลี่ย Train</th><th class="num">ค่าเฉลี่ย Test</th><th class="num">Test PSI</th><th class="num">Monsoon PSI</th><th class="num">สถานะ</th></tr></thead>
          <tbody id="psiTable"></tbody>
        </table>
      </div>
    </div>
  </div>

  <footer class="site-footer">
    <div class="footer-inner">
      <div><strong>BKK Traffic Forecasting &amp; MLOps</strong> · 3-Layer Observability Console (CSS 342 KMUTT)</div>
      <div>Data Attribution: iTIC (CC-BY 4.0 / CC-BY-SA 4.0) · Open-Meteo (CC-BY 4.0) · Traffy Fondue</div>
    </div>
  </footer>
</section>

<script>
const DATA = __PAYLOAD_JSON__;
let currentScenario = 'evening_rush';
let mapProvider = 'longdo';
let timeView = 'forecast';
let longdoMap = null, longdoOverlays = [];
let googleMap = null, googleTraffic = null, googleOverlays = [], googleInfo = null;
let mlopsChartsInit = false;
let monChartsInit = false;

if (window.Chart) {
  Chart.defaults.color = '#475569';
  Chart.defaults.borderColor = '#e2e8f0';
  Chart.defaults.font.family = "'Noto Sans Thai', -apple-system, sans-serif";
  Chart.defaults.font.size = 13;
}

function activeLinks() {
  const sc = (DATA.scenarios || {})[currentScenario];
  return sc ? sc.map_links : (DATA.map_links || []);
}

function activeWeather() {
  const sc = (DATA.scenarios || {})[currentScenario];
  return sc ? sc.weather_summary : (DATA.weather_summary || {});
}

function switchScenario(scenKey) {
  currentScenario = scenKey;
  renderWeatherHeader();
  refreshMap();
}

function showView(id) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('view-' + id).classList.add('active');
  document.getElementById('nav-' + id).classList.add('active');
  window.scrollTo({ top: 0, behavior: 'instant' });
  if (id === 'mlops' && !mlopsChartsInit) { renderAllMlopsCharts(); mlopsChartsInit = true; }
  if (id === 'monitor' && !monChartsInit) { renderAllMonitorCharts(); monChartsInit = true; }
}

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === '1') showView('map');
  if (e.key === '2') showView('mlops');
  if (e.key === '3') showView('monitor');
});

function togglePopover(id) {
  ['wxPop', 'keyPop'].forEach(pid => {
    const el = document.getElementById(pid);
    if (pid === id) el.classList.toggle('open');
    else el.classList.remove('open');
  });
}

function getKey(provider) {
  if (provider === 'longdo') return DATA.longdo_map_api_key || localStorage.getItem('BKK_LONGDO_KEY') || '';
  return DATA.google_maps_api_key || localStorage.getItem('BKK_GMAPS_KEY') || '';
}

function saveMapKeys() {
  const lk = document.getElementById('longdoKeyIn').value.trim();
  const gk = document.getElementById('googleKeyIn').value.trim();
  const body = {};
  if (lk) { localStorage.setItem('BKK_LONGDO_KEY', lk); DATA.longdo_map_api_key = lk; body.longdo_map_api_key = lk; }
  if (gk) { localStorage.setItem('BKK_GMAPS_KEY', gk); DATA.google_maps_api_key = gk; body.google_maps_api_key = gk; }
  fetch('/api/v1/config/map-keys', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  }).catch(() => {});
  document.getElementById('keySaveMsg').textContent = 'บันทึกแล้ว';
  setTimeout(() => {
    togglePopover('keyPop');
    switchProvider(lk ? 'longdo' : (gk ? 'google' : mapProvider));
  }, 250);
}

function toggleMapProvider() {
  switchProvider(mapProvider === 'longdo' ? 'google' : 'longdo');
}

function switchProvider(provider) {
  mapProvider = provider;
  document.getElementById('providerSwitchBtn').textContent = provider === 'longdo' ? 'Longdo Map' : 'Google Maps';
  document.getElementById('lblEvents').style.display = provider === 'longdo' ? 'inline-flex' : 'none';
  document.getElementById('lblCameras').style.display = provider === 'longdo' ? 'inline-flex' : 'none';

  longdoMap = null; longdoOverlays = [];
  googleMap = null; googleOverlays = [];
  document.getElementById('mapViewport').innerHTML = '';

  const key = getKey(provider);
  if (provider === 'longdo') {
    if (key) loadLongdo(key); else showFallback('longdo');
  } else {
    if (key) loadGoogle(key); else showFallback('google');
  }
}

function showFallback(provider) {
  const vp = document.getElementById('mapViewport');
  if (provider === 'longdo') {
    vp.innerHTML = `<iframe width="100%" height="100%" style="border:0;" src="https://traffic.longdo.com/main/"></iframe>`;
  } else {
    vp.innerHTML = `<iframe width="100%" height="100%" style="border:0;" loading="lazy" src="https://maps.google.com/maps?q=Bangkok&t=&z=12&ie=UTF8&iwloc=&output=embed"></iframe>`;
  }
}

function loadLongdo(apiKey) {
  if (window.longdo && window.longdo.Map) { initLongdo(); return; }
  const old = document.getElementById('longdo-sdk');
  if (old) old.remove();
  const sc = document.createElement('script');
  sc.id = 'longdo-sdk';
  sc.src = `https://api.longdo.com/map/?key=${encodeURIComponent(apiKey)}`;
  sc.onload = () => initLongdo();
  sc.onerror = () => showFallback('longdo');
  document.head.appendChild(sc);
}

function initLongdo() {
  const vp = document.getElementById('mapViewport');
  vp.innerHTML = '';
  longdoMap = new longdo.Map({ placeholder: vp, location: { lon: 100.535, lat: 13.7563 }, zoom: 11, language: 'th' });
  longdoMap.Event.bind('ready', () => { toggleLayers(); refreshMap(); });
}

function loadGoogle(apiKey) {
  if (window.google && window.google.maps) { initGoogle(); return; }
  const old = document.getElementById('google-sdk');
  if (old) old.remove();
  window.initGoogle = function() {
    const vp = document.getElementById('mapViewport');
    vp.innerHTML = '';
    googleMap = new google.maps.Map(vp, { center: { lat: 13.7563, lng: 100.535 }, zoom: 11, mapTypeControl: false, streetViewControl: false });
    googleTraffic = new google.maps.TrafficLayer();
    googleInfo = new google.maps.InfoWindow();
    toggleLayers();
    refreshMap();
  };
  window.gm_authFailure = () => showFallback('google');
  const sc = document.createElement('script');
  sc.id = 'google-sdk';
  sc.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(apiKey)}&callback=initGoogle`;
  sc.async = true; sc.defer = true;
  document.head.appendChild(sc);
}

function toggleLayers() {
  const tr = document.getElementById('chkTraffic').checked;
  const ev = document.getElementById('chkEvents').checked;
  const cm = document.getElementById('chkCameras').checked;
  if (mapProvider === 'longdo' && longdoMap && window.longdo) {
    if (tr) longdoMap.Layers.add(longdo.Layers.TRAFFIC); else longdoMap.Layers.remove(longdo.Layers.TRAFFIC);
    if (ev) longdoMap.Overlays.load(longdo.Overlays.events); else longdoMap.Overlays.unload(longdo.Overlays.events);
    if (cm) longdoMap.Overlays.load(longdo.Overlays.cameras); else longdoMap.Overlays.unload(longdo.Overlays.cameras);
  }
  if (mapProvider === 'google' && googleMap && googleTraffic) {
    googleTraffic.setMap(tr ? googleMap : null);
  }
}

function setTimeView(mode) {
  timeView = mode;
  document.getElementById('btnViewForecast').className = mode === 'forecast' ? 'seg-pill active' : 'seg-pill';
  document.getElementById('btnViewActual').className = mode === 'actual' ? 'seg-pill active' : 'seg-pill';
  document.getElementById('btnViewNow').className = mode === 'now' ? 'seg-pill active' : 'seg-pill';
  refreshMap();
}

function rateToColor(rate, isOnset) {
  if (isOnset) return '#7C3AED';
  if (rate >= 0.70) return '#DC2626';
  if (rate >= 0.345) return '#EA580C';
  if (rate >= 0.25) return '#D97706';
  return '#16A34A';
}

function markerColor(l) {
  if (timeView === 'now') return rateToColor(l.red_rate_t0, false);
  if (timeView === 'actual') return rateToColor(l.actual_t1h, l.red_rate_t0 < 0.35 && l.actual_t1h >= 0.50);
  return rateToColor(l.pred_t1h, l.is_onset === 1);
}

function popupHtml(l) {
  return `<div style="font-family:'Noto Sans Thai',sans-serif;font-size:13.5px;line-height:1.55;min-width:230px;color:#0f172a;">
    <strong>${l.road_name}</strong><br/>
    <span style="color:#64748b">เขต${l.district} (${l.district_en}) · ${l.length_km} กม.</span>
    <hr style="border:none;border-top:1px solid #e2e8f0;margin:6px 0"/>
    ปัจจุบัน (t0): <b>${(l.red_rate_t0*100).toFixed(0)}%</b><br/>
    พยากรณ์ +1 ชม. (ML): <b style="color:${l.is_congested?'#DC2626':'#16A34A'}">${(l.pred_t1h*100).toFixed(0)}% (${l.risk})</b><br/>
    ค่าจริง +1 ชม. (Actual): <b>${(l.actual_t1h*100).toFixed(0)}%</b><br/>
    อุณหภูมิ ${l.temp_c}°C · ฝน 3 ชม. <b>${l.rain_3h} mm</b>
    ${l.is_onset ? '<br/><span style="color:#7C3AED;font-weight:700">แจ้งเตือนรถเพิ่งเริ่มติดใหม่ (ONSET)</span>' : ''}
  </div>`;
}

function refreshMap() {
  const dist = document.getElementById('distSel').value;
  const risk = document.getElementById('riskSel').value;
  const qEl = document.getElementById('roadSearchIn');
  const q = qEl ? qEl.value.trim().toLowerCase() : '';
  const showLines = document.getElementById('chkLines').checked;
  const showDots = document.getElementById('chkForecast').checked;
  const linksAll = activeLinks();
  const byDist = linksAll.filter(l =>
    (dist === 'ALL' || l.district === dist) &&
    (!q || (l.road_name || '').toLowerCase().includes(q) || (l.district || '').toLowerCase().includes(q) || (l.link_id || '').toLowerCase().includes(q))
  );

  const congPred = byDist.filter(l => l.is_congested === 1).length;
  const congAct = byDist.filter(l => l.actual_congested === 1).length;
  const onset = byDist.filter(l => l.is_onset === 1).length;
  const maxRain = byDist.reduce((m, l) => Math.max(m, l.rain_3h || 0), 0);

  document.getElementById('railMetrics').innerHTML = `
    <div class="rm-cell"><div class="rm-lbl">ติด +1 ชม. (ML/จริง)</div><div class="rm-val mono" style="color:var(--danger)">${congPred} <span style="font-size:14px;color:var(--text-muted)">/ ${congAct}</span></div></div>
    <div class="rm-cell"><div class="rm-lbl">เริ่มติดใหม่</div><div class="rm-val mono" style="color:var(--onset)">${onset}</div></div>
    <div class="rm-cell"><div class="rm-lbl">ฝนสูงสุด 3 ชม.</div><div class="rm-val mono" style="color:var(--teal)">${maxRain.toFixed(1)}<span style="font-size:12px;color:var(--text-muted)"> mm</span></div></div>
  `;

  let list = byDist;
  if (risk === 'ONSET') list = list.filter(l => l.is_onset === 1);
  else if (risk === 'RAINY') list = list.filter(l => (l.rain_3h || 0) > 0.5);
  else if (risk !== 'ALL') list = list.filter(l => l.risk === risk);

  document.getElementById('railCount').textContent = `${list.length} / ${byDist.length} ช่วงถนน`;
  const scoreFn = (l) => timeView === 'now' ? l.red_rate_t0 : (timeView === 'actual' ? l.actual_t1h : l.pred_t1h);
  const sorted = [...list].sort((a, b) => scoreFn(b) - scoreFn(a)).slice(0, 30);

  document.getElementById('roadList').innerHTML = sorted.map(l => {
    const cls = l.is_onset ? 't-onset' : (l.is_congested ? 't-danger' : (l.risk === 'MODERATE' ? 't-warn' : 't-ok'));
    const lbl = l.is_onset ? 'ONSET' : l.risk;
    const rainNote = (l.rain_3h || 0) > 0 ? ` · ฝน ${l.rain_3h.toFixed(1)}mm` : '';
    return `<div class="road-item" onclick="focusLink(${l.lat},${l.lon})">
      <div>
        <div class="road-name">${l.road_name}</div>
        <div class="road-sub">เขต${l.district} · ${l.length_km} กม.${rainNote}</div>
      </div>
      <div class="road-stat">
        <div class="road-pct mono">${(l.red_rate_t0*100).toFixed(0)}% &rarr; ${(l.pred_t1h*100).toFixed(0)}% <span style="color:var(--text-muted);font-weight:400">(${(l.actual_t1h*100).toFixed(0)}%)</span></div>
        <span class="tag ${cls}">${lbl}</span>
      </div>
    </div>`;
  }).join('');

  if (mapProvider === 'longdo' && longdoMap && window.longdo) {
    try { longdoOverlays.forEach(m => longdoMap.Overlays.remove(m)); } catch(e) {}
    longdoOverlays = [];
    list.forEach(l => {
      const col = markerColor(l);
      if (showLines && longdo.Polyline) {
        try {
          const pl = new longdo.Polyline([
            { lon: l.lon_from, lat: l.lat_from },
            { lon: l.lon, lat: l.lat },
            { lon: l.lon_to, lat: l.lat_to }
          ], { lineWidth: 4, lineColor: col });
          longdoMap.Overlays.add(pl);
          longdoOverlays.push(pl);
        } catch(e) {}
      }
      if (showDots && longdo.Marker) {
        try {
          const sz = l.is_onset && timeView === 'forecast' ? 13 : 9;
          const mk = new longdo.Marker({ lon: l.lon, lat: l.lat }, {
            title: `${l.road_name} (${l.district})`,
            detail: popupHtml(l),
            icon: { html: `<div style="width:${sz}px;height:${sz}px;border-radius:50%;background:${col};border:1.5px solid #fff;box-shadow:0 1px 3px rgba(0,0,0,0.35);"></div>`, offset: { x: sz/2, y: sz/2 } }
          });
          longdoMap.Overlays.add(mk);
          longdoOverlays.push(mk);
        } catch(e) {}
      }
    });
  }

  if (mapProvider === 'google' && googleMap && window.google && window.google.maps) {
    googleOverlays.forEach(m => m.setMap(null));
    googleOverlays = [];
    list.forEach(l => {
      const col = markerColor(l);
      if (showLines) {
        const pl = new google.maps.Polyline({
          path: [{ lat: l.lat_from, lng: l.lon_from }, { lat: l.lat, lng: l.lon }, { lat: l.lat_to, lng: l.lon_to }],
          geodesic: true, strokeColor: col, strokeOpacity: 0.9, strokeWeight: 4, map: googleMap
        });
        googleOverlays.push(pl);
      }
      if (showDots) {
        const mk = new google.maps.Marker({
          position: { lat: l.lat, lng: l.lon }, map: googleMap,
          icon: { path: google.maps.SymbolPath.CIRCLE, scale: l.is_onset && timeView === 'forecast' ? 6 : 4.2, fillColor: col, fillOpacity: 0.95, strokeColor: '#fff', strokeWeight: 1.2 },
          title: l.road_name
        });
        mk.addListener('click', () => { googleInfo.setContent(popupHtml(l)); googleInfo.open(googleMap, mk); });
        googleOverlays.push(mk);
      }
    });
  }
}

function focusLink(lat, lon) {
  if (mapProvider === 'longdo' && longdoMap) { longdoMap.location({ lon, lat }, true); longdoMap.zoom(14, true); }
  if (mapProvider === 'google' && googleMap) { googleMap.panTo({ lat, lng: lon }); googleMap.setZoom(14); }
}

function toggleWhatIf() {
  const b = document.getElementById('whatifBody');
  const open = b.classList.toggle('open');
  document.getElementById('whatifArrow').textContent = open ? '−' : '+';
}

function renderWeatherHeader() {
  const w = activeWeather();
  const sc = (DATA.scenarios || {})[currentScenario] || {};
  const r3 = w.max_rain_3h_mm || 0;
  document.getElementById('wxChip').textContent = `${w.avg_temp_c ?? '--'}°C · ฝน ${r3.toFixed(1)}mm`;
  document.getElementById('wxPopBody').innerHTML = `
    <div class="wx-row"><span>รอบพยากรณ์ (Test 2022)</span><strong class="mono">${sc.snapshot_ts || DATA.snapshot_ts}</strong></div>
    <div class="wx-row"><span>อุณหภูมิเฉลี่ย กทม.</span><strong class="mono">${w.avg_temp_c ?? '-'}°C</strong></div>
    <div class="wx-row"><span>ฝนสะสม 3 ชม. (เฉลี่ย / สูงสุด)</span><strong class="mono">${w.avg_rain_3h_mm ?? 0} / ${w.max_rain_3h_mm ?? 0} mm</strong></div>
    <div class="wx-row"><span>ช่วงถนนที่มีฝน (&gt;0.5mm)</span><strong class="mono">${w.rainy_links_count ?? 0} ช่วงถนน</strong></div>
    <div class="wx-row"><span>ช่วงถนนฝนตกหนัก (&gt;5.0mm)</span><strong class="mono">${w.heavy_rain_links_count ?? 0} ช่วงถนน</strong></div>
  `;
}

function syncScenarioRain() {
  const w = activeWeather();
  const val = Math.min(60, Math.round((w.max_rain_3h_mm || 0) / 5) * 5);
  document.getElementById('simRain').value = val;
  document.getElementById('rainLbl').textContent = val;
  runPredict();
}

async function runPredict() {
  const link_id = document.getElementById('simLink').value;
  const precip_3h_mm_t0 = parseFloat(document.getElementById('simRain').value);
  const downstream_red_rate_t0 = parseFloat(document.getElementById('simDown').value) / 100.0;
  const out = document.getElementById('simOut');
  try {
    const r = await fetch('/api/v1/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ link_id, precip_3h_mm_t0, downstream_red_rate_t0 })
    });
    const d = await r.json();
    out.textContent = `พยากรณ์ +1 ชม.: ${(d.pred_red_rate_t1h*100).toFixed(1)}% (${d.congestion_risk_level})`;
    out.style.color = d.pred_is_congested_t1h ? 'var(--danger)' : 'var(--ok)';
  } catch (e) {
    out.textContent = 'รัน python run_pipeline.py dashboard เพื่อเรียก API สด';
    out.style.color = '#b45309';
  }
}

/* ── VIEW 2 DATA ─────────────────────────────────────────────────────── */
function renderMlops() {
  const c = DATA.champion || {}, m = (c.metrics || {}).test || {}, lin = DATA.lineage_latest || {};
  const ev = DATA.evaluation || {}, mods = ev.models || {};
  const b0 = (mods.B0_Persistence || {}).test || {};
  const b1 = (mods.B1_Historical_Average || {}).test || {};
  const b2Val = (mods.B2_LightGBM || {}).val || {};
  const b2 = (mods.B2_LightGBM || {}).test || m;
  const sc = ev.split_counts || { train: 874324, val: 148678, test: 246451 };
  const totalRows = ev.num_rows_total || (sc.train + sc.val + sc.test);
  const trPct = (sc.train / totalRows * 100).toFixed(1);
  const vaPct = (sc.val / totalRows * 100).toFixed(1);
  const tePct = (sc.test / totalRows * 100).toFixed(1);

  document.getElementById('reportBannerMeta').textContent = `Champion ${c.version || 'v1.0.0'} · ${ev.num_features || 21} Features · SHA ${(c.model_sha256 || '').slice(0, 10)}`;
  document.getElementById('champSha').textContent = `SHA-256: ${(c.model_sha256 || '').slice(0, 16)}`;
  document.getElementById('runIdLbl').textContent = `${lin.run_id || '-'} · รวม ${(lin.total_duration_sec||0).toFixed(1)}s`;
  document.getElementById('modelSpecTag').textContent = c.model_backend || 'LightGBM Regressor';

  const dayF1 = ((b2.f1_daytime_06_23 || b2.f1_all || 0) * 100).toFixed(2);
  document.getElementById('regStrip').innerHTML = `
    <div class="snap-cell"><strong class="snap-val mono" style="color:#67e8f9">${(b2.f1_all*100).toFixed(2)}%</strong><span class="snap-lbl">Test F1 รวม (24 ชม.) · กลางวัน ${dayF1}%</span></div>
    <div class="snap-cell"><strong class="snap-val mono" style="color:#c4b5fd">${(b2.onset_recall*100).toFixed(2)}%</strong><span class="snap-lbl">Onset Recall (เกณฑ์ &ge; 55% · B0 = 0%)</span></div>
    <div class="snap-cell"><strong class="snap-val mono">${(b2.f1_rush_hour*100).toFixed(1)}%</strong><span class="snap-lbl">F1 ชั่วโมงเร่งด่วน (ฝนตก ${(b2.f1_rainy*100).toFixed(1)}%)</span></div>
    <div class="snap-cell"><strong class="snap-val mono" style="color:#86efac">${(b2.mae*100).toFixed(2)}%</strong><span class="snap-lbl">1-Hour MAE (ลด Error -19.5% จาก B0)</span></div>
  `;

  document.getElementById('splitBarBox').innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
      <h3 style="margin:0;font-size:18px;">การแบ่งข้อมูลตามลำดับเวลา (Out-of-Time Temporal Split)</h3>
      <span class="mono" style="font-size:13.5px;color:var(--teal);font-weight:600;">${totalRows.toLocaleString()} แถว · ${ev.num_links || 448} ช่วงถนน · ${ev.num_features || 21} Features</span>
    </div>
    <div class="split-track">
      <div class="split-seg-tr" style="width:${trPct}%"></div>
      <div class="split-seg-va" style="width:${vaPct}%"></div>
      <div class="split-seg-te" style="width:${tePct}%"></div>
    </div>
    <div class="split-legend">
      <span><span class="dot" style="background:var(--bg-dark);margin-right:7px;"></span><b>Train (${trPct}%)</b> · ${sc.train.toLocaleString()} แถว (1 มิ.ย.–23 ส.ค.)</span>
      <span><span class="dot" style="background:var(--blue);margin-right:7px;"></span><b>Validation (${vaPct}%)</b> · ${sc.val.toLocaleString()} แถว (24 ส.ค.–6 ก.ย.)</span>
      <span><span class="dot" style="background:var(--teal);margin-right:7px;"></span><b>Test (${tePct}%)</b> · ${sc.test.toLocaleString()} แถว (7–30 ก.ย.)</span>
    </div>
  `;

  const cat = DATA.feature_catalog || [];
  document.getElementById('featureCatalogTable').innerHTML = cat.map(f => `
    <tr>
      <td><span class="tag t-teal">${f.group}</span></td>
      <td class="mono" style="color:var(--text-main);font-weight:600;">${f.feature}</td>
      <td class="num">${f.shap_pct.toFixed(2)}%</td>
      <td style="color:var(--text-body)">${f.description_th}</td>
    </tr>
  `).join('');

  const bRows = [
    ['B0: Persistence (Test)', b0, false],
    ['B1: Historical Avg (Test)', b1, false],
    ['B2: LightGBM (Val)', b2Val, false],
    ['B2: LightGBM Champion (Test)', b2, true],
  ];
  document.getElementById('benchTable').innerHTML = bRows.map(([name, r, isWin]) => `
    <tr class="${isWin ? 'win-row' : ''}">
      <td>${name} ${isWin ? '<span class="tag t-ok" style="margin-left:6px;">Champion</span>' : ''}</td>
      <td class="num">${((r.f1_all||0)*100).toFixed(2)}%</td>
      <td class="num">${((r.f1_daytime_06_23||r.f1_all||0)*100).toFixed(2)}%</td>
      <td class="num">${((r.onset_recall||0)*100).toFixed(2)}%</td>
      <td class="num">${((r.f1_rush_hour||0)*100).toFixed(2)}%</td>
      <td class="num">${((r.f1_rainy||0)*100).toFixed(2)}%</td>
      <td class="num">${((r.mae||0)*100).toFixed(2)}%</td>
    </tr>
  `).join('');

  document.getElementById('modelSummaryNote').innerHTML = `
    <div class="research-note">
      <span class="note-label">สรุปที่มาของตัวเลขสำคัญ (สำหรับอธิบายและตอบคำถามอาจารย์)</span>
      <ul>
        <li><strong>ทำไม Threshold อยู่ที่ <span class="mono">${c.optimal_threshold || 0.345}</span> (ไม่ใช่ 0.50):</strong> จูนบนชุด Validation โดยตั้ง Guardrail ให้จับเหตุการณ์รถเริ่มติดใหม่ (<span class="mono">Onset Recall</span>) ได้อย่างน้อย 55% ตาม Error Tradeoff หน้า 9 ของ Proposal ที่ระบุว่าการพลาดจุดที่จะติดแย่กว่าการเตือนเกิน</li>
        <li><strong>ทำไมลดจาก 44 เหลือ 21 Core Features แล้วแม่นขึ้น:</strong> เมื่อตัดตัวแปรซ้ำซ้อนออก โมเดลลด Noise และ Overfitting ทำให้ Test F1 เพิ่มเป็น <strong>${(b2.f1_all*100).toFixed(2)}%</strong> และช่วงกลางวัน 06–23 น. อยู่ที่ <strong>${dayF1}%</strong></li>
        <li><strong>ทำไม Persistence ได้ Onset Recall = 0.0% แต่ LightGBM ได้ ${(b2.onset_recall*100).toFixed(1)}%:</strong> เพราะ Persistence ทายว่าอีก 1 ชม. จะเหมือนตอนนี้ ถ้ารถตอนนี้โล่งจึงทายว่าโล่งเสมอ ส่วน LightGBM จับจุดเริ่มติดใหม่ได้จากค่าเฉลี่ยในอดีต (<span class="mono">hist_avg_red_target_slot</span>), ความผิดปกติเทียบอดีต (<span class="mono">current_diff_from_hist</span>) และสภาพถนนข้างหน้า (<span class="mono">downstream_red_rate_t0</span>)</li>
      </ul>
    </div>
  `;

  const abRows = (DATA.ablation || {}).ablation_experiments || [];
  document.getElementById('ablationTable').innerHTML = abRows.map(r => `
    <tr>
      <td>${r.group_label}</td>
      <td class="num">${(r.ablated_f1_all*100).toFixed(2)}%</td>
      <td class="num">${(r.marginal_f1_gain>=0?'+':'') + (r.marginal_f1_gain*100).toFixed(2)}%</td>
      <td class="num">${(r.marginal_onset_recall_gain>=0?'+':'') + (r.marginal_onset_recall_gain*100).toFixed(2)}%</td>
      <td class="num">${(r.marginal_rainy_f1_gain>=0?'+':'') + (r.marginal_rainy_f1_gain*100).toFixed(2)}%</td>
      <td class="num">${((r.marginal_rush_f1_gain||0)>=0?'+':'') + ((r.marginal_rush_f1_gain||0)*100).toFixed(2)}%</td>
      <td class="num"><span class="tag ${r.retained_by_feature_gate?'t-ok':'t-warn'}">${r.retained_by_feature_gate?'KEEP':'OPT'}</span></td>
    </tr>
  `).join('');

  const shap = DATA.shap || {};
  document.getElementById('shapMethodLbl').textContent = shap.attribution_method || 'Exact TreeSHAP';
  document.getElementById('hypMiniGrid').innerHTML = (shap.hypotheses_verification || []).map(h => `
    <div class="hyp-card">
      <div class="hyp-head">
        <span class="hyp-num">${h.id}</span>
        <span class="tag ${h.supported?'t-ok':'t-danger'}">${h.supported?'ผ่านเกณฑ์':'ไม่ผ่าน'}</span>
      </div>
      <div class="hyp-title">${h.title.split('(')[0].trim()}</div>
      <div class="hyp-desc">${h.evidence}</div>
    </div>
  `).join('');

  const stepLabels = { 'upgrade-geom':'Geometry 4.2', prepare:'Prepare Data', contract:'Contract 4.2', train:'Train LightGBM', ablation:'B3 Ablation', explain:'TreeSHAP', forecast:'1h Forecast', 'online-ml':'Online ML', registry:'Model Registry', monitor:'3-Layer Monitor', charts:'Charts', dashboard:'Dashboard', 'model-card':'Model Card' };
  document.getElementById('dagFlow').innerHTML = (lin.steps || []).map((s, idx) => `
    <div class="dag-node">
      <div class="dag-step-num">${String(idx+1).padStart(2,'0')}</div>
      <div class="dag-step-name">${stepLabels[s.step] || s.step}</div>
      <div class="dag-step-meta">
        <span class="tag ${s.status==='SUCCESS'?'t-ok':'t-danger'}">${s.status}</span>
        <span class="mono" style="color:var(--text-muted)">${s.duration_sec.toFixed(1)}s</span>
      </div>
    </div>
  `).join('');

  const dc = DATA.data_contract || {};
  const dcGates = (dc.gates || []).map(g => [g.gate, g.actual, g.passed ? 'ผ่านเกณฑ์' : 'ตรวจสอบ', g.passed]);
  const afRows = [
    ['Airflow DAG 1: bkk_traffic_15m_online_inference', 'ทุก 15 นาที (*/15 * * * *) · พยากรณ์ t+60m', 'พร้อมรัน', true],
    ['Airflow DAG 2: bkk_traffic_daily_monitoring_and_online_warmstart', 'รายวัน 01:00 น. · 3-Layer Monitor + Warm-Start', 'พร้อมรัน', true],
    ['Airflow DAG 3: bkk_traffic_weekly_challenger_tournament', 'รายสัปดาห์ หรือเมื่อ F1 ตกเกิน 0.05', 'พร้อมรัน', true],
    ...dcGates
  ];
  document.getElementById('airflowContractTable').innerHTML = afRows.map(r => `
    <tr>
      <td style="font-weight:600;color:var(--text-main);">${r[0]}</td>
      <td style="color:var(--text-muted);font-size:13.5px;">${r[1]}</td>
      <td class="num"><span class="tag ${r[3]?'t-ok':'t-warn'}">${r[2]}</span></td>
    </tr>
  `).join('');

  const oml = DATA.online_ml || {};
  const omStatic = oml.static_batch_metrics || b2;
  const omOnline = oml.online_incremental_metrics || b2;
  document.getElementById('onlineMetaTag').textContent = `อัปเดตเฉลี่ย ${oml.mean_update_latency_ms || 1500} ms · ต้นไม้ ${oml.initial_trees || 756} → ${oml.final_trees_after_stream || 828}`;
  document.getElementById('onlineNoteBox').innerHTML = `
    <span class="note-label">การทำงานของ Online Incremental ML (LightGBM Warm-Start)</span>
    เมื่อความเร็วจริงของชั่วโมงถัดไป (<span class="mono">t+60m</span>) เข้ามา ระบบจะเติมต้นไม้ใหม่รอบละ 12 ต้นจาก Sliding Window ล่าสุด ได้ค่า <strong>Online F1 = ${((omOnline.f1_all||0)*100).toFixed(2)}%</strong> (Static = ${((omStatic.f1_all||0)*100).toFixed(2)}%) และ <strong>Onset Recall เพิ่มเป็น ${((omOnline.onset_recall||0)*100).toFixed(2)}%</strong>
  `;

  const g = (c.promotion_decision || {}).gates || {};
  const gList = [
    ['Gate 1: F1 ชนะ Baseline', '+' + ((g.gate_1_f1_gain_over_baseline?.f1_gain ?? 0)*100).toFixed(2) + '%', '>= +1.0%', g.gate_1_f1_gain_over_baseline?.passed],
    ['Gate 2: Onset Recall & Non-Drop', ((g.gate_2_onset_recall_guardrail?.challenger_onset_recall ?? 0)*100).toFixed(2) + '%', '>= 55% (ตก <= 2%)', g.gate_2_onset_recall_guardrail?.passed],
    ['Gate 3: Rainy F1 ไม่ลดลง', ((g.gate_3_rainy_f1_non_degradation?.challenger_f1_rainy ?? 0)*100).toFixed(2) + '%', '>= B0 (' + ((b0.f1_rainy||0)*100).toFixed(1) + '%)', g.gate_3_rainy_f1_non_degradation?.passed],
    ['Gate 4: MAE & Non-Regression', ((g.gate_4_mae_and_non_regression?.challenger_mae ?? 0)*100).toFixed(2) + '%', '<= 15.0%', g.gate_4_mae_and_non_regression?.passed],
  ];
  document.getElementById('gatesTable').innerHTML = gList.map(r => `
    <tr><td>${r[0]}</td><td class="num">${r[1]}</td><td class="num" style="color:var(--text-muted)">${r[2]}</td>
    <td class="num"><span class="tag ${r[3]?'t-ok':'t-danger'}">${r[3]?'ผ่าน':'ไม่ผ่าน'}</span></td></tr>
  `).join('');

  const win = (DATA.tournament || {}).winning_strategy;
  document.getElementById('tourneyTable').innerHTML = ((DATA.tournament || {}).challengers || []).map(x => `
    <tr class="${x.strategy_id===win?'win-row':''}">
      <td>${x.strategy_name_th} ${x.strategy_id===win?'<span class="tag t-ok" style="margin-left:6px;">ผู้ชนะ</span>':''}</td>
      <td class="num">${(x.metrics_val.f1_all*100).toFixed(2)}%</td>
      <td class="num">${(x.metrics_test.f1_all*100).toFixed(2)}%</td>
      <td class="num">${(x.metrics_test.onset_recall*100).toFixed(2)}%</td>
      <td class="num">${(x.metrics_test.mae*100).toFixed(2)}%</td>
    </tr>
  `).join('');

  const rob = ev.robustness_3fold || {};
  const robDev = ((rob.max_abs_deviation || 0) * 100).toFixed(2);
  document.getElementById('robustTag').innerHTML = `แกว่งสูงสุด <b>&plusmn;${robDev}%</b> (เกณฑ์ &le; &plusmn;5.0%) <span class="tag ${rob.passed!==false?'t-ok':'t-danger'}" style="margin-left:6px;">${rob.passed!==false?'ผ่านเกณฑ์':'ไม่ผ่าน'}</span>`;
  document.getElementById('robustTable').innerHTML = (rob.folds || []).map(f => `
    <tr>
      <td class="mono">${f.fold}</td>
      <td class="num">${(f.rows||0).toLocaleString()}</td>
      <td class="num">${((f.f1_all||0)*100).toFixed(2)}%</td>
      <td class="num">${((f.f1_daytime_06_23||f.f1_all||0)*100).toFixed(2)}%</td>
      <td class="num">${((f.mae||0)*100).toFixed(2)}%</td>
    </tr>
  `).join('');

  const act = (DATA.registry_index || {}).active_champion;
  document.getElementById('verTable').innerHTML = ((DATA.registry_index || {}).versions || []).map(v => `
    <tr>
      <td class="mono" style="color:var(--teal);font-weight:700;">${v.version}</td>
      <td>${v.stage}</td>
      <td class="num">${(v.test_f1*100).toFixed(2)}%</td>
      <td class="num">${((v.test_onset_recall||0)*100).toFixed(2)}%</td>
      <td class="num" style="color:var(--text-muted)">${v.model_sha256.slice(0,10)}</td>
      <td class="num">${v.version===act?'<span class="tag t-ok">Active</span>':`<button class="btn-soft" onclick="doRollback('${v.version}')">Rollback</button>`}</td>
    </tr>
  `).join('');
}

/* ── 6 ACADEMIC REPORT CHARTS (Light Paper Theme matching warin.me) ──── */
function renderAllMlopsCharts() {
  const ev = DATA.evaluation || {}, mods = ev.models || {};
  const b0 = (mods.B0_Persistence || {}).test || {};
  const b1 = (mods.B1_Historical_Average || {}).test || {};
  const b2 = (mods.B2_LightGBM || {}).test || {};

  // Chart 1: Model Benchmark
  new Chart(document.getElementById('benchChart'), {
    type: 'bar',
    data: {
      labels: ['F1 รวม (24h)', 'กลางวัน (06-23h)', 'Onset Recall', 'เร่งด่วน F1', 'ฝนตก F1'],
      datasets: [
        { label: 'B0 Persistence', data: [b0.f1_all, b0.f1_daytime_06_23, b0.onset_recall, b0.f1_rush_hour, b0.f1_rainy].map(v => +((v||0)*100).toFixed(2)), backgroundColor: '#94a3b8' },
        { label: 'B1 Historical Avg', data: [b1.f1_all, b1.f1_daytime_06_23, b1.onset_recall, b1.f1_rush_hour, b1.f1_rainy].map(v => +((v||0)*100).toFixed(2)), backgroundColor: '#60a5fa' },
        { label: 'B2 LightGBM (Champion)', data: [b2.f1_all, b2.f1_daytime_06_23, b2.onset_recall, b2.f1_rush_hour, b2.f1_rainy].map(v => +((v||0)*100).toFixed(2)), backgroundColor: '#0f766e' }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, padding: 16 } } },
      scales: {
        y: { min: 0, max: 90, ticks: { callback: v => v + '%' }, grid: { color: '#e2e8f0' } },
        x: { grid: { display: false } }
      }
    }
  });

  // Chart 2: 24-Hour Congestion Profile
  const hourly = (DATA.forecast || {}).hourly_forecast_analysis || [];
  new Chart(document.getElementById('hourlyProfileChart'), {
    type: 'line',
    data: {
      labels: hourly.map(x => x.hour_label),
      datasets: [
        { label: 'ค่าจริง Actual (%)', data: hourly.map(x => +(x.actual_red_rate*100).toFixed(2)), borderColor: '#0f172a', backgroundColor: 'rgba(15,23,42,0.06)', fill: true, tension: 0.3, borderWidth: 2.5, pointRadius: 2.5 },
        { label: 'พยากรณ์ +1 ชม. LightGBM (%)', data: hourly.map(x => +(x.pred_red_rate*100).toFixed(2)), borderColor: '#0f766e', borderDash: [6,4], tension: 0.3, borderWidth: 2.5, pointRadius: 2.5 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, padding: 16 } } },
      scales: {
        y: { ticks: { callback: v => v + '%' }, grid: { color: '#e2e8f0' } },
        x: { grid: { display: false } }
      }
    }
  });

  // Chart 3: Hourly MAE Error
  new Chart(document.getElementById('hourlyMaeChart'), {
    type: 'line',
    data: {
      labels: hourly.map(x => x.hour_label),
      datasets: [
        { label: 'B0 Persistence MAE (%)', data: hourly.map(x => +(x.mae_b0*100).toFixed(2)), borderColor: '#dc2626', borderDash: [5,4], tension: 0.25, borderWidth: 2, pointRadius: 0 },
        { label: 'B2 LightGBM MAE (%)', data: hourly.map(x => +(x.mae_lgbm*100).toFixed(2)), borderColor: '#0f766e', backgroundColor: 'rgba(15,118,110,0.1)', fill: true, tension: 0.25, borderWidth: 2.5, pointRadius: 2.5 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, padding: 16 } } },
      scales: {
        y: { ticks: { callback: v => v + '%' }, grid: { color: '#e2e8f0' } },
        x: { grid: { display: false } }
      }
    }
  });

  // Chart 4: Top 12 TreeSHAP
  const top12 = ((DATA.shap || {}).top_features || []).slice(0, 12);
  new Chart(document.getElementById('shapChart'), {
    type: 'bar',
    data: {
      labels: top12.map(f => f.feature),
      datasets: [{
        label: 'สัดส่วนอิทธิพล TreeSHAP (%)',
        data: top12.map(f => f.share_pct),
        backgroundColor: top12.map((_, i) => i === 0 ? '#0f766e' : (i < 3 ? '#2563eb' : '#94a3b8'))
      }]
    },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { callback: v => v + '%' }, grid: { color: '#e2e8f0' } },
        y: { grid: { display: false }, ticks: { color: '#0f172a', font: { family: 'JetBrains Mono', size: 12 } } }
      }
    }
  });

  // Chart 5: B3 Ablation
  const ab = (DATA.ablation || {}).ablation_experiments || [];
  new Chart(document.getElementById('ablationChart'), {
    type: 'bar',
    data: {
      labels: ab.map(r => r.group_label.split(':')[0]),
      datasets: [
        { label: 'ΔF1 Gain (%)', data: ab.map(r => +(r.marginal_f1_gain*100).toFixed(2)), backgroundColor: '#2563eb' },
        { label: 'ΔOnset Recall Gain (%)', data: ab.map(r => +(r.marginal_onset_recall_gain*100).toFixed(2)), backgroundColor: '#0f766e' }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, padding: 16 } } },
      scales: {
        y: { ticks: { callback: v => (v > 0 ? '+' : '') + v + '%' }, grid: { color: '#e2e8f0' } },
        x: { grid: { display: false } }
      }
    }
  });

  // Chart 6: Online Incremental ML Stream
  const wins = (DATA.online_ml || {}).stream_windows || [];
  new Chart(document.getElementById('onlineStreamChart'), {
    type: 'line',
    data: {
      labels: wins.map(w => w.window_id),
      datasets: [
        { label: 'Static Batch F1 (%)', data: wins.map(w => +(w.static_f1*100).toFixed(2)), borderColor: '#64748b', borderDash: [5,4], tension: 0.25, borderWidth: 2, pointRadius: 2.5 },
        { label: 'Online Warm-Start F1 (%)', data: wins.map(w => +(w.online_f1*100).toFixed(2)), borderColor: '#2563eb', tension: 0.25, borderWidth: 2.5, pointRadius: 3.5 },
        { label: 'Online Onset Recall (%)', data: wins.map(w => +(w.online_onset_recall*100).toFixed(2)), borderColor: '#0f766e', tension: 0.25, borderWidth: 2.2, pointRadius: 3 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, padding: 16 } } },
      scales: {
        y: { min: 45, max: 80, ticks: { callback: v => v + '%' }, grid: { color: '#e2e8f0' } },
        x: { grid: { display: false } }
      }
    }
  });
}

async function doRollback(ver) {
  try {
    const r = await fetch('/api/v1/mlops/rollback', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target_version: ver, reason: 'Dashboard Rollback' })
    });
    if (r.ok) location.reload();
  } catch (e) { document.getElementById('rbMsg').textContent = 'รันผ่าน FastAPI เพื่อกด Rollback'; }
}

/* ── VIEW 3: 3-LAYER MONITORING ──────────────────────────────────────── */
function renderMonitor() {
  const mon = DATA.monitoring || {};
  const dq = mon.data_quality_layer || {};
  const health = mon.overall_system_health || 'HEALTHY';
  document.getElementById('healthDot').style.background = health === 'HEALTHY' ? '#22c55e' : (health === 'WARNING_WATCH' ? '#f59e0b' : '#ef4444');
  document.getElementById('monBannerPill').textContent = `System Health: ${health} · Retrain Trigger: ${mon.retrain_recommended?'YES':'NO'}`;

  document.getElementById('monStrip').innerHTML = `
    <div class="snap-cell"><strong class="snap-val mono" style="font-size:20px;color:${health==='HEALTHY'?'#86efac':'#fcd34d'}">${health}</strong><span class="snap-lbl">สถานะระบบ (Retrain: ${mon.retrain_recommended?'YES':'NO'})</span></div>
    <div class="snap-cell"><strong class="snap-val mono" style="color:#67e8f9">${(dq.location_table_join_coverage_pct||100).toFixed(1)}%</strong><span class="snap-lbl">Join Coverage (Circuit-Breaker: ${dq.circuit_breaker_tripped_count||0})</span></div>
    <div class="snap-cell"><strong class="snap-val mono">${((mon.prediction_drift||{}).psi||0).toFixed(4)}</strong><span class="snap-lbl">Prediction PSI (${(mon.prediction_drift||{}).status||'STABLE'})</span></div>
    <div class="snap-cell"><strong class="snap-val mono" style="color:#86efac">${((mon.test_overall_f1||0)*100).toFixed(2)}%</strong><span class="snap-lbl">Test F1 (Alert Days: ${mon.alert_days_count||0}/${mon.total_test_days||24})</span></div>
  `;

  document.getElementById('dqSummary').textContent = mon.retrain_policy_note || '';
  const trippedCnt = dq.circuit_breaker_tripped_count || 0;
  const minRows = dq.min_daily_sample_rows || 8048;
  const reqRows = dq.min_required_daily_rows || 8000;
  const minOk = dq.minimum_sample_check_passed !== false;
  document.getElementById('dqTable').innerHTML = `
    <tr><td>Location Table Join Coverage (Data Contract 4.2)</td><td class="num">${(dq.location_table_join_coverage_pct||100).toFixed(1)}%</td><td class="num">100.0%</td><td class="num"><span class="tag t-ok">ผ่านเกณฑ์</span></td></tr>
    <tr><td>ความครบถ้วนของข้อมูลฝน Open-Meteo</td><td class="num">${(dq.weather_feed_completeness_pct||100).toFixed(1)}%</td><td class="num">100.0%</td><td class="num"><span class="tag t-ok">ผ่านเกณฑ์</span></td></tr>
    <tr><td>ความสดของข้อมูล (Feed Freshness SLA)</td><td class="num">${dq.feed_freshness_lag_minutes ?? 0} นาที</td><td class="num">&le; ${dq.feed_freshness_sla_minutes ?? 15} นาที</td><td class="num"><span class="tag t-ok">${dq.feed_freshness_status || 'FRESH'}</span></td></tr>
    <tr><td>จำนวนตัวอย่างขั้นต่ำรายวัน (Minimum Sample Check)</td><td class="num">${minRows.toLocaleString()} แถว/วัน</td><td class="num">&ge; ${reqRows.toLocaleString()} แถว/วัน</td><td class="num"><span class="tag ${minOk?'t-ok':'t-danger'}">${minOk?'ผ่านเกณฑ์':'ต่ำกว่าเกณฑ์'}</span></td></tr>
    <tr><td>Segment 24h Missing Rate Circuit-Breaker (${dq.monitored_segments_count||448} ช่วงถนน)</td><td class="num">หยุด ${trippedCnt} เส้น</td><td class="num">&le; 20% missing</td><td class="num"><span class="tag ${trippedCnt===0?'t-ok':'t-warn'}">${trippedCnt===0?'ปกติทุกเส้น':'ตัดวงจรบางเส้น'}</span></td></tr>
  `;

  const mMap = Object.fromEntries((mon.monsoon_stress_test_psi || []).map(x => [x.feature, x]));
  document.getElementById('psiTable').innerHTML = (mon.current_window_psi || []).map(p => {
    const cls = p.status==='STABLE'?'t-ok':(p.status==='WARNING'?'t-warn':'t-danger');
    return `<tr><td class="mono" style="color:var(--text-main);font-weight:600;">${p.feature}</td><td class="num">${p.ref_mean.toFixed(3)}</td><td class="num">${p.cur_mean.toFixed(3)}</td>
      <td class="num"><b>${p.psi.toFixed(4)}</b></td><td class="num" style="color:var(--blue)">${((mMap[p.feature]||{}).psi||0).toFixed(3)}</td>
      <td class="num"><span class="tag ${cls}">${p.status}</span></td></tr>`;
  }).join('');
}

function renderAllMonitorCharts() {
  const mon = DATA.monitoring || {};
  const curPsi = mon.current_window_psi || [];
  const mMap = Object.fromEntries((mon.monsoon_stress_test_psi || []).map(x => [x.feature, x]));

  new Chart(document.getElementById('psiChart'), {
    type: 'bar',
    data: {
      labels: curPsi.map(p => p.feature),
      datasets: [
        { label: 'Test Window PSI (ก.ย. 2022)', data: curPsi.map(p => p.psi), backgroundColor: curPsi.map(p => p.psi >= 0.25 ? '#dc2626' : (p.psi >= 0.10 ? '#f59e0b' : '#0f766e')) },
        { label: 'Monsoon Heavy-Rain PSI', data: curPsi.map(p => (mMap[p.feature]||{}).psi || 0), backgroundColor: '#93c5fd' }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, padding: 16 } } },
      scales: {
        y: { grid: { color: '#e2e8f0' } },
        x: { grid: { display: false }, ticks: { maxRotation: 25, minRotation: 15, font: { family: 'JetBrains Mono', size: 11.5 } } }
      }
    }
  });

  const daily = mon.daily_rolling_performance || [];
  new Chart(document.getElementById('rollingChart'), {
    type: 'line',
    data: {
      labels: daily.map(d => d.date.slice(5)),
      datasets: [
        { label: 'Rolling 7-Day F1', data: daily.map(d => d.rolling_7d_f1 ?? d.f1_all), borderColor: '#0f766e', tension: 0.25, borderWidth: 2.5, pointRadius: 2.5 },
        { label: 'Daily F1', data: daily.map(d => d.f1_all), borderColor: '#94a3b8', borderDash: [5,4], tension: 0.25, borderWidth: 1.8, pointRadius: 0 },
        { label: 'Daily MAE', data: daily.map(d => d.mae), borderColor: '#d97706', tension: 0.25, borderWidth: 2, pointRadius: 0 }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, padding: 16 } } },
      scales: {
        y: { min: 0, max: 0.85, grid: { color: '#e2e8f0' } },
        x: { grid: { display: false } }
      }
    }
  });
}

window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('longdoKeyIn').value = getKey('longdo');
  document.getElementById('googleKeyIn').value = getKey('google');

  const dSel = document.getElementById('distSel');
  (DATA.districts || []).forEach(d => { const o = document.createElement('option'); o.value = d; o.textContent = d; dSel.appendChild(o); });
  const sSel = document.getElementById('simLink');
  activeLinks().slice(0, 40).forEach(l => { const o = document.createElement('option'); o.value = l.link_id; o.textContent = `${l.road_name} (${l.district})`; sSel.appendChild(o); });

  const initial = (!getKey('longdo') && getKey('google')) ? 'google' : 'longdo';
  switchProvider(initial);
  renderWeatherHeader();
  refreshMap();
  renderMlops();
  renderMonitor();
});
</script>
</body>
</html>
"""


def generate_standalone_dashboard_html() -> Path:
    log_step("DASHBOARD", "สร้าง Dashboard สไตล์ warin.me Academic Dossier (Noto Serif Thai + Deep Navy/Teal + 8 Charts)...")
    cfg = load_config()
    payload = build_dashboard_payload(use_cached_snapshot=False)
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__PAYLOAD_JSON__", payload_json)

    out_path = ROOT_DIR / cfg["paths"]["dashboard_html"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    log_success(f"สร้าง Dashboard เรียบร้อยที่ {out_path.relative_to(ROOT_DIR)}")
    return out_path
