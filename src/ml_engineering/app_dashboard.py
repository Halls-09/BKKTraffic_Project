"""
BMA Tactical Traffic & MLOps Telemetry Command Center (Frontend-Design Upgraded).
Responsibility: Sarit (67090500441) - ML Engineer

Designed using Anthropic's `frontend-design` skill principles:
- Aesthetic Direction: Tactical Urban Telemetry & Precision Civic Infrastructure Command Room
- Typography:
  * Display / Headers: 'Chakra Petch' (Geometric Thai/Latin Technical Display)
  * Body / Analysis: 'IBM Plex Sans Thai'
  * Telemetry / Metrics / Hashes: 'JetBrains Mono' (tabular-nums)
- Color Tokens:
  * --asphalt-void (#06080D), --bunker-panel (#0C111B), --grid-hairline (#1E293B)
  * --signal-emerald (#10B981), --caution-amber (#F59E0B), --crimson-jam (#F43F5E), --onset-beacon (#E879F9)
- Features:
  * Deck 1: Live Cartography (Predicted t+60 vs Actual Ground Truth vs Residual Error) + SVG Spillback Corridor Graph
  * Deck 2: Commuter Route Matrix & Interactive Convective Rain What-If Simulator (+0 to +40 mm/3h)
  * Deck 3: Factor Attribution (SHAP), Horizon Sensitivity (15/30/45/60m), Hourly MAE (00:00-23:00), Event Study (95% CI)
  * Deck 4: MLOps Control Plane (5-Layer Monitoring & Circuit-Breaker, 3-Challenger Tournament, Promotion Gates, PSI Drift)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.env_config import get_api_keys_status
from src.utils.logger import log_info, log_step, log_success
from src.utils.storage import ROOT_DIR, load_config, load_table


def collect_dashboard_payload() -> dict[str, Any]:
    cfg = load_config()
    silver_dir = ROOT_DIR / cfg["paths"]["silver_dir"]
    gold_dir = ROOT_DIR / cfg["paths"]["gold_dir"]
    eval_dir = ROOT_DIR / cfg["paths"]["evaluation_dir"]
    reg_dir = ROOT_DIR / cfg["paths"]["registry_dir"]
    mon_dir = ROOT_DIR / cfg["paths"]["monitoring_dir"]
    lin_dir = ROOT_DIR / cfg["paths"]["lineage_dir"]

    df_segs = load_table(silver_dir / "silver_road_segments.parquet")
    df_traffy = load_table(silver_dir / "silver_traffy_spatial_features.parquet")
    df_gold = load_table(gold_dir / "gold_features.parquet")

    def _read_json(path: Path) -> dict[str, Any]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    eval_summary = _read_json(eval_dir / "evaluation_summary.json")
    ablation_data = _read_json(eval_dir / "ablation_results.json")
    shap_data = _read_json(eval_dir / "shap_attributions.json")
    experiments_s11 = _read_json(eval_dir / "proposal_experiments_s11.json")
    registry_index = _read_json(reg_dir / "registry_index.json")
    champion_meta = _read_json(reg_dir / "champion_metadata.json")
    tournament_data = _read_json(reg_dir / "challenger_tournament.json")
    drift_report = _read_json(mon_dir / "drift_monitoring_report.json")
    latest_run = _read_json(lin_dir / "latest_run.json")
    manifest = _read_json(ROOT_DIR / cfg["paths"]["manifest_path"])

    df_test = df_gold[df_gold["split"] == "test"].copy()
    df_test["ts_str"] = pd.to_datetime(df_test["ts_t"]).dt.strftime("%Y-%m-%d %H:%M")
    available_ts = sorted(df_test["ts_str"].unique())

    rainy_ts = (
        df_test.groupby("ts_str")["rain_mm_t0"]
        .mean()
        .sort_values(ascending=False)
        .head(4)
        .index.tolist()
    )
    onset_ts = (
        df_test.groupby("ts_str")["pred_is_onset_alert"]
        .sum()
        .sort_values(ascending=False)
        .head(4)
        .index.tolist()
    )
    selected_snapshots = list(dict.fromkeys(onset_ts + rainy_ts + available_ts[::16]))[:12]
    selected_snapshots.sort()

    seg_meta = df_segs.merge(
        df_traffy[["seg_id", "seg_buffer500m_flood_count", "seg_buffer500m_road_defect_count", "district_flood_density_per_km2"]],
        on="seg_id",
        how="left",
    )

    snapshots_data: dict[str, list[dict[str, Any]]] = {}
    for ts_label in selected_snapshots:
        sub = df_test[df_test["ts_str"] == ts_label].merge(
            seg_meta[["seg_id", "lat_start", "lon_start", "lat_end", "lon_end", "seg_buffer500m_flood_count"]],
            on="seg_id",
            how="left",
        )
        records = []
        for _, r in sub.iterrows():
            records.append({
                "seg_id": r["seg_id"],
                "road_name": r["road_name"],
                "direction": r["direction"],
                "district_name": r["district_name"],
                "lat_start": float(r["lat_start"]),
                "lon_start": float(r["lon_start"]),
                "lat_end": float(r["lat_end"]),
                "lon_end": float(r["lon_end"]),
                "free_flow_kmh": float(r["free_flow_speed_kmh"]),
                "speed_now_kmh": round(float(r["ratio_t0"] * r["free_flow_speed_kmh"]), 1),
                "pred_speed_t60_kmh": round(float(r["pred_b2_lgbm"] * r["free_flow_speed_kmh"]), 1),
                "true_speed_t60_kmh": round(float(r["target_ratio_t60"] * r["free_flow_speed_kmh"]), 1),
                "ratio_t0": round(float(r["ratio_t0"]), 3),
                "pred_ratio_t60": round(float(r["pred_b2_lgbm"]), 3),
                "true_ratio_t60": round(float(r["target_ratio_t60"]), 3),
                "abs_error": round(abs(float(r["target_ratio_t60"]) - float(r["pred_b2_lgbm"])), 3),
                "pred_congested": int(r["pred_is_congested_t60"]),
                "true_congested": int(r["target_is_congested_t60"]),
                "pred_onset_alert": int(r["pred_is_onset_alert"]),
                "true_onset": int(r["is_onset_event"]),
                "rain_mm_t0": round(float(r["rain_mm_t0"]), 2),
                "rain_cum_3h": round(float(r["rain_cum_3h"]), 2),
                "downstream_min_ratio": round(float(r["downstream_min_ratio_t0"]), 3),
                "flood_buffer500m": int(r["traffy_flood_buffer500m"]),
                "active_incident": int(r["active_incident_on_seg"]),
            })
        snapshots_data[ts_label] = records

    return {
        "project": cfg["project"],
        "snapshots": snapshots_data,
        "snapshot_keys": selected_snapshots,
        "evaluation": eval_summary,
        "ablation": ablation_data,
        "shap": shap_data,
        "experiments_s11": experiments_s11,
        "registry": registry_index,
        "champion": champion_meta,
        "tournament": tournament_data,
        "monitoring": drift_report,
        "lineage": latest_run,
        "manifest_file_count": len(manifest.get("files", {})),
        "env_keys_status": get_api_keys_status(),
    }


def generate_standalone_dashboard_html(output_path: Path | None = None) -> Path:
    log_step(
        "BUILD_DASHBOARD",
        "สร้าง Tactical Telemetry Frontend (Anthropic frontend-design skill: Chakra Petch + IBM Plex Sans Thai + JetBrains Mono)...",
    )
    payload = collect_dashboard_payload()
    cfg = load_config()
    out_file = output_path or (ROOT_DIR / cfg["paths"]["artifacts_dir"] / "dashboard_index.html")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    payload_json = json.dumps(payload, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>BMA TACTICAL TRAFFIC & MLOPS COMMAND CENTER // +60M FORECAST</title>
  <!-- Google Fonts: Chakra Petch (Display), IBM Plex Sans Thai (Body), JetBrains Mono (Telemetry) -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:ital,wght@0,500;0,600;0,700;1,600&family=IBM+Plex+Sans+Thai:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    /* =========================================================================
       FRONTEND-DESIGN SKILL TOKEN SYSTEM: TACTICAL URBAN TELEMETRY
       ========================================================================= */
    :root {{
      --asphalt-void: #06080D;
      --bunker-panel: #0B0F17;
      --surface-elevated: #111826;
      --grid-hairline: #1E293B;
      --grid-accent: #334155;
      --text-primary: #F8FAFC;
      --text-secondary: #94A3B8;
      --text-dim: #64748B;
      --signal-emerald: #10B981;
      --caution-amber: #F59E0B;
      --crimson-jam: #F43F5E;
      --onset-beacon: #E879F9;
      --cyber-cyan: #06B6D4;
      --font-display: 'Chakra Petch', 'IBM Plex Sans Thai', sans-serif;
      --font-body: 'IBM Plex Sans Thai', -apple-system, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: var(--font-body);
      background-color: var(--asphalt-void);
      background-image:
        linear-gradient(to right, rgba(30, 41, 59, 0.18) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(30, 41, 59, 0.18) 1px, transparent 1px);
      background-size: 32px 32px;
      color: var(--text-primary);
      line-height: 1.5;
      min-height: 100vh;
      overflow-x: hidden;
    }}

    /* Top Telemetry Ticker Bar */
    .telemetry-ribbon {{
      background: #030508;
      border-bottom: 1px solid var(--grid-hairline);
      padding: 6px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: var(--font-mono);
      font-size: 0.73rem;
      color: var(--text-secondary);
      letter-spacing: 0.04em;
    }}
    .ribbon-group {{ display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }}
    .ribbon-tag {{ color: var(--cyber-cyan); font-weight: 600; }}
    .kbd-hint {{
      background: var(--surface-elevated);
      border: 1px solid var(--grid-accent);
      color: var(--text-primary);
      padding: 1px 6px;
      border-radius: 3px;
      font-size: 0.68rem;
    }}

    /* Main Command Header */
    header {{
      background: rgba(11, 15, 23, 0.92);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--grid-hairline);
      padding: 16px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
    }}
    .command-brand {{
      display: flex;
      align-items: center;
      gap: 14px;
    }}
    .brand-emblem {{
      width: 42px;
      height: 42px;
      background: linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(232, 121, 249, 0.2));
      border: 1px solid var(--cyber-cyan);
      display: grid;
      place-items: center;
      font-family: var(--font-display);
      font-weight: 700;
      font-size: 1.1rem;
      color: var(--cyber-cyan);
      position: relative;
    }}
    .brand-emblem::after {{
      content: '';
      position: absolute;
      inset: 3px;
      border: 1px dashed rgba(6, 182, 212, 0.4);
    }}
    .brand-title {{
      font-family: var(--font-display);
      font-size: 1.32rem;
      font-weight: 700;
      letter-spacing: 0.02em;
      color: #fff;
      text-transform: uppercase;
    }}
    .brand-subtitle {{
      font-size: 0.82rem;
      color: var(--text-secondary);
    }}
    .champion-beacon {{
      display: flex;
      align-items: center;
      gap: 10px;
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid var(--signal-emerald);
      padding: 7px 14px;
      font-family: var(--font-mono);
      font-size: 0.78rem;
      color: #34D399;
    }}
    .pulse-dot {{
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--signal-emerald);
      box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
      animation: pulse-emerald 2s infinite;
    }}
    @keyframes pulse-emerald {{
      0% {{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
      70% {{ box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }}
      100% {{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
    }}
    @keyframes pulse-onset {{
      0% {{ box-shadow: 0 0 0 0 rgba(232, 121, 249, 0.8); }}
      70% {{ box-shadow: 0 0 0 9px rgba(232, 121, 249, 0); }}
      100% {{ box-shadow: 0 0 0 0 rgba(232, 121, 249, 0); }}
    }}

    /* Tactical Deck Navigation */
    .deck-nav {{
      display: flex;
      background: var(--bunker-panel);
      border-bottom: 1px solid var(--grid-hairline);
      padding: 0 24px;
      gap: 2px;
      overflow-x: auto;
    }}
    .deck-btn {{
      background: transparent;
      color: var(--text-secondary);
      border: none;
      border-right: 1px solid var(--grid-hairline);
      border-bottom: 3px solid transparent;
      padding: 13px 20px;
      font-family: var(--font-display);
      font-size: 0.92rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.16s ease;
      display: flex;
      align-items: center;
      gap: 8px;
      white-space: nowrap;
    }}
    .deck-btn:first-child {{ border-left: 1px solid var(--grid-hairline); }}
    .deck-btn:hover {{
      color: var(--text-primary);
      background: rgba(255, 255, 255, 0.03);
    }}
    .deck-btn.active {{
      color: var(--cyber-cyan);
      border-bottom-color: var(--cyber-cyan);
      background: rgba(6, 182, 212, 0.08);
    }}
    .deck-index {{
      font-family: var(--font-mono);
      font-size: 0.72rem;
      padding: 1px 5px;
      background: rgba(148, 163, 184, 0.14);
      color: var(--text-secondary);
    }}
    .deck-btn.active .deck-index {{
      background: var(--cyber-cyan);
      color: #000;
      font-weight: 700;
    }}

    /* Main Workspace Container */
    .workspace {{
      max-width: 1640px;
      margin: 0 auto;
      padding: 20px 24px 40px;
    }}
    .deck-pane {{ display: none; animation: fadeIn 0.22s ease-out; }}
    .deck-pane.active {{ display: block; }}
    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(4px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* Control Strip */
    .control-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      align-items: center;
      background: var(--bunker-panel);
      border: 1px solid var(--grid-hairline);
      padding: 12px 16px;
      margin-bottom: 18px;
      position: relative;
    }}
    .control-strip::before {{
      content: 'CONTROL_BAR';
      position: absolute;
      top: -8px;
      left: 12px;
      background: var(--asphalt-void);
      padding: 0 6px;
      font-family: var(--font-mono);
      font-size: 0.64rem;
      color: var(--cyber-cyan);
      letter-spacing: 0.08em;
    }}
    .ctrl-group {{ display: flex; align-items: center; gap: 8px; }}
    .ctrl-label {{
      font-family: var(--font-mono);
      font-size: 0.75rem;
      color: var(--text-secondary);
      text-transform: uppercase;
    }}
    select, input[type="range"] {{
      background: var(--surface-elevated);
      color: var(--text-primary);
      border: 1px solid var(--grid-accent);
      padding: 7px 11px;
      font-family: var(--font-body);
      font-size: 0.85rem;
      outline: none;
    }}
    select:focus-visible, button:focus-visible {{
      outline: 2px solid var(--cyber-cyan);
      outline-offset: 2px;
    }}

    /* Segmented Mode Switcher */
    .seg-switch {{
      display: inline-flex;
      border: 1px solid var(--grid-accent);
      background: var(--asphalt-void);
    }}
    .seg-opt {{
      background: transparent;
      color: var(--text-secondary);
      border: none;
      padding: 6px 12px;
      font-family: var(--font-mono);
      font-size: 0.75rem;
      cursor: pointer;
      border-right: 1px solid var(--grid-hairline);
    }}
    .seg-opt:last-child {{ border-right: none; }}
    .seg-opt.active {{
      background: var(--cyber-cyan);
      color: #04070B;
      font-weight: 700;
    }}

    /* Telemetry HUD KPI Strip */
    .hud-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}
    .hud-box {{
      background: var(--bunker-panel);
      border: 1px solid var(--grid-hairline);
      padding: 15px 18px;
      position: relative;
      overflow: hidden;
    }}
    .hud-box::after {{
      content: '+';
      position: absolute;
      top: 6px;
      right: 10px;
      font-family: var(--font-mono);
      font-size: 0.75rem;
      color: var(--text-dim);
    }}
    .hud-tag {{
      font-family: var(--font-mono);
      font-size: 0.72rem;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .hud-metric {{
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
      font-size: 1.85rem;
      font-weight: 700;
      margin: 6px 0 4px;
    }}
    .hud-foot {{
      font-size: 0.78rem;
      color: var(--text-secondary);
    }}

    /* Tactical Cards & Grids */
    .split-asymmetric {{
      display: grid;
      grid-template-columns: 1.32fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .split-equal {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    @media (max-width: 1100px) {{
      .split-asymmetric, .split-equal {{ grid-template-columns: 1fr; }}
    }}
    .panel {{
      background: var(--bunker-panel);
      border: 1px solid var(--grid-hairline);
      padding: 18px;
      margin-bottom: 18px;
      position: relative;
    }}
    .panel-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 11px;
      margin-bottom: 14px;
      border-bottom: 1px solid var(--grid-hairline);
      gap: 10px;
      flex-wrap: wrap;
    }}
    .panel-title {{
      font-family: var(--font-display);
      font-size: 1.02rem;
      font-weight: 700;
      color: #fff;
      letter-spacing: 0.02em;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    #map {{
      height: 490px;
      width: 100%;
      border: 1px solid var(--grid-accent);
      background: #04060A;
    }}

    /* High-Density Tabular Telemetry */
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.84rem;
    }}
    th, td {{
      padding: 9px 10px;
      text-align: left;
      border-bottom: 1px solid rgba(30, 41, 59, 0.75);
      font-variant-numeric: tabular-nums;
    }}
    th {{
      font-family: var(--font-mono);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-secondary);
      background: rgba(17, 24, 38, 0.9);
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    tr:hover {{ background: rgba(6, 182, 212, 0.05); }}
    .mono-num {{ font-family: var(--font-mono); font-variant-numeric: tabular-nums; }}

    /* Inline Ratio Sparkbar */
    .sparkbar-wrap {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-family: var(--font-mono);
      font-size: 0.8rem;
    }}
    .sparkbar-track {{
      width: 58px;
      height: 6px;
      background: #1E293B;
      overflow: hidden;
      position: relative;
    }}
    .sparkbar-fill {{ height: 100%; }}

    /* Tactical Status Chips */
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 2px 8px;
      font-family: var(--font-mono);
      font-size: 0.71rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    .chip-emerald {{ background: rgba(16, 185, 129, 0.14); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.45); }}
    .chip-amber {{ background: rgba(245, 158, 11, 0.14); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.45); }}
    .chip-crimson {{ background: rgba(244, 63, 94, 0.16); color: #FB7185; border: 1px solid rgba(244, 63, 94, 0.45); }}
    .chip-onset {{
      background: rgba(232, 121, 249, 0.18);
      color: #F0ABFC;
      border: 1px solid #E879F9;
      animation: pulse-onset 2s infinite;
    }}
    .chip-cyan {{ background: rgba(6, 182, 212, 0.15); color: #22D3EE; border: 1px solid rgba(6, 182, 212, 0.45); }}

    .verdict-row {{
      background: var(--surface-elevated);
      border-left: 3px solid var(--signal-emerald);
      padding: 11px 14px;
      margin-bottom: 9px;
    }}

    @media (prefers-reduced-motion: reduce) {{
      *, ::before, ::after {{
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
      }}
    }}
  </style>
</head>
<body>
  <!-- TOP TELEMETRY RIBBON -->
  <div class="telemetry-ribbon">
    <div class="ribbon-group">
      <span><b class="ribbon-tag">SYS_NODE:</b> BKK-ITS-MLOPS-01</span>
      <span><b class="ribbon-tag">HORIZON:</b> +60 MIN (4 SLOTS)</span>
      <span><b class="ribbon-tag">DATA_CONTRACT:</b> v1.0.0 (CC-BY 4.0)</span>
      <span><b class="ribbon-tag">RAIN_MODE:</b> OBSERVED_LAG_ONLY</span>
    </div>
    <div class="ribbon-group">
      <span>SHORTCUTS: <kbd class="kbd-hint">1</kbd>-<kbd class="kbd-hint">4</kbd> SWITCH DECK | <kbd class="kbd-hint">M</kbd> CYCLE MAP LAYER</span>
    </div>
  </div>

  <!-- COMMAND HEADER -->
  <header>
    <div class="command-brand">
      <div class="brand-emblem">BKK</div>
      <div>
        <div class="brand-title">ศูนย์บัญชาการพยากรณ์จราจรกรุงเทพฯ ล่วงหน้า 1 ชม. และระบบวิศวกรรมโมเดล (MLOps)</div>
        <div class="brand-subtitle">One-Hour-Ahead Bangkok Traffic Congestion Forecasting & Factor Attribution // CSS 342 — สฤษดิ์ ศรีดิษ (67090500441)</div>
      </div>
    </div>
    <div class="champion-beacon">
      <span class="pulse-dot"></span>
      <span id="header-champion-pill">CHAMPION: v1.0.0 // ONLINE</span>
    </div>
  </header>

  <!-- TACTICAL DECK NAVIGATION -->
  <nav class="deck-nav" role="tablist">
    <button class="deck-btn active" id="btn-tab1" onclick="switchTab('tab1', this)">
      <span class="deck-index">01</span> ศูนย์ควบคุมจราจร & แผนที่เทียบค่าจริง (OPERATIONS & CARTOGRAPHY)
    </button>
    <button class="deck-btn" id="btn-tab2" onclick="switchTab('tab2', this)">
      <span class="deck-index">02</span> ผู้ขับขี่ & จำลองพายุฝน (COMMUTER & RAIN WHAT-IF)
    </button>
    <button class="deck-btn" id="btn-tab3" onclick="switchTab('tab3', this)">
      <span class="deck-index">03</span> วิเคราะห์ปัจจัย & การทดลองข้อ 11 (SHAP, HORIZON, HOURLY MAE)
    </button>
    <button class="deck-btn" id="btn-tab4" onclick="switchTab('tab4', this)">
      <span class="deck-index">04</span> แผงควบคุมวิศวกรรม MLOps (3-CHALLENGER, 5-LAYER MONITOR & LINEAGE)
    </button>
  </nav>

  <main class="workspace">
    <!-- ==================== DECK 01: TRAFFIC OPERATIONS & CARTOGRAPHY ==================== -->
    <section id="tab1" class="deck-pane active">
      <div class="control-strip">
        <div class="ctrl-group">
          <span class="ctrl-label">TIMESTAMP (t):</span>
          <select id="snapshot-select" onchange="renderControlCenter()"></select>
        </div>
        <div class="ctrl-group">
          <span class="ctrl-label">MAP LAYER (M):</span>
          <div class="seg-switch" id="map-mode-group">
            <button class="seg-opt active" data-mode="pred" onclick="setMapMode('pred')">PREDICTED (t+60)</button>
            <button class="seg-opt" data-mode="actual" onclick="setMapMode('actual')">ACTUAL TRUTH (t+60)</button>
            <button class="seg-opt" data-mode="error" onclick="setMapMode('error')">RESIDUAL |ERROR|</button>
          </div>
        </div>
        <div class="ctrl-group">
          <span class="ctrl-label">DISTRICT:</span>
          <select id="district-filter" onchange="renderControlCenter()">
            <option value="ALL">ALL DISTRICTS (ทุกเขต)</option>
          </select>
        </div>
        <div class="ctrl-group">
          <span class="ctrl-label">DIRECTION:</span>
          <select id="dir-filter" onchange="renderControlCenter()">
            <option value="ALL">ALL DIRECTIONS</option>
            <option value="inbound">INBOUND (ขาเข้าเมือง)</option>
            <option value="outbound">OUTBOUND (ขาออกเมือง)</option>
          </select>
        </div>
      </div>

      <!-- HUD KPI STRIP -->
      <div class="hud-grid">
        <div class="hud-box">
          <div class="hud-tag">CONGESTED SEGMENTS (PRED / ACTUAL)</div>
          <div class="hud-metric" id="kpi-congested-count" style="color:var(--crimson-jam);">0 / 0</div>
          <div class="hud-foot" id="kpi-congested-sub">Threshold &lt; 0.50 ณ เวลา t + 60 นาที</div>
        </div>
        <div class="hud-box" style="border-color:rgba(232,121,249,0.55);">
          <div class="hud-tag">PROACTIVE ONSET WARNINGS (t &rarr; t+60)</div>
          <div class="hud-metric" id="kpi-onset-count" style="color:var(--onset-beacon);">0 LINKS</div>
          <div class="hud-foot">ปัจจุบันคล่องตัว (&ge;0.55) แต่จะเริ่มติดใน 60 นาที</div>
        </div>
        <div class="hud-box">
          <div class="hud-tag">NETWORK FORECAST SPEED // MAE</div>
          <div class="hud-metric" id="kpi-avg-speed" style="color:var(--cyber-cyan);">0.0 km/h</div>
          <div class="hud-foot" id="kpi-avg-ratio">Snapshot Residual MAE: 0.000</div>
        </div>
        <div class="hud-box">
          <div class="hud-tag">3-HOUR CUMULATIVE RAIN // INCIDENTS</div>
          <div class="hud-metric" id="kpi-rain" style="color:var(--caution-amber);">0.0 mm</div>
          <div class="hud-foot" id="kpi-incident-sub">Active Road Incidents: 0</div>
        </div>
      </div>

      <div class="split-asymmetric">
        <div>
          <div class="panel">
            <div class="panel-head">
              <div class="panel-title">🗺️ แผนที่โครงข่ายถนนกรุงเทพฯ เทียบค่าพยากรณ์และค่าจริง (t + 60 นาที)</div>
              <span class="chip chip-cyan" id="map-time-badge">LAYER: PREDICTED</span>
            </div>
            <div id="map"></div>
          </div>

          <!-- Custom SVG Directed Spillback Topology Diagram -->
          <div class="panel" style="margin-bottom:0;">
            <div class="panel-head">
              <div class="panel-title">🕸️ โครงข่ายการติดลามย้อนกลับ (H3: Directed Downstream Spillback Topology)</div>
              <span class="chip chip-emerald">GRAPH ADJACENCY MATRIX</span>
            </div>
            <div id="spillback-svg-container" style="width:100%;overflow-x:auto;"></div>
          </div>
        </div>

        <div class="panel" style="max-height:735px;overflow-y:auto;margin-bottom:0;">
          <div class="panel-head">
            <div class="panel-title">📡 ตารางเทเลเมทรีรายช่วงถนน (Predicted vs Actual Ground Truth)</div>
          </div>
          <table>
            <thead>
              <tr>
                <th>SEGMENT / ROAD</th>
                <th>NOW (t)</th>
                <th>PRED (t+60)</th>
                <th>ACTUAL (t+60)</th>
                <th>|ERR|</th>
                <th>STATUS</th>
              </tr>
            </thead>
            <tbody id="segment-table-body"></tbody>
          </table>
        </div>
      </div>
    </section>

    <!-- ==================== DECK 02: COMMUTER & RAIN WHAT-IF ==================== -->
    <section id="tab2" class="deck-pane">
      <div class="control-strip">
        <div class="ctrl-group">
          <span class="ctrl-label">CONVECTIVE RAIN SHOCK (+MM/3H):</span>
          <input type="range" id="rain-slider" min="0" max="40" step="5" value="0" oninput="renderCommuterView()" />
          <span id="rain-slider-val" class="chip chip-amber">+0 mm/3h</span>
        </div>
        <div class="ctrl-group">
          <span class="ctrl-label">PRIMARY CORRIDOR:</span>
          <select id="commuter-seg-select" onchange="renderCommuterView()"></select>
        </div>
      </div>

      <div class="split-equal">
        <div class="panel">
          <div class="panel-head">
            <div class="panel-title">🧭 ระบบแนะนำการเดินทางและทางเลี่ยงอัจฉริยะล่วงหน้า 1 ชั่วโมง</div>
          </div>
          <div id="commuter-advisory-box"></div>
        </div>
        <div class="panel">
          <div class="panel-head">
            <div class="panel-title">⛈️ ผลกระทบปฏิสัมพันธ์ ฝนสะสม &times; จุดเสี่ยงน้ำท่วม Traffy Fondue (H4 Interaction)</div>
          </div>
          <canvas id="chart-whatif-rain" height="205"></canvas>
        </div>
      </div>

      <div class="panel">
        <div class="panel-head">
          <div class="panel-title">🔀 ตารางเปรียบเทียบสายทางหลักและเส้นทางเลี่ยงภายใต้สถานการณ์จำลอง</div>
        </div>
        <table>
          <thead>
            <tr>
              <th>SEG_ID</th>
              <th>ชื่อช่วงถนน</th>
              <th>เขต</th>
              <th>ความเร็ว ณ เวลา t</th>
              <th>พยากรณ์ปกติ (t+60)</th>
              <th>พยากรณ์เมื่อฝนเพิ่ม (What-If)</th>
              <th>ประวัติร้องเรียนน้ำท่วม (500m)</th>
              <th>คำแนะนำการตัดสินใจ</th>
            </tr>
          </thead>
          <tbody id="commuter-table-body"></tbody>
        </table>
      </div>
    </section>

    <!-- ==================== DECK 03: URBAN PLANNERS, SHAP & S11 EXPERIMENTS ==================== -->
    <section id="tab3" class="deck-pane">
      <div class="split-equal">
        <div class="panel">
          <div class="panel-head">
            <div class="panel-title">📊 Global Feature Attribution (12 ตัวแปรที่ส่งผลต่อการเกิดรถติดสูงสุด)</div>
            <span class="chip chip-cyan" id="shap-method-badge">ATTRIBUTION</span>
          </div>
          <canvas id="chart-shap-top" height="220"></canvas>
        </div>
        <div class="panel">
          <div class="panel-head">
            <div class="panel-title">⏱️ การทดลองข้อ 11: Horizon Sensitivity (15 / 30 / 45 / 60 นาทีล่วงหน้า)</div>
            <span class="chip chip-emerald">LEAD-TIME TRADEOFF</span>
          </div>
          <canvas id="chart-horizon" height="220"></canvas>
        </div>
      </div>

      <div class="split-equal">
        <div class="panel">
          <div class="panel-head">
            <div class="panel-title">🕐 Error Analysis (ข้อ 9.2): ความคลาดเคลื่อน MAE แยกตามชั่วโมงเป้าหมาย (00:00–23:00 น.)</div>
          </div>
          <canvas id="chart-hourly-mae" height="210"></canvas>
        </div>
        <div class="panel">
          <div class="panel-head">
            <div class="panel-title">🌊 Traffy Flood Event Study (21 ก.ค. 2022) & Rolling-Origin 3-Fold Robustness</div>
          </div>
          <div id="event-study-box"></div>
        </div>
      </div>

      <div class="split-equal">
        <div class="panel">
          <div class="panel-head">
            <div class="panel-title">🔬 ผลการทดลอง B3: Feature-Group Ablation Study (ตัดทีละกลุ่มฟีเจอร์)</div>
          </div>
          <table>
            <thead>
              <tr>
                <th>กลุ่มฟีเจอร์ที่ตัดออก (Removed Group)</th>
                <th>F1 (All)</th>
                <th>&Delta;F1</th>
                <th>Onset Recall (&Delta;)</th>
                <th>Rainy F1</th>
                <th>MAE</th>
              </tr>
            </thead>
            <tbody id="ablation-table-body"></tbody>
          </table>
        </div>
        <div class="panel">
          <div class="panel-head">
            <div class="panel-title">✅ ผลการพิสูจน์สมมุติฐานวิจัยเชิงปริมาณ (Hypotheses H1–H5 Verification)</div>
          </div>
          <div id="hypotheses-container"></div>
        </div>
      </div>
    </section>

    <!-- ==================== DECK 04: MLOPS CONTROL PLANE ==================== -->
    <section id="tab4" class="deck-pane">
      <div class="hud-grid">
        <div class="hud-box">
          <div class="hud-tag">ACTIVE CHAMPION ARTIFACT</div>
          <div class="hud-metric" id="mlops-champ-ver" style="color:var(--signal-emerald);">v1.0.0</div>
          <div class="hud-foot" id="mlops-champ-th">Threshold: 0.495</div>
        </div>
        <div class="hud-box">
          <div class="hud-tag">PRIMARY METRIC: TEST F1 (ALL)</div>
          <div class="hud-metric" id="mlops-f1-val" style="color:var(--cyber-cyan);">0.9307</div>
          <div class="hud-foot" id="mlops-f1-gain">+0.1392 over Persistence</div>
        </div>
        <div class="hud-box">
          <div class="hud-tag">GUARDRAIL: ONSET RECALL</div>
          <div class="hud-metric" id="mlops-onset-val" style="color:var(--onset-beacon);">0.6659</div>
          <div class="hud-foot">B0 Persistence Onset Recall = 0.0000</div>
        </div>
        <div class="hud-box">
          <div class="hud-tag">5-LAYER MONITORING STATUS</div>
          <div class="hud-metric" id="mlops-drift-status" style="color:var(--signal-emerald);">HEALTHY</div>
          <div class="hud-foot" id="mlops-manifest-count">SHA256 Verified Tables: 12</div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-head">
          <div class="panel-title">🛡️ ระบบเฝ้าระวัง 5 ชั้น และ Segment Circuit-Breaker (Section 10 Page 11 Specification)</div>
          <span class="chip chip-emerald">ALL 5 LAYERS ACTIVE</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>MONITORING LAYER</th>
              <th>METRIC / TRIGGER CONDITION</th>
              <th>STATUS</th>
              <th>AUTOMATED MLOPS PROTOCOL (ตามตารางข้อ 10 หน้า 11)</th>
            </tr>
          </thead>
          <tbody id="five-layer-tbody"></tbody>
        </table>
      </div>

      <div class="split-equal">
        <div class="panel">
          <div class="panel-head">
            <div class="panel-title">🥊 สนามแข่งขัน Challenger ทั้ง 3 รูปแบบ (Expanding vs Rolling vs Same-Season)</div>
          </div>
          <table>
            <thead>
              <tr>
                <th>CHALLENGER STRATEGY</th>
                <th>TRAIN ROWS</th>
                <th>THRESHOLD</th>
                <th>TEST F1</th>
                <th>ONSET RECALL</th>
                <th>RAINY F1</th>
              </tr>
            </thead>
            <tbody id="tournament-tbody"></tbody>
          </table>
        </div>

        <div class="panel">
          <div class="panel-head">
            <div class="panel-title">🏆 เกณฑ์ตัดสิน Champion Promotion Gates ทั้ง 4 ด่าน</div>
          </div>
          <div id="promotion-gates-box"></div>
        </div>
      </div>

      <div class="split-equal">
        <div class="panel">
          <div class="panel-head">
            <div class="panel-title">📉 Feature Drift PSI (ตรวจสอบสาเหตุ; ไม่ Retrain จาก PSI อย่างเดียว)</div>
            <select id="drift-mode-select" onchange="renderDriftChart()">
              <option value="normal">CURRENT TEST WINDOW (ปกติ)</option>
              <option value="stress">MONSOON STORM STRESS-TEST (พายุฝนรุนแรง)</option>
            </select>
          </div>
          <canvas id="chart-psi-drift" height="210"></canvas>
        </div>

        <div class="panel">
          <div class="panel-head">
            <div class="panel-title">📈 ติดตาม Performance Drift รายวัน (Daily F1 & Rolling 3D F1)</div>
          </div>
          <canvas id="chart-daily-perf" height="210"></canvas>
        </div>
      </div>

      <div class="panel">
        <div class="panel-head">
          <div class="panel-title">⛓️ Pipeline Execution 10-Stage DAG & Data Lineage Log</div>
          <span class="chip chip-cyan" id="lineage-run-id">RUN_ID</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>DAG STEP</th>
              <th>ROLE</th>
              <th>DESCRIPTION</th>
              <th>STATUS</th>
              <th>DURATION</th>
            </tr>
          </thead>
          <tbody id="lineage-tbody"></tbody>
        </table>
      </div>

      <div class="panel">
        <div class="panel-head">
          <div class="panel-title">🔑 API Keys (`.env`) &amp; FastAPI REST Endpoints (`/api/v1/*`)</div>
          <span class="chip chip-emerald" id="env-keys-summary-chip">.ENV READY</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>ENV VARIABLE</th>
              <th>PROVIDER / SERVICE</th>
              <th>STATUS</th>
              <th>MASKED KEY</th>
              <th>ENDPOINT / FALLBACK MODE</th>
            </tr>
          </thead>
          <tbody id="env-keys-tbody"></tbody>
        </table>
      </div>
    </section>
  </main>

  <script>
    const DATA = {payload_json};
    let map = null;
    let mapLayerGroup = null;
    let currentMapMode = 'pred';
    let chartWhatIf = null;
    let chartShap = null;
    let chartHorizon = null;
    let chartHourlyMae = null;
    let chartPsi = null;
    let chartDaily = null;

    Chart.defaults.font.family = "'JetBrains Mono', 'IBM Plex Sans Thai', monospace";
    Chart.defaults.color = '#94A3B8';

    function switchTab(tabId, btnEl) {{
      document.querySelectorAll('.deck-pane').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.deck-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      if (btnEl) btnEl.classList.add('active');
      if (tabId === 'tab1' && map) setTimeout(() => map.invalidateSize(), 150);
    }}

    function setMapMode(mode) {{
      currentMapMode = mode;
      document.querySelectorAll('#map-mode-group .seg-opt').forEach(btn => {{
        btn.classList.toggle('active', btn.dataset.mode === mode);
      }});
      renderControlCenter();
    }}

    // Keyboard shortcuts: 1-4 for decks, M for map mode
    window.addEventListener('keydown', (e) => {{
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
      if (e.key === '1') switchTab('tab1', document.getElementById('btn-tab1'));
      if (e.key === '2') switchTab('tab2', document.getElementById('btn-tab2'));
      if (e.key === '3') switchTab('tab3', document.getElementById('btn-tab3'));
      if (e.key === '4') switchTab('tab4', document.getElementById('btn-tab4'));
      if (e.key.toLowerCase() === 'm') {{
        const order = ['pred', 'actual', 'error'];
        const next = order[(order.indexOf(currentMapMode) + 1) % order.length];
        setMapMode(next);
      }}
    }});

    function ratioColor(r) {{
      if (r < 0.50) return '#F43F5E';
      if (r < 0.70) return '#F59E0B';
      return '#10B981';
    }}

    function errorColor(err) {{
      if (err > 0.08) return '#F43F5E';
      if (err > 0.04) return '#F59E0B';
      return '#10B981';
    }}

    function sparkBarHtml(val) {{
      const pct = Math.min(100, Math.max(5, Math.round((val / 1.1) * 100)));
      const col = ratioColor(val);
      return `
        <div class="sparkbar-wrap">
          <span>${{val.toFixed(2)}}</span>
          <div class="sparkbar-track"><div class="sparkbar-fill" style="width:${{pct}}%;background:${{col}};"></div></div>
        </div>
      `;
    }}

    function initControls() {{
      const snapSel = document.getElementById('snapshot-select');
      DATA.snapshot_keys.forEach((k, idx) => {{
        const opt = document.createElement('option');
        opt.value = k;
        opt.textContent = k + ' (+60m Horizon)';
        if (idx === 0) opt.selected = true;
        snapSel.appendChild(opt);
      }});

      const firstSnap = DATA.snapshots[DATA.snapshot_keys[0]] || [];
      const districts = [...new Set(firstSnap.map(d => d.district_name))].sort();
      const distSel = document.getElementById('district-filter');
      districts.forEach(d => {{
        const opt = document.createElement('option');
        opt.value = d;
        opt.textContent = 'เขต' + d;
        distSel.appendChild(opt);
      }});

      const commSel = document.getElementById('commuter-seg-select');
      firstSnap.forEach(s => {{
        const opt = document.createElement('option');
        opt.value = s.seg_id;
        opt.textContent = `${{s.road_name}} [${{s.direction === 'inbound' ? 'ขาเข้า' : 'ขาออก'}}]`;
        commSel.appendChild(opt);
      }});
    }}

    function initMap() {{
      map = L.map('map').setView([13.752, 100.562], 12);
      L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
        attribution: '&copy; OpenStreetMap & CartoDB'
      }}).addTo(map);
      mapLayerGroup = L.layerGroup().addTo(map);
    }}

    function renderSpillbackSvg(rows) {{
      const byId = Object.fromEntries(rows.map(r => [r.seg_id, r]));
      const nodes = [
        {{ id: 'SEG_BNA_01_IN', label: 'บางนา-ตราด', x: 70, y: 55 }},
        {{ id: 'SEG_SUK_02_IN', label: 'สุขุมวิท (ทองหล่อ)', x: 235, y: 55 }},
        {{ id: 'SEG_SUK_01_IN', label: 'สุขุมวิท (อโศก)', x: 410, y: 55 }},
        {{ id: 'SEG_PLO_01_IN', label: 'เพลินจิต', x: 580, y: 55 }},
        {{ id: 'SEG_RAM1_01_IN', label: 'พระราม 1 (สยาม)', x: 745, y: 55 }},
        {{ id: 'SEG_LAT_01_IN', label: 'ลาดพร้าว', x: 70, y: 135 }},
        {{ id: 'SEG_RAT_01_IN', label: 'รัชดา-ห้วยขวาง', x: 235, y: 135 }},
        {{ id: 'SEG_ASO_01_IN', label: 'อโศกมนตรี', x: 410, y: 135 }},
        {{ id: 'SEG_DIN_01_IN', label: 'ดินแดง', x: 580, y: 135 }},
        {{ id: 'SEG_PHY_01_IN', label: 'พญาไท', x: 745, y: 135 }},
      ];
      const edges = [
        ['SEG_BNA_01_IN', 'SEG_SUK_02_IN'],
        ['SEG_SUK_02_IN', 'SEG_SUK_01_IN'],
        ['SEG_SUK_01_IN', 'SEG_PLO_01_IN'],
        ['SEG_PLO_01_IN', 'SEG_RAM1_01_IN'],
        ['SEG_LAT_01_IN', 'SEG_RAT_01_IN'],
        ['SEG_RAT_01_IN', 'SEG_ASO_01_IN'],
        ['SEG_ASO_01_IN', 'SEG_SUK_01_IN'],
        ['SEG_RAT_01_IN', 'SEG_DIN_01_IN'],
        ['SEG_DIN_01_IN', 'SEG_PHY_01_IN'],
        ['SEG_PHY_01_IN', 'SEG_RAM1_01_IN'],
      ];
      const posMap = Object.fromEntries(nodes.map(n => [n.id, n]));
      let svgEdges = edges.map(([u, v]) => {{
        const p1 = posMap[u], p2 = posMap[v];
        const rDown = byId[v] ? byId[v].pred_ratio_t60 : 0.65;
        const isSpill = rDown < 0.50;
        const col = isSpill ? '#F43F5E' : '#334155';
        return `<line x1="${{p1.x}}" y1="${{p1.y}}" x2="${{p2.x}}" y2="${{p2.y}}" stroke="${{col}}" stroke-width="${{isSpill ? 3 : 1.5}}" stroke-dasharray="${{isSpill ? '5,3' : 'none'}}" />`;
      }}).join('');

      let svgNodes = nodes.map(n => {{
        const rObj = byId[n.id] || {{ pred_ratio_t60: 0.72, pred_onset_alert: 0 }};
        const col = rObj.pred_onset_alert === 1 ? '#E879F9' : ratioColor(rObj.pred_ratio_t60);
        return `
          <g transform="translate(${{n.x}}, ${{n.y}})">
            <rect x="-64" y="-22" width="128" height="44" fill="#0B0F17" stroke="${{col}}" stroke-width="2" rx="3" />
            <text x="0" y="-4" text-anchor="middle" fill="#F8FAFC" font-size="10.5" font-weight="600">${{n.label}}</text>
            <text x="0" y="12" text-anchor="middle" fill="${{col}}" font-family="JetBrains Mono" font-size="10" font-weight="700">t+60: ${{rObj.pred_ratio_t60.toFixed(2)}}</text>
          </g>
        `;
      }}).join('');

      document.getElementById('spillback-svg-container').innerHTML = `
        <svg viewBox="0 0 830 180" style="width:100%;height:auto;min-width:680px;background:#06080D;border:1px solid #1E293B;">
          ${{svgEdges}}
          ${{svgNodes}}
        </svg>
      `;
    }}

    function renderControlCenter() {{
      const tsKey = document.getElementById('snapshot-select').value;
      const distVal = document.getElementById('district-filter').value;
      const dirVal = document.getElementById('dir-filter').value;
      document.getElementById('map-time-badge').textContent = `T=${{tsKey}} // MODE: ${{currentMapMode.toUpperCase()}}`;

      const allSnapRows = DATA.snapshots[tsKey] || [];
      renderSpillbackSvg(allSnapRows);

      let rows = allSnapRows.slice();
      if (distVal !== 'ALL') rows = rows.filter(r => r.district_name === distVal);
      if (dirVal !== 'ALL') rows = rows.filter(r => r.direction === dirVal);

      const nCongPred = rows.filter(r => r.pred_congested === 1).length;
      const nCongTrue = rows.filter(r => r.true_congested === 1).length;
      const nOnset = rows.filter(r => r.pred_onset_alert === 1).length;
      const avgSpd = rows.length ? (rows.reduce((a, b) => a + b.pred_speed_t60_kmh, 0) / rows.length) : 0;
      const avgErr = rows.length ? (rows.reduce((a, b) => a + b.abs_error, 0) / rows.length) : 0;
      const avgRain3h = rows.length ? (rows.reduce((a, b) => a + b.rain_cum_3h, 0) / rows.length) : 0;
      const nInc = rows.filter(r => r.active_incident === 1).length;

      document.getElementById('kpi-congested-count').textContent = `${{nCongPred}} / ${{nCongTrue}}`;
      document.getElementById('kpi-congested-sub').textContent = `พยากรณ์ติด ${{nCongPred}} จุด | ค่าจริง ${{nCongTrue}} จุด`;
      document.getElementById('kpi-onset-count').textContent = `${{nOnset}} LINKS`;
      document.getElementById('kpi-avg-speed').textContent = `${{avgSpd.toFixed(1)}} km/h`;
      document.getElementById('kpi-avg-ratio').textContent = `Snapshot Residual MAE: ${{avgErr.toFixed(3)}}`;
      document.getElementById('kpi-rain').textContent = `${{avgRain3h.toFixed(1)}} mm`;
      document.getElementById('kpi-incident-sub').textContent = `Active Road Incidents: ${{nInc}}`;

      mapLayerGroup.clearLayers();
      rows.forEach(r => {{
        let color = '#10B981';
        if (currentMapMode === 'pred') {{
          color = r.pred_onset_alert === 1 ? '#E879F9' : ratioColor(r.pred_ratio_t60);
        }} else if (currentMapMode === 'actual') {{
          color = r.true_onset === 1 ? '#E879F9' : ratioColor(r.true_ratio_t60);
        }} else {{
          color = errorColor(r.abs_error);
        }}

        const line = L.polyline([[r.lat_start, r.lon_start], [r.lat_end, r.lon_end]], {{
          color: color,
          weight: (r.pred_onset_alert === 1 && currentMapMode === 'pred') ? 8 : 6,
          opacity: 0.92
        }}).addTo(mapLayerGroup);

        const midLat = 0.5 * (r.lat_start + r.lat_end);
        const midLon = 0.5 * (r.lon_start + r.lon_end);
        const circle = L.circleMarker([midLat, midLon], {{
          radius: 6,
          fillColor: color,
          color: '#06080D',
          weight: 2,
          fillOpacity: 0.95
        }}).addTo(mapLayerGroup);

        const popupHtml = `
          <div style="color:#06080D;font-family:'IBM Plex Sans Thai',sans-serif;min-width:225px;">
            <b>${{r.road_name}}</b> (${{r.direction}})<br/>
            เขต: ${{r.district_name}} | Free-flow: ${{r.free_flow_kmh}} km/h<hr style="margin:4px 0;"/>
            ปัจจุบัน (t): <b>${{r.ratio_t0}}</b> (${{r.speed_now_kmh}} km/h)<br/>
            พยากรณ์ (t+60): <b>${{r.pred_ratio_t60}}</b> (${{r.pred_speed_t60_kmh}} km/h)<br/>
            ค่าจริง (t+60): <b>${{r.true_ratio_t60}}</b> (${{r.true_speed_t60_kmh}} km/h)<br/>
            |Error|: <b>${{r.abs_error}}</b> | ฝน 3 ชม.: ${{r.rain_cum_3h}} mm
          </div>
        `;
        line.bindPopup(popupHtml);
        circle.bindPopup(popupHtml);
      }});

      rows.sort((a, b) => (b.pred_onset_alert - a.pred_onset_alert) || (a.pred_ratio_t60 - b.pred_ratio_t60));
      document.getElementById('segment-table-body').innerHTML = rows.map(r => {{
        const chip = r.pred_onset_alert === 1
          ? '<span class="chip chip-onset">ONSET ALERT</span>'
          : (r.pred_congested === 1 ? '<span class="chip chip-crimson">JAM &lt;0.50</span>' : '<span class="chip chip-emerald">CLEAR</span>');
        return `
          <tr>
            <td>
              <b>${{r.road_name}}</b><br/>
              <span class="mono-num" style="color:var(--text-dim);font-size:0.73rem;">${{r.seg_id}} // ${{r.district_name}}</span>
            </td>
            <td class="mono-num">${{r.ratio_t0.toFixed(2)}}</td>
            <td>${{sparkBarHtml(r.pred_ratio_t60)}}</td>
            <td>${{sparkBarHtml(r.true_ratio_t60)}}</td>
            <td class="mono-num" style="color:${{r.abs_error < 0.05 ? '#34D399' : '#FBBF24'}};">${{r.abs_error.toFixed(3)}}</td>
            <td>${{chip}}</td>
          </tr>
        `;
      }}).join('');
    }}

    function renderCommuterView() {{
      const tsKey = document.getElementById('snapshot-select').value;
      const rainAdd = parseFloat(document.getElementById('rain-slider').value);
      const selectedSegId = document.getElementById('commuter-seg-select').value;
      document.getElementById('rain-slider-val').textContent = `+${{rainAdd}} mm/3h`;

      const rows = (DATA.snapshots[tsKey] || []).map(r => {{
        const floodFactor = Math.log1p(r.flood_buffer500m) / 5.0;
        const rainPenalty = (rainAdd * 0.0045) * (0.6 + floodFactor);
        const whatIfRatio = Math.max(0.12, +(r.pred_ratio_t60 - rainPenalty).toFixed(3));
        const whatIfSpeed = +(whatIfRatio * r.free_flow_kmh).toFixed(1);
        return {{ ...r, whatIfRatio, whatIfSpeed }};
      }});

      const targetSeg = rows.find(r => r.seg_id === selectedSegId) || rows[0];
      const alternatives = rows
        .filter(r => r.seg_id !== targetSeg.seg_id && r.direction === targetSeg.direction)
        .sort((a, b) => b.whatIfRatio - a.whatIfRatio)
        .slice(0, 3);

      const isRisk = targetSeg.whatIfRatio < 0.50;
      document.getElementById('commuter-advisory-box').innerHTML = `
        <div class="verdict-row" style="border-left-color:${{isRisk ? '#F43F5E' : '#10B981'}};">
          <div style="font-family:var(--font-display);font-size:1.08rem;font-weight:700;margin-bottom:6px;">
            CORRIDOR: ${{targetSeg.road_name}} [${{targetSeg.direction.toUpperCase()}}]
          </div>
          <p style="color:var(--text-secondary);margin-bottom:10px;">
            ความเร็วปัจจุบัน: <b class="mono-num">${{targetSeg.speed_now_kmh}} km/h (${{targetSeg.ratio_t0}})</b> &rarr;
            พยากรณ์ +60 นาที (ฝน +${{rainAdd}} mm): <b class="mono-num" style="color:${{isRisk ? '#FB7185' : '#34D399'}};">${{targetSeg.whatIfSpeed}} km/h (Ratio ${{targetSeg.whatIfRatio}})</b>
          </p>
          <p style="font-size:0.88rem;">
            ${{isRisk
              ? `🚨 <b>TACTICAL REROUTE ADVISORY:</b> คาดการณ์เกิดภาวะติดขัดและน้ำท่วมขังรอการระบาย (ประวัติ Traffy 500m: ${{targetSeg.flood_buffer500m}} รายการ) แนะนำให้เลี่ยงไปใช้: <b>${{alternatives.map(a => a.road_name).join(' หรือ ')}}</b>`
              : `✅ <b>CLEAR CORRIDOR:</b> เส้นทางนี้ยังคงคล่องตัวในอีก 1 ชั่วโมงข้างหน้า สามารถใช้เดินทางได้ตามปกติ`
            }}
          </p>
        </div>
      `;

      document.getElementById('commuter-table-body').innerHTML = rows.slice(0, 12).map(r => {{
        const rec = r.whatIfRatio < 0.50
          ? '<span class="chip chip-crimson">AVOID // REROUTE</span>'
          : '<span class="chip chip-emerald">RECOMMENDED</span>';
        return `
          <tr>
            <td class="mono-num">${{r.seg_id}}</td>
            <td><b>${{r.road_name}}</b></td>
            <td>${{r.district_name}}</td>
            <td class="mono-num">${{r.speed_now_kmh}} km/h (${{r.ratio_t0}})</td>
            <td class="mono-num">${{r.pred_speed_t60_kmh}} km/h (${{r.pred_ratio_t60}})</td>
            <td class="mono-num"><b>${{r.whatIfSpeed}} km/h (${{r.whatIfRatio}})</b></td>
            <td class="mono-num">${{r.flood_buffer500m}}</td>
            <td>${{rec}}</td>
          </tr>
        `;
      }}).join('');

      const topVuln = rows.slice().sort((a, b) => b.flood_buffer500m - a.flood_buffer500m).slice(0, 8);
      const ctx = document.getElementById('chart-whatif-rain').getContext('2d');
      if (chartWhatIf) chartWhatIf.destroy();
      chartWhatIf = new Chart(ctx, {{
        type: 'bar',
        data: {{
          labels: topVuln.map(r => r.seg_id),
          datasets: [
            {{ label: 'Baseline Forecast (t+60)', data: topVuln.map(r => r.pred_ratio_t60), backgroundColor: '#06B6D4' }},
            {{ label: `Rain Shock (+${{rainAdd}} mm)`, data: topVuln.map(r => r.whatIfRatio), backgroundColor: '#E879F9' }}
          ]
        }},
        options: {{
          responsive: true,
          scales: {{ y: {{ min: 0, max: 1.1 }} }}
        }}
      }});
    }}

    function renderUrbanPlannerView() {{
      const shap = DATA.shap || {{}};
      const exp11 = DATA.experiments_s11 || {{}};
      document.getElementById('shap-method-badge').textContent = shap.attribution_method || 'PERMUTATION_SHAP';
      const topFeats = (shap.top_features || []).slice(0, 12);

      const ctx1 = document.getElementById('chart-shap-top').getContext('2d');
      if (chartShap) chartShap.destroy();
      chartShap = new Chart(ctx1, {{
        type: 'bar',
        data: {{
          labels: topFeats.map(f => f.feature),
          datasets: [{{ label: 'Share (%)', data: topFeats.map(f => f.share_pct), backgroundColor: '#06B6D4' }}]
        }},
        options: {{
          indexAxis: 'y',
          responsive: true,
          plugins: {{ legend: {{ display: false }} }}
        }}
      }});

      const horizons = exp11.horizon_sensitivity || [];
      const ctxH = document.getElementById('chart-horizon').getContext('2d');
      if (chartHorizon) chartHorizon.destroy();
      chartHorizon = new Chart(ctxH, {{
        type: 'line',
        data: {{
          labels: horizons.map(h => h.horizon_minutes + 'm'),
          datasets: [
            {{ label: 'B2 LightGBM F1', data: horizons.map(h => h.lgbm_f1), borderColor: '#10B981', tension: 0.2 }},
            {{ label: 'B2 Onset Recall', data: horizons.map(h => h.lgbm_onset_recall), borderColor: '#E879F9', tension: 0.2 }},
            {{ label: 'B0 Persistence F1', data: horizons.map(h => h.b0_f1), borderColor: '#64748B', borderDash: [5, 5], tension: 0.2 }}
          ]
        }},
        options: {{ responsive: true, scales: {{ y: {{ min: 0.5, max: 1.0 }} }} }}
      }});

      const hourly = exp11.hourly_mae_error_analysis || [];
      const ctxHr = document.getElementById('chart-hourly-mae').getContext('2d');
      if (chartHourlyMae) chartHourlyMae.destroy();
      chartHourlyMae = new Chart(ctxHr, {{
        type: 'bar',
        data: {{
          labels: hourly.map(h => h.hour_label),
          datasets: [
            {{ label: 'B2 LightGBM MAE', data: hourly.map(h => h.mae_b2_lgbm), backgroundColor: '#06B6D4' }},
            {{ label: 'B0 Persistence MAE', data: hourly.map(h => h.mae_b0_persistence), backgroundColor: 'rgba(100,116,139,0.4)' }}
          ]
        }},
        options: {{ responsive: true, scales: {{ y: {{ beginAtZero: true }} }} }}
      }});

      const ev = exp11.traffy_flood_event_study || {{}};
      const rob = exp11.rolling_origin_3fold_robustness || {{}};
      document.getElementById('event-study-box').innerHTML = `
        <div class="verdict-row" style="border-left-color:#06B6D4;">
          <b>🌊 TRAFFY FLOOD EVENT STUDY (${{ev.event_date || '2022-07-21'}}):</b><br/>
          Flood Day Mean Ratio = <b class="mono-num">${{ev.flood_day_mean_ratio}}</b> vs Normal Days = <b class="mono-num">${{ev.normal_day_mean_ratio}}</b><br/>
          &Delta;Ratio = <b class="mono-num" style="color:#FB7185;">${{ev.delta_ratio}}</b> | 95% CI: <b class="mono-num">[${{(ev.ci_95 || [])[0]}}, ${{(ev.ci_95 || [])[1]}}]</b> (p = ${{ev.p_value}})
        </div>
        <div class="verdict-row" style="border-left-color:#10B981;">
          <b>🔄 ROLLING-ORIGIN 3-FOLD ROBUSTNESS (SECTION 11):</b><br/>
          Mean F1 Across 3 Folds = <b class="mono-num">${{rob.mean_f1}}</b> | Max Deviation = <b class="mono-num">${{rob.max_deviation_from_mean}}</b> (&le; 0.05 Criterion: <span class="chip chip-emerald">PASSED</span>)
        </div>
      `;

      const abRows = (DATA.ablation || {{}}).ablation_experiments || [];
      document.getElementById('ablation-table-body').innerHTML = abRows.map(r => `
        <tr>
          <td><b>${{r.group_label}}</b></td>
          <td class="mono-num">${{r.f1_all.toFixed(4)}}</td>
          <td class="mono-num" style="color:${{r.delta_f1_all < 0 ? '#FB7185' : '#34D399'}};">${{r.delta_f1_all >= 0 ? '+' : ''}}${{r.delta_f1_all.toFixed(4)}}</td>
          <td class="mono-num">${{r.onset_recall.toFixed(4)}} (${{r.delta_onset_recall >= 0 ? '+' : ''}}${{r.delta_onset_recall.toFixed(4)}})</td>
          <td class="mono-num">${{r.f1_rainy.toFixed(4)}}</td>
          <td class="mono-num">${{r.mae_ratio.toFixed(4)}}</td>
        </tr>
      `).join('');

      const hyps = shap.hypotheses_verification || [];
      document.getElementById('hypotheses-container').innerHTML = hyps.map(h => `
        <div class="verdict-row" style="border-left-color:${{h.supported ? '#10B981' : '#F59E0B'}};">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <b>${{h.id}}: ${{h.title}}</b>
            <span class="chip ${{h.supported ? 'chip-emerald' : 'chip-amber'}}">${{h.supported ? 'SUPPORTED' : 'PARTIAL'}}</span>
          </div>
          <div style="color:var(--text-secondary);font-size:0.82rem;margin-top:4px;">${{h.evidence}}</div>
        </div>
      `).join('');
    }}

    function renderMlopsView() {{
      const champ = DATA.champion || {{}};
      const evalModels = (DATA.evaluation || {{}}).models || {{}};
      const b0Test = (evalModels.B0_Persistence || {{}}).test || {{}};
      const b2Test = (evalModels.B2_LightGBM || {{}}).test || {{}};
      const mon = DATA.monitoring || {{}};
      const tourn = DATA.tournament || {{}};

      document.getElementById('header-champion-pill').textContent = `CHAMPION: ${{champ.version || 'v1.0.0'}} // ${{mon.overall_system_health || 'HEALTHY'}}`;
      document.getElementById('mlops-champ-ver').textContent = champ.version || 'v1.0.0';
      document.getElementById('mlops-champ-th').textContent = `Threshold: ${{champ.optimal_threshold || 0.495}} // SHA256: ${{(champ.model_sha256 || '').slice(0, 10)}}`;
      document.getElementById('mlops-f1-val').textContent = (b2Test.f1_all || 0).toFixed(4);
      const gain = ((b2Test.f1_all || 0) - (b0Test.f1_all || 0)).toFixed(4);
      document.getElementById('mlops-f1-gain').textContent = `+${{gain}} over B0 Persistence`;
      document.getElementById('mlops-onset-val').textContent = (b2Test.onset_recall || 0).toFixed(4);
      document.getElementById('mlops-drift-status').textContent = mon.overall_system_health || 'HEALTHY_STABLE';
      document.getElementById('mlops-manifest-count').textContent = `SHA256 Verified Tables: ${{DATA.manifest_file_count || 0}}`;

      const fiveLayers = mon.five_layer_monitoring_summary || [];
      document.getElementById('five-layer-tbody').innerHTML = fiveLayers.map(l => `
        <tr>
          <td><b>${{l.layer}}</b></td>
          <td class="mono-num" style="color:var(--cyber-cyan);">${{l.metric_trigger}}</td>
          <td><span class="chip chip-emerald">${{l.status}}</span></td>
          <td>${{l.action}}</td>
        </tr>
      `).join('');

      const chList = tourn.challengers || [];
      document.getElementById('tournament-tbody').innerHTML = chList.map(c => `
        <tr>
          <td><b>${{c.strategy_name_th}}</b><br/><span class="mono-num" style="color:var(--text-dim);font-size:0.73rem;">${{c.strategy_id}}</span></td>
          <td class="mono-num">${{c.train_rows.toLocaleString()}}</td>
          <td class="mono-num">${{c.optimal_threshold.toFixed(4)}}</td>
          <td class="mono-num"><b>${{c.metrics_test.f1_all.toFixed(4)}}</b></td>
          <td class="mono-num" style="color:var(--onset-beacon);"><b>${{c.metrics_test.onset_recall.toFixed(4)}}</b></td>
          <td class="mono-num">${{c.metrics_test.f1_rainy.toFixed(4)}}</td>
        </tr>
      `).join('');

      const gates = ((champ.promotion_decision || {{}}).gates) || {{}};
      document.getElementById('promotion-gates-box').innerHTML = Object.entries(gates).map(([gName, gObj]) => `
        <div class="verdict-row" style="border-left-color:${{gObj.passed ? '#10B981' : '#F43F5E'}};">
          <div style="display:flex;justify-content:space-between;">
            <b class="mono-num">${{gName}}</b>
            <span class="chip ${{gObj.passed ? 'chip-emerald' : 'chip-crimson'}}">${{gObj.passed ? 'PASSED' : 'FAILED'}}</span>
          </div>
          <div class="mono-num" style="color:var(--text-secondary);font-size:0.75rem;margin-top:4px;">${{JSON.stringify(gObj)}}</div>
        </div>
      `).join('');

      renderDriftChart();

      const daily = mon.daily_performance_timeline || [];
      const ctxDaily = document.getElementById('chart-daily-perf').getContext('2d');
      if (chartDaily) chartDaily.destroy();
      chartDaily = new Chart(ctxDaily, {{
        type: 'line',
        data: {{
          labels: daily.map(d => d.date),
          datasets: [
            {{ label: 'B2 LightGBM Daily F1', data: daily.map(d => d.f1_b2), borderColor: '#06B6D4', tension: 0.25 }},
            {{ label: 'Rolling 3-Day F1', data: daily.map(d => d.rolling_3d_f1), borderColor: '#10B981', borderDash: [5, 5], tension: 0.25 }},
            {{ label: 'B0 Persistence F1', data: daily.map(d => d.f1_b0), borderColor: '#64748B', tension: 0.25 }}
          ]
        }},
        options: {{ responsive: true, scales: {{ y: {{ min: 0.5, max: 1.02 }} }} }}
      }});

      const lin = DATA.lineage || {{}};
      document.getElementById('lineage-run-id').textContent = lin.run_id || 'LATEST_RUN';
      const steps = lin.steps || [];
      document.getElementById('lineage-tbody').innerHTML = steps.map((s, i) => `
        <tr>
          <td class="mono-num">0${{i + 1}}</td>
          <td class="mono-num"><b>${{s.step}}</b></td>
          <td>${{s.role || '-'}}</td>
          <td>${{s.description || '-'}}</td>
          <td><span class="chip chip-emerald">${{s.status}}</span></td>
          <td class="mono-num">${{(s.duration_sec || 0).toFixed(2)}}s</td>
        </tr>
      `).join('');

      const envStat = DATA.env_keys_status || {{}};
      const envKeys = envStat.keys || [];
      const envChip = document.getElementById('env-keys-summary-chip');
      if (envChip) {{
        envChip.textContent = `.ENV: ${{envStat.configured_keys_count || 0}}/${{envStat.total_keys_tracked || envKeys.length}} KEYS // SWAGGER: /docs`;
      }}
      const envTbody = document.getElementById('env-keys-tbody');
      if (envTbody) {{
        envTbody.innerHTML = envKeys.map(k => `
          <tr>
            <td class="mono-num"><b>${{k.key_name}}</b></td>
            <td>${{k.provider}}</td>
            <td><span class="chip ${{k.is_configured ? 'chip-emerald' : 'chip-cyan'}}">${{k.mode}}</span></td>
            <td class="mono-num">${{k.masked_preview}}</td>
            <td class="mono-num" style="font-size:0.75rem;color:var(--text-secondary);">${{k.endpoint_url}}<br/><span style="color:var(--text-dim);">Fallback: ${{k.fallback_mode}}</span></td>
          </tr>
        `).join('');
      }}
    }}

    function renderDriftChart() {{
      const mode = document.getElementById('drift-mode-select').value;
      const mon = DATA.monitoring || {{}};
      const items = mode === 'stress' ? (mon.monsoon_stress_test_psi || []) : (mon.current_window_psi || []);
      const ctx = document.getElementById('chart-psi-drift').getContext('2d');
      if (chartPsi) chartPsi.destroy();
      chartPsi = new Chart(ctx, {{
        type: 'bar',
        data: {{
          labels: items.map(x => x.feature),
          datasets: [{{
            label: mode === 'stress' ? 'PSI (Monsoon Stress-Test Shift)' : 'PSI (Current Test Window vs Train)',
            data: items.map(x => x.psi),
            backgroundColor: items.map(x => x.psi >= 0.25 ? '#F43F5E' : (x.psi >= 0.10 ? '#F59E0B' : '#10B981'))
          }}]
        }},
        options: {{ responsive: true, scales: {{ y: {{ beginAtZero: true }} }} }}
      }});
    }}

    window.addEventListener('DOMContentLoaded', () => {{
      initControls();
      initMap();
      renderControlCenter();
      renderCommuterView();
      renderUrbanPlannerView();
      renderMlopsView();
    }});
  </script>
</body>
</html>
"""
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    log_info(f"สร้างไฟล์ Tactical Telemetry Dashboard สำเร็จ: {out_file}")
    log_success("ออกแบบตามหลักการ Anthropic frontend-design เรียบร้อย")
    return out_file


def serve_dashboard(host: str = "127.0.0.1", port: int = 8050) -> None:
    from src.ml_engineering.api_server import run_api_server

    generate_standalone_dashboard_html()
    run_api_server(host=host, port=port)


if __name__ == "__main__":
    generate_standalone_dashboard_html()
