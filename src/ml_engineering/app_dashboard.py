"""
Interactive Bangkok Traffic Forecasting & MLOps Dashboard.
Responsibility: Sarit (67090500441) - ML Engineer

Supports two execution modes:
1. Standalone Web Server & Static HTML Export (FastAPI + Leaflet.js + Chart.js)
   -> Works out-of-the-box in any environment via `python run_pipeline.py dashboard`
   -> Generates `artifacts/dashboard_index.html` and serves live API at http://127.0.0.1:8050
2. Native Streamlit App (`streamlit run src/ml_engineering/app_dashboard.py`)

Includes 4 Core Views:
- Tab 1: ศูนย์ควบคุมการจราจร (Traffic Operations Center - Interactive Map & Onset Alerts)
- Tab 2: ข้อมูลผู้ขับขี่และจำลองสถานการณ์ฝน (Commuter Guidance & Rain What-If Simulator)
- Tab 3: วิเคราะห์ปัจจัยเชิงลึกและผังเมือง (SHAP Feature Importance, Ablation B3, Hypotheses H1-H5)
- Tab 4: MLOps Control Plane (Pipeline DAG Lineage, Model Registry, PSI & Concept Drift Monitoring)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import log_info, log_step, log_success
from src.utils.storage import ROOT_DIR, load_config, load_table


def collect_dashboard_payload() -> dict[str, Any]:
    """Aggregates Silver, Gold, Registry, Evaluation, and Monitoring data into a single JSON payload."""
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
    registry_index = _read_json(reg_dir / "registry_index.json")
    champion_meta = _read_json(reg_dir / "champion_metadata.json")
    drift_report = _read_json(mon_dir / "drift_monitoring_report.json")
    latest_run = _read_json(lin_dir / "latest_run.json")
    manifest = _read_json(ROOT_DIR / cfg["paths"]["manifest_path"])

    # Select 6 representative snapshots from Test split (Morning Rush, Midday, Evening Rush, Rainy Storm, Night)
    df_test = df_gold[df_gold["split"] == "test"].copy()
    df_test["ts_str"] = pd.to_datetime(df_test["ts_t"]).dt.strftime("%Y-%m-%d %H:%M")
    available_ts = sorted(df_test["ts_str"].unique())

    # Pick interesting snapshots including rainy evening and morning rush
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
                "ratio_t0": round(float(r["ratio_t0"]), 3),
                "pred_ratio_t60": round(float(r["pred_b2_lgbm"]), 3),
                "true_ratio_t60": round(float(r["target_ratio_t60"]), 3),
                "pred_congested": int(r["pred_is_congested_t60"]),
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
        "registry": registry_index,
        "champion": champion_meta,
        "monitoring": drift_report,
        "lineage": latest_run,
        "manifest_file_count": len(manifest.get("files", {})),
    }


def generate_standalone_dashboard_html(output_path: Path | None = None) -> Path:
    """Generates a rich, self-contained interactive HTML dashboard in artifacts/dashboard_index.html."""
    log_step("BUILD_DASHBOARD", "สร้าง Interactive Dashboard (4 Views: Control Center, Commuter, Urban Planner, MLOps)...")
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
  <title>Bangkok Traffic Forecasting & MLOps Dashboard (1-Hr Ahead)</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg-dark: #0b1120;
      --bg-card: #151f32;
      --bg-card-hover: #1e293b;
      --border: #283548;
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --accent-purple: #a855f7;
      --green: #22c55e;
      --amber: #f59e0b;
      --red: #ef4444;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-dark);
      color: var(--text-main);
      line-height: 1.5;
    }}
    header {{
      background: linear-gradient(90deg, #0f172a 0%, #1e1b4b 100%);
      border-bottom: 1px solid var(--border);
      padding: 16px 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .brand-title {{ font-size: 1.35rem; font-weight: 700; color: #fff; display: flex; align-items: center; gap: 10px; }}
    .brand-sub {{ font-size: 0.85rem; color: var(--text-muted); }}
    .status-pill {{
      background: rgba(34, 197, 94, 0.15);
      border: 1px solid var(--green);
      color: #4ade80;
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 600;
    }}
    .nav-tabs {{
      display: flex;
      background: #0f172a;
      border-bottom: 1px solid var(--border);
      padding: 0 28px;
      gap: 8px;
      overflow-x: auto;
    }}
    .tab-btn {{
      background: transparent;
      color: var(--text-muted);
      border: none;
      border-bottom: 3px solid transparent;
      padding: 14px 20px;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      white-space: nowrap;
    }}
    .tab-btn:hover {{ color: #fff; background: rgba(255,255,255,0.03); }}
    .tab-btn.active {{
      color: var(--accent);
      border-bottom-color: var(--accent);
      background: rgba(56, 189, 248, 0.08);
    }}
    .container {{ max-width: 1560px; margin: 0 auto; padding: 22px 28px; }}
    .tab-pane {{ display: none; }}
    .tab-pane.active {{ display: block; }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}
    .kpi-card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px 20px;
    }}
    .kpi-label {{ font-size: 0.82rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; }}
    .kpi-val {{ font-size: 1.9rem; font-weight: 700; margin: 6px 0 4px; }}
    .kpi-sub {{ font-size: 0.8rem; color: var(--text-muted); }}
    .grid-2 {{ display: grid; grid-template-columns: 1.35fr 1fr; gap: 20px; margin-bottom: 20px; }}
    .grid-equal {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
    @media (max-width: 1024px) {{
      .grid-2, .grid-equal {{ grid-template-columns: 1fr; }}
    }}
    .card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--border);
    }}
    .card-title {{ font-size: 1.05rem; font-weight: 600; color: #fff; }}
    #map {{ height: 500px; width: 100%; border-radius: 10px; border: 1px solid var(--border); }}
    .controls-bar {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      align-items: center;
      background: #0f172a;
      padding: 12px 16px;
      border-radius: 10px;
      border: 1px solid var(--border);
      margin-bottom: 16px;
    }}
    select, input[type="range"] {{
      background: var(--bg-card);
      color: #fff;
      border: 1px solid var(--border);
      padding: 8px 12px;
      border-radius: 6px;
      font-size: 0.9rem;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--text-muted); font-weight: 600; background: rgba(15, 23, 42, 0.6); }}
    tr:hover {{ background: rgba(255, 255, 255, 0.02); }}
    .badge {{
      display: inline-block;
      padding: 3px 9px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .badge-green {{ background: rgba(34, 197, 94, 0.18); color: #4ade80; border: 1px solid rgba(34,197,94,0.4); }}
    .badge-amber {{ background: rgba(245, 158, 11, 0.18); color: #fbbf24; border: 1px solid rgba(245,158,11,0.4); }}
    .badge-red {{ background: rgba(239, 68, 68, 0.18); color: #f87171; border: 1px solid rgba(239,68,68,0.4); }}
    .badge-purple {{ background: rgba(168, 85, 247, 0.22); color: #c084fc; border: 1px solid rgba(168,85,247,0.5); }}
    .hyp-card {{
      background: rgba(15, 23, 42, 0.65);
      border-left: 4px solid var(--green);
      padding: 12px 16px;
      border-radius: 6px;
      margin-bottom: 10px;
    }}
    .legend-box {{ display: flex; gap: 14px; font-size: 0.8rem; color: var(--text-muted); flex-wrap: wrap; }}
    .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; vertical-align: middle; margin-right: 5px; }}
  </style>
</head>
<body>
  <header>
    <div>
      <div class="brand-title">🚦 ระบบพยากรณ์จุดรถติดในกรุงเทพมหานครล่วงหน้า 1 ชั่วโมง และวิเคราะห์ปัจจัย (CSS 342)</div>
      <div class="brand-sub">Medallion Lakehouse + LightGBM + Guardrail Onset Recall + MLOps Control Plane | ผู้รับผิดชอบระบบ MLOps & Dashboard: สฤษดิ์ (67090500441)</div>
    </div>
    <div id="header-champion-pill" class="status-pill">● Champion Active</div>
  </header>

  <nav class="nav-tabs">
    <button class="tab-btn active" onclick="switchTab('tab1', this)">📍 Tab 1: ศูนย์ควบคุมการจราจร (Control Center)</button>
    <button class="tab-btn" onclick="switchTab('tab2', this)">🚗 Tab 2: ผู้ขับขี่ & จำลองฝน (Commuter & What-If)</button>
    <button class="tab-btn" onclick="switchTab('tab3', this)">📊 Tab 3: วิเคราะห์ปัจจัย & ผังเมือง (SHAP & Ablation)</button>
    <button class="tab-btn" onclick="switchTab('tab4', this)">⚙️ Tab 4: MLOps Control Plane (Registry, Drift & Lineage)</button>
  </nav>

  <div class="container">
    <!-- ==================== TAB 1: TRAFFIC OPERATIONS CENTER ==================== -->
    <div id="tab1" class="tab-pane active">
      <div class="controls-bar">
        <div>
          <label for="snapshot-select" style="color:var(--text-muted);font-size:0.85rem;margin-right:6px;">🕒 เวลาปัจจุบัน (t):</label>
          <select id="snapshot-select" onchange="renderControlCenter()"></select>
        </div>
        <div>
          <label for="district-filter" style="color:var(--text-muted);font-size:0.85rem;margin-right:6px;">🏙️ เขต (District):</label>
          <select id="district-filter" onchange="renderControlCenter()">
            <option value="ALL">ทุกเขตใน กทม. (All Districts)</option>
          </select>
        </div>
        <div>
          <label for="dir-filter" style="color:var(--text-muted);font-size:0.85rem;margin-right:6px;">🧭 ทิศทาง:</label>
          <select id="dir-filter" onchange="renderControlCenter()">
            <option value="ALL">ทุกทิศทาง (Inbound & Outbound)</option>
            <option value="inbound">ขาเข้าเมือง (Inbound)</option>
            <option value="outbound">ขาออกเมือง (Outbound)</option>
          </select>
        </div>
        <div class="legend-box" style="margin-left:auto;">
          <span><i class="legend-dot" style="background:#22c55e;"></i>คล่องตัว (&ge;0.70)</span>
          <span><i class="legend-dot" style="background:#f59e0b;"></i>ชะลอตัว (0.50–0.69)</span>
          <span><i class="legend-dot" style="background:#ef4444;"></i>ติดขัด (&lt;0.50)</span>
          <span><i class="legend-dot" style="background:#a855f7;"></i>🚨 Onset Alert (เริ่มติดใหม่ใน 60 นาที)</span>
        </div>
      </div>

      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-label">พยากรณ์จุดรถติด (t + 60 นาที)</div>
          <div class="kpi-val" id="kpi-congested-count" style="color:var(--red);">0</div>
          <div class="kpi-sub" id="kpi-congested-sub">จากทั้งหมด 0 ช่วงถนน</div>
        </div>
        <div class="kpi-card" style="border-color:rgba(168,85,247,0.5);">
          <div class="kpi-label">🚨 แจ้งเตือนเริ่มติดใหม่ (Onset Warning)</div>
          <div class="kpi-val" id="kpi-onset-count" style="color:var(--accent-purple);">0</div>
          <div class="kpi-sub">ตอนนี้คล่องตัว (&ge;0.55) แต่อีก 1 ชม. จะติด (&lt;0.50)</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">ความเร็วเฉลี่ยพยากรณ์ (t + 60 นาที)</div>
          <div class="kpi-val" id="kpi-avg-speed" style="color:var(--accent);">0.0 km/h</div>
          <div class="kpi-sub" id="kpi-avg-ratio">Congestion Ratio เฉลี่ย: 0.00</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">ปริมาณฝนสะสม 3 ชม. เฉลี่ย</div>
          <div class="kpi-val" id="kpi-rain" style="color:var(--amber);">0.0 mm</div>
          <div class="kpi-sub" id="kpi-incident-sub">อุบัติเหตุบนสายทาง: 0 จุด</div>
        </div>
      </div>

      <div class="grid-2">
        <div class="card">
          <div class="card-header">
            <div class="card-title">🗺️ แผนที่พยากรณ์สภาพจราจรล่วงหน้า 1 ชั่วโมง (t + 60 นาที)</div>
            <span class="badge badge-purple" id="map-time-badge">Snapshot</span>
          </div>
          <div id="map"></div>
        </div>
        <div class="card" style="max-height:575px;overflow-y:auto;">
          <div class="card-header">
            <div class="card-title">🚨 รายการเฝ้าระวังรายช่วงถนน (จัดลำดับตามความวิกฤต t+60)</div>
          </div>
          <table>
            <thead>
              <tr>
                <th>ถนน / ทิศทาง</th>
                <th>เขต</th>
                <th>Ratio (t)</th>
                <th>พยากรณ์ (t+60)</th>
                <th>สถานะ 1 ชม. ข้างหน้า</th>
              </tr>
            </thead>
            <tbody id="segment-table-body"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ==================== TAB 2: COMMUTER & WHAT-IF SIMULATOR ==================== -->
    <div id="tab2" class="tab-pane">
      <div class="controls-bar">
        <div>
          <label style="color:var(--text-muted);font-size:0.85rem;margin-right:8px;">🌧️ จำลองปริมาณฝนตกหนักเพิ่มขึ้น (What-If Rain Adder):</label>
          <input type="range" id="rain-slider" min="0" max="40" step="5" value="0" oninput="renderCommuterView()" />
          <span id="rain-slider-val" class="badge badge-amber" style="margin-left:8px;font-size:0.9rem;">+0 mm/3h</span>
        </div>
        <div>
          <label style="color:var(--text-muted);font-size:0.85rem;margin-right:8px;">🛣️ เลือกสายทางหลักที่ต้องการเดินทาง:</label>
          <select id="commuter-seg-select" onchange="renderCommuterView()"></select>
        </div>
      </div>

      <div class="grid-equal">
        <div class="card">
          <div class="card-header">
            <div class="card-title">🧭 คำแนะนำการเดินทางล่วงหน้า 1 ชั่วโมง (Smart Commuter Advisory)</div>
          </div>
          <div id="commuter-advisory-box" style="padding:10px 0;"></div>
        </div>
        <div class="card">
          <div class="card-header">
            <div class="card-title">⛈️ ผลกระทบจำลองเมื่อฝนตกหนักร่วมกับจุดเสี่ยงน้ำท่วม Traffy Fondue</div>
          </div>
          <canvas id="chart-whatif-rain" height="210"></canvas>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title">🔀 ตารางเปรียบเทียบเส้นทางหลักและเส้นทางเลี่ยงในพื้นที่ใกล้เคียง</div>
        </div>
        <table>
          <thead>
            <tr>
              <th>รหัสสายทาง</th>
              <th>ชื่อถนน</th>
              <th>เขต</th>
              <th>ความเร็วปัจจุบัน (t)</th>
              <th>พยากรณ์ปกติ (t+60)</th>
              <th>พยากรณ์เมื่อฝนเพิ่ม (What-If)</th>
              <th>จุดเสี่ยงน้ำท่วม (500m)</th>
              <th>ข้อแนะนำสำหรับผู้ขับขี่</th>
            </tr>
          </thead>
          <tbody id="commuter-table-body"></tbody>
        </table>
      </div>
    </div>

    <!-- ==================== TAB 3: URBAN PLANNERS, SHAP & ABLATION ==================== -->
    <div id="tab3" class="tab-pane">
      <div class="grid-equal">
        <div class="card">
          <div class="card-header">
            <div class="card-title">📊 Global Feature Attribution (Top 12 ตัวแปรที่ส่งผลต่อการเกิดรถติดสูงสุด)</div>
            <span class="badge badge-green" id="shap-method-badge">Attribution</span>
          </div>
          <canvas id="chart-shap-top" height="240"></canvas>
        </div>
        <div class="card">
          <div class="card-header">
            <div class="card-title">🧩 สัดส่วนความสำคัญแยกตามกลุ่มฟีเจอร์ทั้ง 8 กลุ่ม (8 Feature Groups Share %)</div>
          </div>
          <canvas id="chart-group-share" height="240"></canvas>
        </div>
      </div>

      <div class="grid-equal">
        <div class="card">
          <div class="card-header">
            <div class="card-title">🔬 ผลการทดลอง B3: Ablation Study (วัดผลกระทบเมื่อตัดฟีเจอร์ทีละกลุ่ม)</div>
          </div>
          <table>
            <thead>
              <tr>
                <th>กลุ่มฟีเจอร์ที่ตัดออก (Removed Group)</th>
                <th>F1 (All)</th>
                <th>&Delta;F1</th>
                <th>Onset Recall</th>
                <th>Rainy F1</th>
                <th>MAE</th>
              </tr>
            </thead>
            <tbody id="ablation-table-body"></tbody>
          </table>
        </div>
        <div class="card">
          <div class="card-header">
            <div class="card-title">✅ ผลการทดสอบสมมุติฐานวิจัย (Hypotheses H1–H5 Verification)</div>
          </div>
          <div id="hypotheses-container"></div>
        </div>
      </div>
    </div>

    <!-- ==================== TAB 4: MLOPS CONTROL PLANE ==================== -->
    <div id="tab4" class="tab-pane">
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-label">Champion Model Version</div>
          <div class="kpi-val" id="mlops-champ-ver" style="color:var(--green);">v1.0.0</div>
          <div class="kpi-sub" id="mlops-champ-th">Optimal Threshold: 0.50</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Test F1-Score (B2 vs B0)</div>
          <div class="kpi-val" id="mlops-f1-val" style="color:var(--accent);">0.000</div>
          <div class="kpi-sub" id="mlops-f1-gain">+0.000 over Persistence</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Guardrail: Onset Recall</div>
          <div class="kpi-val" id="mlops-onset-val" style="color:var(--accent-purple);">0.000</div>
          <div class="kpi-sub">B0 Persistence Onset Recall = 0.0000</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Data Drift Status (PSI)</div>
          <div class="kpi-val" id="mlops-drift-status" style="color:var(--green);">STABLE</div>
          <div class="kpi-sub" id="mlops-manifest-count">SHA256 Tracked Files: 0</div>
        </div>
      </div>

      <div class="grid-equal">
        <div class="card">
          <div class="card-header">
            <div class="card-title">🏆 เปรียบเทียบประสิทธิภาพโมเดล (B0 Persistence vs B1 Historical vs B2 LightGBM)</div>
          </div>
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Split</th>
                <th>F1 (All)</th>
                <th>Onset Recall (Guardrail)</th>
                <th>Rush Hour F1</th>
                <th>Rainy F1</th>
                <th>MAE Ratio</th>
              </tr>
            </thead>
            <tbody id="model-comp-tbody"></tbody>
          </table>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-title">🛡️ Champion Promotion Gates & Model Registry Lineage</div>
          </div>
          <div id="promotion-gates-box"></div>
        </div>
      </div>

      <div class="grid-equal">
        <div class="card">
          <div class="card-header">
            <div class="card-title">📉 การตรวจจับ Data Drift ด้วยค่า PSI (Normal Test vs Monsoon Stress-Test)</div>
            <select id="drift-mode-select" onchange="renderDriftChart()">
              <option value="normal">ชุดข้อมูล Test ปกติ (Current Window)</option>
              <option value="stress">จำลองพายุฝนรุนแรง (Monsoon Drift Stress-Test)</option>
            </select>
          </div>
          <canvas id="chart-psi-drift" height="210"></canvas>
        </div>

        <div class="card">
          <div class="card-header">
            <div class="card-title">📈 ติดตาม Concept Drift รายวัน (Daily F1 & Onset Recall Timeline)</div>
          </div>
          <canvas id="chart-daily-perf" height="210"></canvas>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div class="card-title">⛓️ Pipeline Execution DAG & Data Lineage Log</div>
          <span class="badge badge-green" id="lineage-run-id">RUN_ID</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>DAG Step</th>
              <th>ผู้รับผิดชอบ (Role)</th>
              <th>รายละเอียดการทำงาน</th>
              <th>สถานะ</th>
              <th>เวลาที่ใช้ (วินาที)</th>
            </tr>
          </thead>
          <tbody id="lineage-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const DATA = {payload_json};
    let map = null;
    let mapLayerGroup = null;
    let chartWhatIf = null;
    let chartShap = null;
    let chartGroup = null;
    let chartPsi = null;
    let chartDaily = null;

    function switchTab(tabId, btnEl) {{
      document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
      btnEl.classList.add('active');
      if (tabId === 'tab1' && map) {{
        setTimeout(() => map.invalidateSize(), 150);
      }}
    }}

    function ratioColor(r) {{
      if (r < 0.50) return '#ef4444';
      if (r < 0.70) return '#f59e0b';
      return '#22c55e';
    }}

    function initControls() {{
      const snapSel = document.getElementById('snapshot-select');
      DATA.snapshot_keys.forEach((k, idx) => {{
        const opt = document.createElement('option');
        opt.value = k;
        opt.textContent = k + ' น. (พยากรณ์ +60 นาที)';
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

    function renderControlCenter() {{
      const tsKey = document.getElementById('snapshot-select').value;
      const distVal = document.getElementById('district-filter').value;
      const dirVal = document.getElementById('dir-filter').value;
      document.getElementById('map-time-badge').textContent = 'เวลา t = ' + tsKey;

      let rows = (DATA.snapshots[tsKey] || []).slice();
      if (distVal !== 'ALL') rows = rows.filter(r => r.district_name === distVal);
      if (dirVal !== 'ALL') rows = rows.filter(r => r.direction === dirVal);

      const nCong = rows.filter(r => r.pred_congested === 1).length;
      const nOnset = rows.filter(r => r.pred_onset_alert === 1).length;
      const avgSpd = rows.length ? (rows.reduce((a, b) => a + b.pred_speed_t60_kmh, 0) / rows.length) : 0;
      const avgRat = rows.length ? (rows.reduce((a, b) => a + b.pred_ratio_t60, 0) / rows.length) : 0;
      const avgRain3h = rows.length ? (rows.reduce((a, b) => a + b.rain_cum_3h, 0) / rows.length) : 0;
      const nInc = rows.filter(r => r.active_incident === 1).length;

      document.getElementById('kpi-congested-count').textContent = `${{nCong}} จุด`;
      document.getElementById('kpi-congested-sub').textContent = `จากทั้งหมด ${{rows.length}} ช่วงถนนที่เลือก`;
      document.getElementById('kpi-onset-count').textContent = `${{nOnset}} จุด`;
      document.getElementById('kpi-avg-speed').textContent = `${{avgSpd.toFixed(1)}} km/h`;
      document.getElementById('kpi-avg-ratio').textContent = `Congestion Ratio เฉลี่ย: ${{avgRat.toFixed(2)}}`;
      document.getElementById('kpi-rain').textContent = `${{avgRain3h.toFixed(1)}} mm`;
      document.getElementById('kpi-incident-sub').textContent = `อุบัติเหตุกำลังเกิด: ${{nInc}} จุด`;

      mapLayerGroup.clearLayers();
      rows.forEach(r => {{
        const color = r.pred_onset_alert === 1 ? '#a855f7' : ratioColor(r.pred_ratio_t60);
        const line = L.polyline([[r.lat_start, r.lon_start], [r.lat_end, r.lon_end]], {{
          color: color,
          weight: r.pred_onset_alert === 1 ? 8 : 6,
          opacity: 0.92
        }}).addTo(mapLayerGroup);

        const midLat = 0.5 * (r.lat_start + r.lat_end);
        const midLon = 0.5 * (r.lon_start + r.lon_end);
        const circle = L.circleMarker([midLat, midLon], {{
          radius: r.pred_onset_alert === 1 ? 8 : 5,
          fillColor: color,
          color: '#fff',
          weight: 1.5,
          fillOpacity: 0.95
        }}).addTo(mapLayerGroup);

        const popupHtml = `
          <div style="color:#0f172a;min-width:210px;">
            <b>${{r.road_name}}</b> (${{r.direction}})<br/>
            เขต: ${{r.district_name}} | Free-flow: ${{r.free_flow_kmh}} km/h<hr style="margin:4px 0;"/>
            ปัจจุบัน (t): <b>${{r.speed_now_kmh}} km/h</b> (Ratio ${{r.ratio_t0}})<br/>
            พยากรณ์ (t+60): <b>${{r.pred_speed_t60_kmh}} km/h</b> (Ratio ${{r.pred_ratio_t60}})<br/>
            ฝนสะสม 3 ชม.: ${{r.rain_cum_3h}} mm | จุดเสี่ยงน้ำท่วม Traffy: ${{r.flood_buffer500m}} รายการ
          </div>
        `;
        line.bindPopup(popupHtml);
        circle.bindPopup(popupHtml);
      }});

      rows.sort((a, b) => (b.pred_onset_alert - a.pred_onset_alert) || (a.pred_ratio_t60 - b.pred_ratio_t60));
      const tbody = document.getElementById('segment-table-body');
      tbody.innerHTML = rows.map(r => {{
        let badge = '';
        if (r.pred_onset_alert === 1) {{
          badge = '<span class="badge badge-purple">🚨 ONSET ALERT (เริ่มติดใหม่)</span>';
        }} else if (r.pred_congested === 1) {{
          badge = '<span class="badge badge-red">🔴 ติดขัด (Congested)</span>';
        }} else if (r.pred_ratio_t60 < 0.70) {{
          badge = '<span class="badge badge-amber">🟡 ชะลอตัว</span>';
        }} else {{
          badge = '<span class="badge badge-green">🟢 คล่องตัว</span>';
        }}
        return `
          <tr>
            <td><b>${{r.road_name}}</b><br/><span style="color:var(--text-muted);font-size:0.78rem;">${{r.seg_id}} (${{r.direction}})</span></td>
            <td>${{r.district_name}}</td>
            <td>${{r.ratio_t0.toFixed(2)}} (${{r.speed_now_kmh}} km/h)</td>
            <td><b>${{r.pred_ratio_t60.toFixed(2)}}</b> (${{r.pred_speed_t60_kmh}} km/h)</td>
            <td>${{badge}}</td>
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

      const advBox = document.getElementById('commuter-advisory-box');
      const isRisk = targetSeg.whatIfRatio < 0.50;
      advBox.innerHTML = `
        <div style="background:rgba(15,23,42,0.7);padding:16px;border-radius:10px;border-left:4px solid ${{isRisk ? '#ef4444' : '#22c55e'}};">
          <div style="font-size:1.1rem;font-weight:700;margin-bottom:6px;">
            เส้นทางที่เลือก: ${{targetSeg.road_name}} (${{targetSeg.direction === 'inbound' ? 'ขาเข้า' : 'ขาออก'}})
          </div>
          <p style="color:var(--text-muted);margin-bottom:10px;">
            ความเร็วปัจจุบัน: <b>${{targetSeg.speed_now_kmh}} km/h</b> (Ratio ${{targetSeg.ratio_t0}}) &rarr;
            พยากรณ์ในอีก 60 นาที (ภายใต้ฝน +${{rainAdd}} mm): <b style="color:${{isRisk ? '#f87171' : '#4ade80'}};">${{targetSeg.whatIfSpeed}} km/h (Ratio ${{targetSeg.whatIfRatio}})</b>
          </p>
          <p style="font-size:0.9rem;">
            ${{isRisk
              ? `⚠️ <b>คำเตือน:</b> คาดการณ์ว่าจะเกิดรถติดหนักและชะลอตัวสูง (ประวัติร้องเรียนน้ำท่วมในรัศมี 500m: ${{targetSeg.flood_buffer500m}} รายการ) แนะนำให้เลี่ยงไปใช้: <b>${{alternatives.map(a => a.road_name).join(' หรือ ')}}</b>`
              : `✅ <b>คำแนะนำ:</b> เส้นทางนี้ยังคงคล่องตัวในอีก 1 ชั่วโมงข้างหน้า สามารถเดินทางได้ตามปกติ`
            }}
          </p>
        </div>
      `;

      const tbody = document.getElementById('commuter-table-body');
      tbody.innerHTML = rows.slice(0, 12).map(r => {{
        const rec = r.whatIfRatio < 0.50
          ? '<span class="badge badge-red">ควรหลีกเลี่ยง / เสี่ยงติดสะสม</span>'
          : '<span class="badge badge-green">แนะนำให้ใช้เดินทาง</span>';
        return `
          <tr>
            <td>${{r.seg_id}}</td>
            <td><b>${{r.road_name}}</b></td>
            <td>${{r.district_name}}</td>
            <td>${{r.speed_now_kmh}} km/h (${{r.ratio_t0}})</td>
            <td>${{r.pred_speed_t60_kmh}} km/h (${{r.pred_ratio_t60}})</td>
            <td><b>${{r.whatIfSpeed}} km/h (${{r.whatIfRatio}})</b></td>
            <td>${{r.flood_buffer500m}} รายการ</td>
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
            {{ label: 'พยากรณ์ปกติ (t+60)', data: topVuln.map(r => r.pred_ratio_t60), backgroundColor: '#38bdf8' }},
            {{ label: `จำลองเมื่อฝนตกเพิ่ม (+${{rainAdd}} mm)`, data: topVuln.map(r => r.whatIfRatio), backgroundColor: '#a855f7' }}
          ]
        }},
        options: {{
          responsive: true,
          plugins: {{ legend: {{ labels: {{ color: '#cbd5e1' }} }} }},
          scales: {{
            x: {{ ticks: {{ color: '#94a3b8' }} }},
            y: {{ min: 0, max: 1.1, ticks: {{ color: '#94a3b8' }}, title: {{ display: true, text: 'Congestion Ratio', color: '#94a3b8' }} }}
          }}
        }}
      }});
    }}

    function renderUrbanPlannerView() {{
      const shap = DATA.shap || {{}};
      document.getElementById('shap-method-badge').textContent = shap.attribution_method || 'TreeSHAP';
      const topFeats = (shap.top_features || []).slice(0, 12);

      const ctx1 = document.getElementById('chart-shap-top').getContext('2d');
      if (chartShap) chartShap.destroy();
      chartShap = new Chart(ctx1, {{
        type: 'bar',
        data: {{
          labels: topFeats.map(f => f.feature),
          datasets: [{{
            label: 'สัดส่วนความสำคัญ (%)',
            data: topFeats.map(f => f.share_pct),
            backgroundColor: '#38bdf8'
          }}]
        }},
        options: {{
          indexAxis: 'y',
          responsive: true,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ ticks: {{ color: '#94a3b8' }}, title: {{ display: true, text: 'Attribution Share (%)', color: '#94a3b8' }} }},
            y: {{ ticks: {{ color: '#f1f5f9' }} }}
          }}
        }}
      }});

      const grpShares = shap.group_attribution || [];
      const ctx2 = document.getElementById('chart-group-share').getContext('2d');
      if (chartGroup) chartGroup.destroy();
      chartGroup = new Chart(ctx2, {{
        type: 'bar',
        data: {{
          labels: grpShares.map(g => g.group_label.split('(')[0]),
          datasets: [{{
            label: 'Group Share (%)',
            data: grpShares.map(g => g.share_pct),
            backgroundColor: '#a855f7'
          }}]
        }},
        options: {{
          indexAxis: 'y',
          responsive: true,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{
            x: {{ ticks: {{ color: '#94a3b8' }} }},
            y: {{ ticks: {{ color: '#f1f5f9' }} }}
          }}
        }}
      }});

      const abRows = (DATA.ablation || {{}}).ablation_experiments || [];
      const abBody = document.getElementById('ablation-table-body');
      abBody.innerHTML = abRows.map(r => `
        <tr>
          <td><b>${{r.group_label}}</b></td>
          <td>${{r.f1_all.toFixed(4)}}</td>
          <td style="color:${{r.delta_f1_all < 0 ? '#f87171' : '#4ade80'}};">${{r.delta_f1_all >= 0 ? '+' : ''}}${{r.delta_f1_all.toFixed(4)}}</td>
          <td>${{r.onset_recall.toFixed(4)}} (${{r.delta_onset_recall >= 0 ? '+' : ''}}${{r.delta_onset_recall.toFixed(4)}})</td>
          <td>${{r.f1_rainy.toFixed(4)}}</td>
          <td>${{r.mae_ratio.toFixed(4)}}</td>
        </tr>
      `).join('');

      const hyps = shap.hypotheses_verification || [];
      document.getElementById('hypotheses-container').innerHTML = hyps.map(h => `
        <div class="hyp-card" style="border-left-color:${{h.supported ? '#22c55e' : '#f59e0b'}};">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <b>${{h.id}}: ${{h.title}}</b>
            <span class="badge ${{h.supported ? 'badge-green' : 'badge-amber'}}">${{h.supported ? '✔ SUPPORTED' : 'PARTIAL'}}</span>
          </div>
          <div style="color:var(--text-muted);font-size:0.84rem;margin-top:4px;">${{h.evidence}}</div>
        </div>
      `).join('');
    }}

    function renderMlopsView() {{
      const champ = DATA.champion || {{}};
      const evalModels = (DATA.evaluation || {{}}).models || {{}};
      const b0Test = (evalModels.B0_Persistence || {{}}).test || {{}};
      const b1Test = (evalModels.B1_Historical_Average || {{}}).test || {{}};
      const b2Val = (evalModels.B2_LightGBM || {{}}).val || {{}};
      const b2Test = (evalModels.B2_LightGBM || {{}}).test || {{}};
      const mon = DATA.monitoring || {{}};

      document.getElementById('header-champion-pill').textContent = `● Champion: ${{champ.version || 'v1.0.0'}} (${{mon.overall_system_health || 'HEALTHY'}})`;
      document.getElementById('mlops-champ-ver').textContent = champ.version || 'v1.0.0';
      document.getElementById('mlops-champ-th').textContent = `Optimal Threshold: ${{champ.optimal_threshold || 0.50}} | SHA256: ${{(champ.model_sha256 || '').slice(0, 10)}}...`;
      document.getElementById('mlops-f1-val').textContent = (b2Test.f1_all || 0).toFixed(4);
      const gain = ((b2Test.f1_all || 0) - (b0Test.f1_all || 0)).toFixed(4);
      document.getElementById('mlops-f1-gain').textContent = `+${{gain}} เหนือกว่า B0 Persistence`;
      document.getElementById('mlops-onset-val').textContent = (b2Test.onset_recall || 0).toFixed(4);
      document.getElementById('mlops-drift-status').textContent = mon.overall_system_health || 'HEALTHY_STABLE';
      document.getElementById('mlops-manifest-count').textContent = `Manifest Tracked Tables: ${{DATA.manifest_file_count || 0}} ไฟล์`;

      const compRows = [
        ['B0: Persistence (ทายว่าเหมือนตอนนี้)', 'Test', b0Test],
        ['B1: Historical Profile Average', 'Test', b1Test],
        ['B2: LightGBM Regressor (Val Tuned)', 'Val', b2Val],
        ['B2: LightGBM Regressor (Champion)', 'Test', b2Test],
      ];
      document.getElementById('model-comp-tbody').innerHTML = compRows.map(([name, sp, m]) => `
        <tr>
          <td><b>${{name}}</b></td>
          <td><span class="badge badge-purple">${{sp}}</span></td>
          <td><b>${{(m.f1_all || 0).toFixed(4)}}</b></td>
          <td style="color:${{(m.onset_recall || 0) > 0.5 ? '#4ade80' : '#f87171'}};"><b>${{(m.onset_recall || 0).toFixed(4)}}</b></td>
          <td>${{(m.f1_rush_hour || 0).toFixed(4)}}</td>
          <td>${{(m.f1_rainy || 0).toFixed(4)}}</td>
          <td>${{(m.mae_ratio || 0).toFixed(4)}}</td>
        </tr>
      `).join('');

      const gates = ((champ.promotion_decision || {{}}).gates) || {{}};
      document.getElementById('promotion-gates-box').innerHTML = Object.entries(gates).map(([gName, gObj]) => `
        <div class="hyp-card" style="border-left-color:${{gObj.passed ? '#22c55e' : '#ef4444'}};">
          <div style="display:flex;justify-content:space-between;">
            <b>${{gName}}</b>
            <span class="badge ${{gObj.passed ? 'badge-green' : 'badge-red'}}">${{gObj.passed ? '✔ PASSED' : '✖ FAILED'}}</span>
          </div>
          <div style="color:var(--text-muted);font-size:0.8rem;margin-top:4px;">${{JSON.stringify(gObj)}}</div>
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
            {{ label: 'B2 LightGBM Daily F1', data: daily.map(d => d.f1_b2), borderColor: '#38bdf8', tension: 0.25 }},
            {{ label: 'Rolling 3-Day F1', data: daily.map(d => d.rolling_3d_f1), borderColor: '#22c55e', borderDash: [5, 5], tension: 0.25 }},
            {{ label: 'B0 Persistence F1', data: daily.map(d => d.f1_b0), borderColor: '#94a3b8', tension: 0.25 }}
          ]
        }},
        options: {{
          responsive: true,
          plugins: {{ legend: {{ labels: {{ color: '#cbd5e1' }} }} }},
          scales: {{
            x: {{ ticks: {{ color: '#94a3b8' }} }},
            y: {{ min: 0.0, max: 1.05, ticks: {{ color: '#94a3b8' }} }}
          }}
        }}
      }});

      const lin = DATA.lineage || {{}};
      document.getElementById('lineage-run-id').textContent = lin.run_id || 'LATEST_RUN';
      const steps = lin.steps || [];
      document.getElementById('lineage-tbody').innerHTML = steps.map((s, i) => `
        <tr>
          <td>${{i + 1}}</td>
          <td><b>${{s.step}}</b></td>
          <td>${{s.role || '-'}}</td>
          <td>${{s.description || '-'}}</td>
          <td><span class="badge badge-green">✔ ${{s.status}}</span></td>
          <td>${{(s.duration_sec || 0).toFixed(2)}}s</td>
        </tr>
      `).join('');
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
            backgroundColor: items.map(x => x.psi >= 0.25 ? '#ef4444' : (x.psi >= 0.10 ? '#f59e0b' : '#22c55e'))
          }}]
        }},
        options: {{
          responsive: true,
          plugins: {{ legend: {{ labels: {{ color: '#cbd5e1' }} }} }},
          scales: {{
            x: {{ ticks: {{ color: '#94a3b8' }} }},
            y: {{ beginAtZero: true, ticks: {{ color: '#94a3b8' }}, title: {{ display: true, text: 'PSI Value (Crit >= 0.25)', color: '#94a3b8' }} }}
          }}
        }}
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

    log_info(f"สร้างไฟล์ Interactive HTML Dashboard สำเร็จ: {out_file}")
    log_success("สามารถเปิดดูผ่าน Browser หรือรันผ่าน FastAPI / Streamlit ได้ทันที")
    return out_file


def serve_dashboard(host: str = "127.0.0.1", port: int = 8050) -> None:
    """Serves the interactive dashboard & REST API via FastAPI + Uvicorn."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn

    html_path = generate_standalone_dashboard_html()
    app = FastAPI(title="Bangkok Traffic Forecasting & MLOps API", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return html_path.read_text(encoding="utf-8")

    @app.get("/api/payload", response_class=JSONResponse)
    def api_payload() -> dict[str, Any]:
        return collect_dashboard_payload()

    log_info(f"กำลังเปิดเซิร์ฟเวอร์ Dashboard ที่ http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    generate_standalone_dashboard_html()
