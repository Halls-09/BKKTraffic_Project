"""
Realistic Synthetic Data Generator for Bangkok Traffic Forecasting (Bronze Layer).
Simulates:
- 24 real Bangkok road segments across 6 weather zones & 12 districts with directed graph neighbors
- 15-minute traffic probe speeds incorporating hypotheses H1-H5:
  * H1: Morning Inbound vs Evening Outbound rush hours
  * H2: Friday evening & Payday window effects
  * H3: Downstream spillback propagation (t -> t+60)
  * H4: Convective rain + Traffy flood vulnerability interaction (including July 21, 2022 flood spike)
  * H5: Active road incidents
- Intentional sensor anomalies & missing nighttime probes to test Quality Gate 4.2
- Intentional Traffy Fondue duplicate tickets & Platform Adoption Bias
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import log_info, log_step, log_success
from src.utils.storage import ROOT_DIR, load_config, save_table


BANGKOK_SEGMENTS = [
    # (seg_id, road_name, direction, lat_s, lon_s, lat_e, lon_e, len_km, district, zone, free_flow, flood_prone, up_ids, down_ids)
    ("SEG_SUK_01_IN", "ถนนสุขุมวิท (อโศก-นานา)", "inbound", 13.737, 100.560, 13.742, 100.553, 1.2, "วัฒนา", "ZONE_CENTRAL", 48.0, 0.85, ["SEG_SUK_02_IN", "SEG_ASO_01_IN"], ["SEG_PLO_01_IN"]),
    ("SEG_SUK_01_OUT", "ถนนสุขุมวิท (นานา-อโศก)", "outbound", 13.742, 100.553, 13.737, 100.560, 1.2, "วัฒนา", "ZONE_CENTRAL", 48.0, 0.85, ["SEG_PLO_01_OUT"], ["SEG_SUK_02_OUT", "SEG_ASO_01_OUT"]),
    ("SEG_SUK_02_IN", "ถนนสุขุมวิท (ทองหล่อ-พร้อมพงษ์)", "inbound", 13.724, 100.578, 13.731, 100.569, 1.5, "วัฒนา", "ZONE_EAST", 52.0, 0.90, ["SEG_BNA_01_IN"], ["SEG_SUK_01_IN"]),
    ("SEG_SUK_02_OUT", "ถนนสุขุมวิท (พร้อมพงษ์-ทองหล่อ)", "outbound", 13.731, 100.569, 13.724, 100.578, 1.5, "วัฒนา", "ZONE_EAST", 52.0, 0.90, ["SEG_SUK_01_OUT"], ["SEG_BNA_01_OUT"]),
    ("SEG_PLO_01_IN", "ถนนเพลินจิต (ชิดลม-ราชประสงค์)", "inbound", 13.744, 100.548, 13.745, 100.539, 1.1, "ปทุมวัน", "ZONE_CENTRAL", 45.0, 0.45, ["SEG_SUK_01_IN"], ["SEG_RAM1_01_IN"]),
    ("SEG_PLO_01_OUT", "ถนนเพลินจิต (ราชประสงค์-ชิดลม)", "outbound", 13.745, 100.539, 13.744, 100.548, 1.1, "ปทุมวัน", "ZONE_CENTRAL", 45.0, 0.45, ["SEG_RAM1_01_OUT"], ["SEG_SUK_01_OUT"]),
    ("SEG_RAM1_01_IN", "ถนนพระราม 1 (สยาม-ปทุมวัน)", "inbound", 13.745, 100.539, 13.746, 100.530, 1.0, "ปทุมวัน", "ZONE_CENTRAL", 44.0, 0.40, ["SEG_PLO_01_IN"], ["SEG_PHY_01_IN"]),
    ("SEG_RAM1_01_OUT", "ถนนพระราม 1 (ปทุมวัน-สยาม)", "outbound", 13.746, 100.530, 13.745, 100.539, 1.0, "ปทุมวัน", "ZONE_CENTRAL", 44.0, 0.40, ["SEG_PHY_01_OUT"], ["SEG_PLO_01_OUT"]),
    ("SEG_ASO_01_IN", "ถนนอโศกมนตรี (เพชรบุรี-สุขุมวิท)", "inbound", 13.748, 100.563, 13.737, 100.560, 1.3, "วัฒนา", "ZONE_CENTRAL", 42.0, 0.95, ["SEG_RAT_01_IN"], ["SEG_SUK_01_IN", "SEG_RAM4_01_IN"]),
    ("SEG_ASO_01_OUT", "ถนนอโศกมนตรี (สุขุมวิท-เพชรบุรี)", "outbound", 13.737, 100.560, 13.748, 100.563, 1.3, "วัฒนา", "ZONE_CENTRAL", 42.0, 0.95, ["SEG_SUK_01_OUT", "SEG_RAM4_01_OUT"], ["SEG_RAT_01_OUT"]),
    ("SEG_RAT_01_IN", "ถนนรัชดาภิเษก (ห้วยขวาง-พระราม 9)", "inbound", 13.778, 100.574, 13.758, 100.565, 2.4, "ห้วยขวาง", "ZONE_NORTH", 58.0, 0.88, ["SEG_LAT_01_IN"], ["SEG_ASO_01_IN", "SEG_DIN_01_IN"]),
    ("SEG_RAT_01_OUT", "ถนนรัชดาภิเษก (พระราม 9-ห้วยขวาง)", "outbound", 13.758, 100.565, 13.778, 100.574, 2.4, "ห้วยขวาง", "ZONE_NORTH", 58.0, 0.88, ["SEG_ASO_01_OUT", "SEG_DIN_01_OUT"], ["SEG_LAT_01_OUT"]),
    ("SEG_LAT_01_IN", "ถนนลาดพร้าว (โชคชัย 4-รัชดา)", "inbound", 13.796, 100.595, 13.806, 100.575, 2.5, "ลาดพร้าว", "ZONE_NORTH", 50.0, 0.92, ["SEG_RAMK_01_IN"], ["SEG_RAT_01_IN", "SEG_PHA_01_IN"]),
    ("SEG_LAT_01_OUT", "ถนนลาดพร้าว (รัชดา-โชคชัย 4)", "outbound", 13.806, 100.575, 13.796, 100.595, 2.5, "ลาดพร้าว", "ZONE_NORTH", 50.0, 0.92, ["SEG_RAT_01_OUT", "SEG_PHA_01_OUT"], ["SEG_RAMK_01_OUT"]),
    ("SEG_VIB_01_IN", "ถนนวิภาวดีรังสิต (จตุจักร-ดินแดง)", "inbound", 13.814, 100.560, 13.765, 100.558, 5.2, "จตุจักร", "ZONE_NORTH", 75.0, 0.75, ["SEG_PHA_01_IN"], ["SEG_DIN_01_IN"]),
    ("SEG_VIB_01_OUT", "ถนนวิภาวดีรังสิต (ดินแดง-จตุจักร)", "outbound", 13.765, 100.558, 13.814, 100.560, 5.2, "จตุจักร", "ZONE_NORTH", 75.0, 0.75, ["SEG_DIN_01_OUT"], ["SEG_PHA_01_OUT"]),
    ("SEG_DIN_01_IN", "ถนนดินแดง (วิภาวดี-อนุสาวรีย์ชัยฯ)", "inbound", 13.765, 100.558, 13.764, 100.538, 2.1, "ดินแดง", "ZONE_CENTRAL", 52.0, 0.70, ["SEG_VIB_01_IN", "SEG_RAT_01_IN"], ["SEG_PHY_01_IN"]),
    ("SEG_DIN_01_OUT", "ถนนดินแดง (อนุสาวรีย์ชัยฯ-วิภาวดี)", "outbound", 13.764, 100.538, 13.765, 100.558, 2.1, "ดินแดง", "ZONE_CENTRAL", 52.0, 0.70, ["SEG_PHY_01_OUT"], ["SEG_VIB_01_OUT", "SEG_RAT_01_OUT"]),
    ("SEG_PHY_01_IN", "ถนนพญาไท (อนุสาวรีย์ชัยฯ-ราชเทวี)", "inbound", 13.764, 100.538, 13.752, 100.532, 1.5, "ราชเทวี", "ZONE_CENTRAL", 46.0, 0.55, ["SEG_DIN_01_IN"], ["SEG_RAM1_01_IN"]),
    ("SEG_PHY_01_OUT", "ถนนพญาไท (ราชเทวี-อนุสาวรีย์ชัยฯ)", "outbound", 13.752, 100.532, 13.764, 100.538, 1.5, "ราชเทวี", "ZONE_CENTRAL", 46.0, 0.55, ["SEG_RAM1_01_OUT"], ["SEG_DIN_01_OUT"]),
    ("SEG_SAT_01_IN", "ถนนสาทรเหนือ (สะพานตากสิน-วิทยุ)", "inbound", 13.719, 100.514, 13.726, 100.544, 3.3, "สาทร", "ZONE_SOUTH", 54.0, 0.50, ["SEG_SIL_01_IN"], ["SEG_RAM4_01_IN"]),
    ("SEG_SAT_01_OUT", "ถนนสาทรใต้ (วิทยุ-สะพานตากสิน)", "outbound", 13.726, 100.544, 13.719, 100.514, 3.3, "สาทร", "ZONE_SOUTH", 54.0, 0.50, ["SEG_RAM4_01_OUT"], ["SEG_SIL_01_OUT"]),
    ("SEG_RAM4_01_IN", "ถนนพระราม 4 (คลองเตย-สีลม)", "inbound", 13.721, 100.558, 13.728, 100.534, 2.8, "คลองเตย", "ZONE_SOUTH", 55.0, 0.65, ["SEG_ASO_01_IN"], ["SEG_SAT_01_IN", "SEG_SIL_01_IN"]),
    ("SEG_RAM4_01_OUT", "ถนนพระราม 4 (สีลม-คลองเตย)", "outbound", 13.728, 100.534, 13.721, 100.558, 2.8, "คลองเตย", "ZONE_SOUTH", 55.0, 0.65, ["SEG_SAT_01_OUT", "SEG_SIL_01_OUT"], ["SEG_ASO_01_OUT"]),
]

EXTRA_SEGMENTS = [
    ("SEG_PHA_01_IN", "ถนนพหลโยธิน (เสนานิคม-จตุจักร)", "inbound", 13.835, 100.570, 13.802, 100.553, 3.8, "จตุจักร", "ZONE_NORTH", 56.0, 0.82, [], ["SEG_VIB_01_IN"]),
    ("SEG_PHA_01_OUT", "ถนนพหลโยธิน (จตุจักร-เสนานิคม)", "outbound", 13.802, 100.553, 13.835, 100.570, 3.8, "จตุจักร", "ZONE_NORTH", 56.0, 0.82, ["SEG_VIB_01_OUT"], []),
    ("SEG_BNA_01_IN", "ถนนบางนา-ตราด (บางนา-สุขุมวิท)", "inbound", 13.668, 100.635, 13.680, 100.608, 3.2, "บางนา", "ZONE_EAST", 65.0, 0.86, [], ["SEG_SUK_02_IN"]),
    ("SEG_BNA_01_OUT", "ถนนบางนา-ตราด (สุขุมวิท-บางนา)", "outbound", 13.680, 100.608, 13.668, 100.635, 3.2, "บางนา", "ZONE_EAST", 65.0, 0.86, ["SEG_SUK_02_OUT"], []),
    ("SEG_RAMK_01_IN", "ถนนรามคำแหง (หัวหมาก-คลองตัน)", "inbound", 13.758, 100.625, 13.743, 100.601, 3.0, "บางกะปิ", "ZONE_EAST", 50.0, 0.91, [], ["SEG_LAT_01_IN"]),
    ("SEG_RAMK_01_OUT", "ถนนรามคำแหง (คลองตัน-หัวหมาก)", "outbound", 13.743, 100.601, 13.758, 100.625, 3.0, "บางกะปิ", "ZONE_EAST", 50.0, 0.91, ["SEG_LAT_01_OUT"], []),
    ("SEG_SIL_01_IN", "ถนนสีลม (บางรัก-ศาลาแดง)", "inbound", 13.722, 100.517, 13.728, 100.534, 2.0, "บางรัก", "ZONE_SOUTH", 42.0, 0.48, [], ["SEG_RAM4_01_IN"]),
    ("SEG_SIL_01_OUT", "ถนนสีลม (ศาลาแดง-บางรัก)", "outbound", 13.728, 100.534, 13.722, 100.517, 2.0, "บางรัก", "ZONE_SOUTH", 42.0, 0.48, ["SEG_RAM4_01_OUT"], []),
]


def generate_synthetic_bronze_data(days_per_split: tuple[int, int, int] = (20, 7, 7)) -> dict[str, int]:
    """
    Generates realistic Bronze tables in data/01_bronze/ matching the exact schema
    and temporal dynamics of Bangkok traffic in June-September 2022.
    """
    log_step("SYNTHETIC_GEN", "กำลังสร้างชุดข้อมูลจำลอง iTIC + Open-Meteo + Traffy Fondue (Bronze Layer)...")
    cfg = load_config()
    rng = np.random.default_rng(cfg["project"]["random_seed"])

    all_segs = BANGKOK_SEGMENTS + EXTRA_SEGMENTS
    seg_records = []
    flood_vuln_map = {}
    free_flow_map = {}

    for row in all_segs:
        seg_id, rname, direction, lats, lons, late, lone, lkm, dist, zone, ff, fvuln, up_ids, down_ids = row
        seg_records.append({
            "seg_id": seg_id,
            "road_name": rname,
            "direction": direction,
            "lat_start": lats,
            "lon_start": lons,
            "lat_end": late,
            "lon_end": lone,
            "length_km": lkm,
            "district_name": dist,
            "weather_zone_id": zone,
            "upstream_seg_ids": up_ids,
            "downstream_seg_ids": down_ids,
            "true_free_flow_kmh": ff,
        })
        flood_vuln_map[seg_id] = fvuln
        free_flow_map[seg_id] = ff

    df_segments = pd.DataFrame(seg_records)

    # Create timestamps across Train, Val, and Test periods so OOT split works cleanly
    # Train: 2022-07-15 to 2022-08-15 (includes July 21 historic flood)
    # Val:   2022-08-16 to 2022-08-25
    # Test:  2022-09-01 to 2022-09-10
    ts_train = pd.date_range("2022-07-18 00:00:00+07:00", "2022-08-02 23:45:00+07:00", freq="15min")
    ts_val = pd.date_range("2022-08-16 00:00:00+07:00", "2022-08-21 23:45:00+07:00", freq="15min")
    ts_test = pd.date_range("2022-09-01 00:00:00+07:00", "2022-09-06 23:45:00+07:00", freq="15min")
    all_ts = ts_train.union(ts_val).union(ts_test)
    n_ts = len(all_ts)

    # 1. Generate Weather data per zone (ZONE_CENTRAL, ZONE_NORTH, ZONE_EAST, ZONE_SOUTH)
    zones = ["ZONE_CENTRAL", "ZONE_NORTH", "ZONE_EAST", "ZONE_SOUTH"]
    weather_rows = []
    hours = all_ts.hour + all_ts.minute / 60.0
    dates_str = all_ts.strftime("%Y-%m-%d")

    zone_rain_series: dict[str, np.ndarray] = {}
    for z_idx, zone in enumerate(zones):
        rain = np.zeros(n_ts, dtype=np.float64)
        # Tropical afternoon/evening convective rain bursts (15:00 - 21:00)
        for i in range(n_ts):
            hr = hours[i]
            d_str = dates_str[i]
            # July 21, 2022 historic Bangkok flood event
            if d_str == "2022-07-21" and 15.0 <= hr <= 22.5:
                rain[i] = rng.uniform(6.0, 18.5)
            elif d_str in ("2022-07-25", "2022-07-30", "2022-08-18", "2022-08-20", "2022-09-03", "2022-09-05") and 15.5 <= hr <= 20.5:
                rain[i] = max(0.0, rng.normal(4.5 + 0.8 * z_idx, 2.2))
            elif 14.0 <= hr <= 21.0 and rng.random() < 0.08:
                rain[i] = rng.exponential(2.0)

        # Smooth rain slightly across adjacent 15-min slots
        rain = np.convolve(rain, [0.2, 0.6, 0.2], mode="same")
        zone_rain_series[zone] = np.round(rain, 2)

        temp = 29.5 + 4.0 * np.sin(2 * np.pi * (hours - 9.0) / 24.0) - 0.25 * rain + rng.normal(0, 0.3, n_ts)
        for i in range(n_ts):
            weather_rows.append({
                "weather_zone_id": zone,
                "ts_15m": all_ts[i],
                "rain_mm": float(zone_rain_series[zone][i]),
                "temp_c": round(float(temp[i]), 2),
            })

    df_weather = pd.DataFrame(weather_rows)

    # 2. Generate Traffic Incidents (reported_time <= t)
    incident_rows = []
    seg_ids = [r["seg_id"] for r in seg_records]
    active_incident_matrix = np.zeros((len(seg_ids), n_ts), dtype=np.float64)
    inc_id = 1
    for s_idx, s_id in enumerate(seg_ids):
        for t_idx in range(0, n_ts - 6, 12):
            if rng.random() < 0.035:
                dur = int(rng.integers(2, 6))
                active_incident_matrix[s_idx, t_idx : min(n_ts, t_idx + dur)] = 1.0
                incident_rows.append({
                    "incident_id": f"INC_{inc_id:05d}",
                    "seg_id": s_id,
                    "event_time": all_ts[t_idx] - pd.Timedelta(minutes=int(rng.integers(2, 10))),
                    "reported_time": all_ts[t_idx],
                    "cleared_time": all_ts[min(n_ts - 1, t_idx + dur)],
                    "severity": int(rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])),
                })
                inc_id += 1
    df_incidents = pd.DataFrame(incident_rows)

    # 3. Generate Traffy Fondue reports (including duplicates & Platform Adoption Bias)
    traffy_rows = []
    districts = list({r["district_name"] for r in seg_records})
    ticket_idx = 1
    for r in seg_records:
        f_vuln = flood_vuln_map[r["seg_id"]]
        n_flood_tickets = int(20 + 120 * f_vuln)
        for _ in range(n_flood_tickets):
            # Platform Adoption Bias: more reports in June/July than August/Sept
            month_choice = int(rng.choice([6, 7, 8, 9], p=[0.42, 0.33, 0.15, 0.10]))
            day_choice = int(rng.integers(1, 28))
            hr_choice = int(rng.integers(0, 24))
            ts_ticket = pd.Timestamp(year=2022, month=month_choice, day=day_choice, hour=hr_choice, tz="Asia/Bangkok")
            lat_j = round(r["lat_start"] + rng.normal(0, 0.002), 4)
            lon_j = round(r["lon_start"] + rng.normal(0, 0.002), 4)
            row_t = {
                "ticket_id": f"TRAFFY_{ticket_idx:06d}",
                "timestamp": ts_ticket,
                "timestamp_hour": ts_ticket.strftime("%Y-%m-%d %H:00"),
                "type": rng.choice(["น้ำท่วม", "ถนน/ผิวจราจร"], p=[0.72, 0.28]),
                "district_name": r["district_name"],
                "nearest_seg_id": r["seg_id"],
                "lat": lat_j,
                "lon": lon_j,
            }
            traffy_rows.append(row_t)
            ticket_idx += 1
            # Inject ~25% duplicate submissions to test deduplication
            if rng.random() < 0.25:
                dup_row = dict(row_t)
                dup_row["ticket_id"] = f"TRAFFY_{ticket_idx:06d}"
                traffy_rows.append(dup_row)
                ticket_idx += 1

    df_traffy = pd.DataFrame(traffy_rows)

    # 4. Simulate coupled spatio-temporal traffic ratio dynamics across the road graph
    n_segs = len(seg_records)
    seg_idx_map = {r["seg_id"]: idx for idx, r in enumerate(seg_records)}
    ratios = np.full((n_segs, n_ts), 0.78, dtype=np.float64)

    dow = all_ts.dayofweek.to_numpy()
    day_num = all_ts.day.to_numpy()
    is_wknd = (dow >= 5).astype(np.float64)
    is_fri = (dow == 4).astype(np.float64)
    is_payday = ((day_num >= 25) | (day_num <= 2)).astype(np.float64)

    # Precompute 3h cumulative rain per zone
    zone_rain_cum3h = {
        z: pd.Series(zone_rain_series[z]).rolling(12, min_periods=1).sum().to_numpy()
        for z in zones
    }

    for t in range(n_ts):
        hr = hours[t]
        # Base diurnal profile (H1: Morning Inbound 06:30-09:30, Evening Outbound 16:00-20:00)
        morning_wave = np.exp(-0.5 * ((hr - 8.0) / 1.15) ** 2)
        evening_wave = np.exp(-0.5 * ((hr - 18.0) / 1.45) ** 2)
        midday_wave = 0.12 * np.exp(-0.5 * ((hr - 13.0) / 2.2) ** 2)

        for s_idx, r in enumerate(seg_records):
            s_id = r["seg_id"]
            direction = r["direction"]
            zone = r["weather_zone_id"]
            f_vuln = flood_vuln_map[s_id]

            if is_wknd[t] > 0.5:
                rush_drop = 0.22 * evening_wave + 0.15 * midday_wave
            else:
                if direction == "inbound":
                    rush_drop = 0.44 * morning_wave + 0.24 * evening_wave + midday_wave
                else:
                    rush_drop = 0.20 * morning_wave + 0.46 * evening_wave + midday_wave

            # H2: Calendar effects (Friday evening & Payday window)
            cal_drop = 0.0
            if is_fri[t] > 0.5 and 15.5 <= hr <= 21.5:
                cal_drop += 0.11
            if is_payday[t] > 0.5 and 16.0 <= hr <= 20.5:
                cal_drop += 0.07

            # H4: Rain & Flood Vulnerability Interaction (lags 0..4 slots)
            r_now = zone_rain_series[zone][t]
            r_3h = zone_rain_cum3h[zone][t]
            rain_drop = 0.025 * r_now + 0.014 * r_3h * (0.4 + f_vuln)

            # H5: Incident drop
            inc_drop = 0.22 * active_incident_matrix[s_idx, t]

            # H3: Downstream Spillback from t-1 to t-4 (so downstream congestion at t-4 strongly causes onset at t!)
            spillback_drop = 0.0
            if t >= 4 and len(r["downstream_seg_ids"]) > 0:
                down_indices = [seg_idx_map[d] for d in r["downstream_seg_ids"] if d in seg_idx_map]
                if down_indices:
                    down_ratio_lag4 = float(np.min(ratios[down_indices, t - 4]))
                    down_ratio_lag1 = float(np.min(ratios[down_indices, t - 1]))
                    if down_ratio_lag4 < 0.48:
                        spillback_drop += 0.55 * (0.48 - down_ratio_lag4)
                    if down_ratio_lag1 < 0.45:
                        spillback_drop += 0.30 * (0.45 - down_ratio_lag1)

            # Autoregressive inertia + innovation
            target_equilibrium = 0.92 - rush_drop - cal_drop - rain_drop - inc_drop - spillback_drop
            prev_r = ratios[s_idx, t - 1] if t > 0 else target_equilibrium
            new_r = 0.52 * prev_r + 0.48 * target_equilibrium + rng.normal(0, 0.025)
            ratios[s_idx, t] = float(np.clip(new_r, 0.12, 1.08))

    # Build traffic table & inject realistic sensor anomalies for Quality Gate 4.2
    traffic_records = []
    for s_idx, r in enumerate(seg_records):
        ff = r["true_free_flow_kmh"]
        s_id = r["seg_id"]
        for t in range(n_ts):
            hr = hours[t]
            # Simulate occasional missing nighttime probe slot (01:00 - 04:30)
            if 1.0 <= hr <= 4.5 and rng.random() < 0.015:
                continue

            spd = round(float(ratios[s_idx, t] * ff), 2)
            probes = int(rng.integers(8, 85)) if 6.0 <= hr <= 22.0 else int(rng.integers(2, 18))

            traffic_records.append({
                "seg_id": s_id,
                "ts_15m": all_ts[t],
                "speed_median_kmh": spd,
                "probe_count": probes,
            })

    df_traffic = pd.DataFrame(traffic_records)

    # Inject 35 explicit sensor outliers (> 150 km/h or negative or massive jump) so Quality Gate 4.2 quarantines/flags them
    outlier_indices = rng.choice(len(df_traffic), size=35, replace=False)
    df_traffic.loc[outlier_indices[:20], "speed_median_kmh"] = 185.0  # Exceeds 150 km/h contract
    df_traffic.loc[outlier_indices[20:30], "speed_median_kmh"] = 135.0  # >60 km/h jump
    df_traffic.loc[outlier_indices[30:], "probe_count"] = 0  # Zero probes

    bronze_dir = ROOT_DIR / cfg["paths"]["bronze_dir"]
    save_table(df_segments, bronze_dir / "bronze_itic_segments.parquet", layer="bronze")
    save_table(df_traffic, bronze_dir / "bronze_itic_traffic_15m.parquet", layer="bronze")
    save_table(df_weather, bronze_dir / "bronze_weather_15m.parquet", layer="bronze")
    save_table(df_traffy, bronze_dir / "bronze_traffy_reports.parquet", layer="bronze")
    save_table(df_incidents, bronze_dir / "bronze_incidents.parquet", layer="bronze")

    summary = {
        "segments": len(df_segments),
        "traffic_rows": len(df_traffic),
        "weather_rows": len(df_weather),
        "traffy_raw_rows": len(df_traffy),
        "incidents_rows": len(df_incidents),
    }
    log_info(
        f"สร้างข้อมูล Bronze สำเร็จ: ถนน {summary['segments']} ช่วง | "
        f"Traffic {summary['traffic_rows']:,} แถว | Weather {summary['weather_rows']:,} แถว | "
        f"Traffy {summary['traffy_raw_rows']:,} แถว"
    )
    log_success("บันทึกไฟล์ทั้งหมดลง data/01_bronze/ พร้อมอัปเดต SHA256 Manifest เรียบร้อย")
    return summary
