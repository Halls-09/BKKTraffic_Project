"""
Seaborn + DuckDB Exploratory Data Analysis (EDA) for Bangkok Traffic Congestion
Baseline Visual Style: Chart 06 (Clean Geospatial Traffic-Light Palette + High-Contrast Tahoma Typography)

Primary Data Source: data/bkk_traffic.duckdb
  - 23,840,561 records (15-min slots) | 4,068 road segments | 50 districts | 122 days (Jun-Sep 2022)

Easy Run (from project root):
  python eda.py              # Run all 8 charts + JSON summary + HTML visual gallery
  python eda.py 1 2          # Run only charts 1 and 2
  python eda.py --list       # Show chart menu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
from fontTools.ttLib import TTFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ROOT_DIR = Path(__file__).resolve().parents[2]
DB_PATH = ROOT_DIR / "data" / "bkk_traffic.duckdb"
OUT_DIR = ROOT_DIR / "outputs" / "eda"
CHARTS_DIR = OUT_DIR / "charts"
FONTS_DIR = ROOT_DIR / "src" / "eda" / "fonts"
SUMMARY_PATH = OUT_DIR / "eda_summary.json"
HTML_REPORT_PATH = OUT_DIR / "eda_report.html"

# High-contrast editorial colors (Dark Ink so fonts never sink or fade into background!)
INK = "#0F172A"
NAVY = "#0F2942"
SUB_INK = "#1E293B"
TEAL = "#0D9488"
GREEN = "#15803D"
AMBER = "#D97706"
CRIMSON = "#DC2626"
DARK_RED = "#991B1B"
SLATE = "#334155"

WHITE_HALO = [pe.withStroke(linewidth=3.2, foreground="white")]
DARK_HALO = [pe.withStroke(linewidth=2.5, foreground="#0F172A")]

FOOTER_TEXT = (
    "ที่มา: bkk_traffic.duckdb (iTIC 15-min + Open-Meteo + Traffy Fondue + ปฏิทิน ธปท. | "
    "1 มิ.ย. – 30 ก.ย. 2022 | 23.84M records, 4,068 ช่วงถนน, 50 เขต กทม.)"
)

LANDMARK_DISTRICTS = [
    "คลองเตย", "ราชเทวี", "บางรัก", "บางกะปิ", "จตุจักร",
    "บางนา", "ดอนเมือง", "บางพลัด", "หนองจอก", "มีนบุรี", "บางขุนเทียน", "ธนบุรี",
]


def ensure_bkk_thai_font() -> str:
    """
    Build & register a GPOS-compensated Thai font ('BKKThaiUI') on Windows (from Leelawadee UI),
    or gracefully fall back to installed Thai fonts on macOS/Linux.
    Fixes Matplotlib's FreeType ft2font limitation where Thai tone marks (่ ้ ๊ ๋ ์)
    sink/collide into upper vowels (ิ ี ึ ื ั ำ) when OpenType MarkToMark GPOS is absent.
    """
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    reg_out = FONTS_DIR / "BKKThaiUI-Regular.ttf"
    bold_out = FONTS_DIR / "BKKThaiUI-Bold.ttf"

    if not (reg_out.exists() and bold_out.exists()):
        try:
            reg_src = fm.findfont(fm.FontProperties(family="Leelawadee UI", weight="normal"), fallback_to_default=False)
            bold_src = fm.findfont(fm.FontProperties(family="Leelawadee UI", weight="bold"), fallback_to_default=False)
            for src_file, out_path in [(reg_src, reg_out), (bold_src, bold_out)]:
                tt = TTFont(src_file)
                cmap = tt.getBestCmap()
                glyf = tt["glyf"]
                dy = 470  # Lift tone marks cleanly above upper vowels (yMax ~1509 -> tone yMin ~1580)
                for cp in (0x0E48, 0x0E49, 0x0E4A, 0x0E4B, 0x0E4C):
                    gname = cmap.get(cp)
                    if not gname:
                        continue
                    g = glyf[gname]
                    if g.isComposite():
                        for comp in g.components:
                            comp.y += dy
                    elif g.numberOfContours > 0:
                        coords = g.coordinates
                        for idx in range(len(coords)):
                            x, y = coords[idx]
                            coords[idx] = (x, y + dy)
                    g.recalcBounds(glyf)
                tt["head"].yMax = max(tt["head"].yMax, 2200)
                tt["hhea"].ascent = max(tt["hhea"].ascent, 2200)
                tt["OS/2"].usWinAscent = max(tt["OS/2"].usWinAscent, 2200)
                for rec in tt["name"].names:
                    if rec.nameID in (1, 4, 6, 16):
                        rec.string = "BKKThaiUI"
                tt.save(out_path)
        except Exception:
            # Non-Windows fallback (macOS / Linux)
            for candidate in ("Thonburi", "Noto Sans Thai", "Sarabun", "Tahoma", "Garuda", "Waree"):
                try:
                    fm.findfont(fm.FontProperties(family=candidate), fallback_to_default=False)
                    return candidate
                except Exception:
                    continue
            return "DejaVu Sans"

    if reg_out.exists() and bold_out.exists():
        for fp in (reg_out, bold_out):
            fm.fontManager.addfont(str(fp))
        return "BKKThaiUI"
    return "Tahoma"


def setup_seaborn_style() -> None:
    """Configure high-contrast Seaborn theme with tone-mark-fixed BKKThaiUI typography."""
    thai_font = ensure_bkk_thai_font()
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font=thai_font,
        rc={
            "font.family": thai_font,
            "font.sans-serif": [thai_font, "Leelawadee UI", "Tahoma"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#94A3B8",
            "axes.linewidth": 1.1,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "grid.color": "#CBD5E1",
            "grid.linestyle": "--",
            "grid.alpha": 0.65,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.5,
            "axes.labelweight": "bold",
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "legend.frameon": True,
            "legend.framealpha": 0.96,
            "legend.facecolor": "white",
            "legend.edgecolor": "#94A3B8",
        },
    )
    plt.rcParams["font.family"] = thai_font


def add_header_and_footer(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, fontsize=16.5, fontweight="bold", color=INK, y=0.978)
    fig.text(
        0.5, 0.928, subtitle,
        ha="center", va="top", fontsize=11.5, fontweight="bold", color=SUB_INK,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F1F5F9", edgecolor="#CBD5E1", lw=0.8),
    )
    fig.text(0.5, 0.008, FOOTER_TEXT, ha="center", va="bottom", fontsize=9.2, fontweight="bold", color=SLATE)


def overlay_bkk_landmarks(ax: plt.Axes, landmarks_df: pd.DataFrame) -> None:
    """Add high-contrast opaque district badges with dark borders so labels never sink into map dots."""
    offsets = {
        "ราชเทวี": (-0.068, 0.036),
        "บางรัก": (-0.068, -0.026),
        "คลองเตย": (0.048, -0.030),
        "ธนบุรี": (-0.078, -0.042),
        "บางพลัด": (-0.078, 0.030),
        "จตุจักร": (0.038, 0.030),
        "บางกะปิ": (0.048, 0.020),
        "บางนา": (0.038, -0.032),
        "ดอนเมือง": (-0.015, 0.022),
        "มีนบุรี": (0.015, 0.018),
        "หนองจอก": (-0.010, -0.026),
        "บางขุนเทียน": (0.010, -0.018),
    }
    for _, r in landmarks_df.iterrows():
        name = r["district"]
        lon, lat = float(r["centroid_lon"]), float(r["centroid_lat"])
        dlon, dlat = offsets.get(name, (0.0, 0.0))
        ax.annotate(
            name,
            xy=(lon, lat),
            xytext=(lon + dlon, lat + dlat),
            fontsize=9.8,
            fontweight="bold",
            color=INK,
            ha="center",
            va="center",
            zorder=10,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=INK, alpha=0.98, lw=1.2),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, alpha=0.95) if (dlon != 0 or dlat != 0) else None,
        )


def welch_test(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    ma, mb = float(np.mean(a)), float(np.mean(b))
    return {
        "mean_a_pct": round(ma * 100, 2),
        "mean_b_pct": round(mb * 100, 2),
        "lift_pct": round(((ma - mb) / mb * 100.0) if mb else 0.0, 2),
        "t_stat": round(float(t_stat), 2),
        "p_value": float(p_val),
    }


def get_landmarks(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    quoted = ",".join(f"'{d}'" for d in LANDMARK_DISTRICTS)
    return con.execute(
        f"SELECT district, centroid_lat, centroid_lon FROM silver.dim_district WHERE district IN ({quoted})"
    ).df()


# ------------------------------------------------------------------------------
# CHART 01 (BASELINE FROM CHART 06): Bangkok Geospatial Map — Overall & Rain Lift
# ------------------------------------------------------------------------------
def chart_01_bkk_map_baseline(con: duckdb.DuckDBPyConnection, summary: dict) -> dict:
    geo = con.execute("""
        SELECT
            d.seg_id, d.lat, d.lon, d.district,
            AVG(CASE WHEN m.is_congested THEN 1.0 ELSE 0.0 END) * 100 AS pct_cong,
            AVG(CASE WHEN m.is_rainy_gt_1mm THEN CASE WHEN m.is_congested THEN 1.0 ELSE 0.0 END END) * 100 AS pct_rain,
            AVG(CASE WHEN NOT m.is_rainy_gt_1mm THEN CASE WHEN m.is_congested THEN 1.0 ELSE 0.0 END END) * 100 AS pct_dry
        FROM gold.merged_segment_15m m
        JOIN silver.dim_segment d ON m.seg_id = d.seg_id
        WHERE d.lat IS NOT NULL AND d.lon IS NOT NULL
        GROUP BY 1, 2, 3, 4
    """).df()
    geo["rain_delta"] = geo["pct_rain"] - geo["pct_dry"]
    landmarks = get_landmarks(con)

    geo_cong = geo.sort_values("pct_cong")
    geo_rain = geo.dropna(subset=["rain_delta"]).sort_values("rain_delta")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.5, 7.8))
    add_header_and_footer(
        fig,
        "01. แผนที่ฐานข้อมูล กทม. (Baseline): จุดรถติดสะสม และความอ่อนไหวต่อฝนทั้ง 4,068 ช่วงถนน",
        "กทม. ชั้นใน (21.6%) รถติดหนักกว่าชั้นกลาง (16.9%) และชั้นนอก (13.2%) ถึง +63% | 84.8% ของถนนติดหนักขึ้นเมื่อฝนตก",
    )

    norm1 = mcolors.Normalize(vmin=0, vmax=45)
    sns.scatterplot(
        data=geo_cong, x="lon", y="lat", hue="pct_cong", palette="RdYlGn_r",
        hue_norm=norm1, s=18, alpha=0.9, linewidth=0, legend=False, ax=ax1,
    )
    sm1 = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm1)
    cbar1 = fig.colorbar(sm1, ax=ax1, fraction=0.044, pad=0.025)
    cbar1.set_label("สัดส่วนเวลารถติดเฉลี่ย (%)", fontsize=11, fontweight="bold", color=INK)
    overlay_bkk_landmarks(ax1, landmarks)
    ax1.set_title("A. สัดส่วนเวลารถติดเฉลี่ย 24 ชม. (เขียว = คล่องตัว, แดง = ติดหนัก)", color=INK, pad=10)
    ax1.set_xlabel("Longitude (°E)")
    ax1.set_ylabel("Latitude (°N)")
    ax1.set_aspect("equal", adjustable="datalim")

    norm2 = mcolors.Normalize(vmin=0, vmax=15)
    sns.scatterplot(
        data=geo_rain, x="lon", y="lat", hue="rain_delta", palette="YlOrRd",
        hue_norm=norm2, s=18, alpha=0.9, linewidth=0, legend=False, ax=ax2,
    )
    sm2 = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm2)
    cbar2 = fig.colorbar(sm2, ax=ax2, fraction=0.044, pad=0.025)
    cbar2.set_label("รถติดเพิ่มขึ้นเมื่อฝนตก (+pp)", fontsize=11, fontweight="bold", color=INK)
    overlay_bkk_landmarks(ax2, landmarks)
    ax2.set_title("B. ผลกระทบจากฝนตก > 1 mm/ชม. (ยิ่งเข้ม = ยิ่งติดเพิ่มมาก)", color=INK, pad=10)
    ax2.set_xlabel("Longitude (°E)")
    ax2.set_ylabel("Latitude (°N)")
    ax2.set_aspect("equal", adjustable="datalim")

    plt.tight_layout(rect=[0, 0.035, 1, 0.88])
    out = CHARTS_DIR / "01_bkk_map_baseline.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    summary["map_baseline"] = {
        "n_segments": int(len(geo)),
        "pct_worse_in_rain": round(float((geo_rain["rain_delta"] > 0).mean() * 100), 2),
        "median_rain_delta_pp": round(float(geo_rain["rain_delta"].median()), 2),
        "p90_rain_delta_pp": round(float(geo_rain["rain_delta"].quantile(0.90)), 2),
    }
    return {
        "id": 1,
        "file": out.name,
        "title": "01. แผนที่ฐาน กทม. (Baseline): จุดรถติดสะสม & ผลกระทบฝนตก",
        "insight": "กทม. ชั้นใน (21.6%) ติดหนักกว่าชั้นนอก (13.2%) ถึง +63.1% และ 84.8% ของถนนทั้ง 4,068 ช่วงแย่ลงเมื่อฝนตก (ค่ากลาง +4.55 pp, p90 สูงถึง +13.09 pp)",
    }


# ------------------------------------------------------------------------------
# CHART 02: Extending Chart 06 Baseline — Morning Rush vs Evening Rush on Map
# ------------------------------------------------------------------------------
def chart_02_bkk_map_morning_vs_evening(con: duckdb.DuckDBPyConnection, summary: dict) -> dict:
    geo = con.execute("""
        SELECT
            d.seg_id, d.lat, d.lon,
            AVG(CASE WHEN t.hour BETWEEN 7 AND 8 THEN CASE WHEN t.is_congested THEN 1.0 ELSE 0.0 END END) * 100 AS am_cong,
            AVG(CASE WHEN t.hour BETWEEN 17 AND 18 THEN CASE WHEN t.is_congested THEN 1.0 ELSE 0.0 END END) * 100 AS pm_cong
        FROM gold.itic_seg15_targets t
        JOIN silver.dim_segment d ON t.seg_id = d.seg_id
        JOIN silver.calendar_daily c ON t.date = c.date
        WHERE c.is_working_day = TRUE AND d.lat IS NOT NULL AND d.lon IS NOT NULL
        GROUP BY 1, 2, 3
    """).df()
    landmarks = get_landmarks(con)

    am_avg = float(geo["am_cong"].mean())
    pm_avg = float(geo["pm_cong"].mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.5, 7.8))
    add_header_and_footer(
        fig,
        "02. แผนที่เปรียบเทียบชั่วโมงเร่งด่วนวันทำงาน: เช้า (07:00–09:00) vs เย็น (17:00–19:00)",
        f"ช่วงเย็นวันทำงานติดหนักกว่าช่วงเช้าชัดเจน ({pm_avg:.1f}% vs {am_avg:.1f}%) โดยพื้นที่สีแดงขยายวงกว้างครอบคลุมทั้งชั้นในและวงแหวนชั้นกลาง",
    )

    norm = mcolors.Normalize(vmin=0, vmax=75)
    for ax, col, panel_title in [
        (ax1, "am_cong", f"A. ชั่วโมงเร่งด่วนเช้า 07:00–09:00 น. (เฉลี่ย {am_avg:.1f}%)"),
        (ax2, "pm_cong", f"B. ชั่วโมงเร่งด่วนเย็น 17:00–19:00 น. (เฉลี่ย {pm_avg:.1f}%)"),
    ]:
        sub = geo.dropna(subset=[col]).sort_values(col)
        sns.scatterplot(
            data=sub, x="lon", y="lat", hue=col, palette="RdYlGn_r",
            hue_norm=norm, s=18, alpha=0.9, linewidth=0, legend=False, ax=ax,
        )
        sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=norm)
        cbar = fig.colorbar(sm, ax=ax, fraction=0.044, pad=0.025)
        cbar.set_label("สัดส่วนเวลารถติด (%)", fontsize=11, fontweight="bold", color=INK)
        overlay_bkk_landmarks(ax, landmarks)
        ax.set_title(panel_title, color=INK, pad=10)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout(rect=[0, 0.035, 1, 0.88])
    out = CHARTS_DIR / "02_bkk_map_morning_vs_evening.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    summary["map_rush_comparison"] = {
        "workday_morning_07_09_mean_pct": round(am_avg, 2),
        "workday_evening_17_19_mean_pct": round(pm_avg, 2),
        "evening_minus_morning_pp": round(pm_avg - am_avg, 2),
    }
    return {
        "id": 2,
        "file": out.name,
        "title": "02. แผนที่ชั่วโมงเร่งด่วนวันทำงาน: เช้า vs เย็น",
        "insight": f"ช่วงเย็น 17:00–19:00 น. ({pm_avg:.1f}%) รถติดแผ่วงกว้างกว่าช่วงเช้า ({am_avg:.1f}%) ถึง +{pm_avg - am_avg:.1f} pp โดยเฉพาะแนวสุขุมวิท–รัชดา–พหลโยธิน–ลาดพร้าว",
    }


# ------------------------------------------------------------------------------
# CHART 03: Seaborn Temporal Heatmap + Clean 15-Minute Curves
# ------------------------------------------------------------------------------
def chart_03_temporal_heatmap_and_15m(con: duckdb.DuckDBPyConnection, summary: dict) -> dict:
    dow_labels = ["จันทร์", "อังคาร", "พุธ", "พฤหัสฯ", "ศุกร์", "เสาร์", "อาทิตย์"]

    hm_df = con.execute("""
        SELECT c.dow, t.hour,
               AVG(CASE WHEN t.is_congested THEN 1.0 ELSE 0.0 END) * 100 AS pct_cong
        FROM gold.itic_seg15_targets t
        JOIN silver.calendar_daily c ON t.date = c.date
        WHERE c.is_holiday = FALSE
        GROUP BY 1, 2
    """).df()
    pivot = hm_df.pivot(index="dow", columns="hour", values="pct_cong").reindex(index=range(7), columns=range(24))
    pivot.index = dow_labels

    prof_df = con.execute("""
        SELECT
            CASE
                WHEN c.is_holiday THEN 'วันหยุดนักขัตฤกษ์'
                WHEN c.is_weekend THEN 'วันเสาร์–อาทิตย์'
                WHEN c.is_friday THEN 'วันศุกร์ (ศุกร์แห่งชาติ)'
                ELSE 'วันทำงาน จ.–พฤ.'
            END AS group_label,
            t.hour + t.minute / 60.0 AS time_hr,
            AVG(CASE WHEN t.is_congested THEN 1.0 ELSE 0.0 END) * 100 AS pct_cong,
            AVG(t.speed_kmh) AS avg_speed
        FROM gold.itic_seg15_targets t
        JOIN silver.calendar_daily c ON t.date = c.date
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).df()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.8, 6.6), gridspec_kw={"width_ratios": [1.14, 1]})
    add_header_and_footer(
        fig,
        "03. จังหวะเวลารถติดของกรุงเทพฯ: Heatmap รายชั่วโมง × วัน และกราฟละเอียดทุก 15 นาที",
        "จุดพีคสูงสุดทั้งสัปดาห์คือ 'วันศุกร์ 17:45 น.' (49.0% ติดขัดพร้อมกัน) ขณะที่วันทำงานทั่วไปพีคเช้า 07:45 น. (41.3%) และเย็น 17:45 น. (46.1%)",
    )

    sns.heatmap(
        pivot, cmap="YlOrRd", vmin=0, vmax=52, linewidths=0.5, linecolor="white",
        cbar_kws={"label": "สัดส่วนช่วงถนนติดขัด (%)", "shrink": 0.92}, ax=ax1,
    )
    ax1.set_title("A. Heatmap ความติดขัดรายชั่วโมง (ไม่รวมวันหยุดนักขัตฤกษ์)", color=INK, pad=10)
    ax1.set_xlabel("เวลา (ชั่วโมง)")
    ax1.set_ylabel("")
    ax1.set_xticks(np.arange(0, 24, 2) + 0.5)
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45, ha="right", fontweight="bold")
    ax1.set_yticklabels(dow_labels, rotation=0, fontweight="bold")

    max_pos = np.unravel_index(np.nanargmax(pivot.values), pivot.values.shape)
    peak_val = float(pivot.values[max_pos])
    ax1.add_patch(plt.Rectangle((max_pos[1], max_pos[0]), 1, 1, fill=False, edgecolor=INK, lw=2.8))
    ax1.annotate(
        f"พีคสุด: ศุกร์ 18:00 น. ({peak_val:.1f}%)",
        xy=(max_pos[1] + 0.5, max_pos[0] + 0.5),
        xytext=(max_pos[1] - 7.5, max_pos[0] + 1.8),
        fontsize=10.2, fontweight="bold", color=INK,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor=INK, alpha=0.98, lw=1.3),
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5),
    )

    palette = {
        "วันศุกร์ (ศุกร์แห่งชาติ)": CRIMSON,
        "วันทำงาน จ.–พฤ.": NAVY,
        "วันเสาร์–อาทิตย์": TEAL,
        "วันหยุดนักขัตฤกษ์": AMBER,
    }
    dashes = {
        "วันศุกร์ (ศุกร์แห่งชาติ)": "",
        "วันทำงาน จ.–พฤ.": "",
        "วันเสาร์–อาทิตย์": (4, 2),
        "วันหยุดนักขัตฤกษ์": (2, 2),
    }
    ax2.axvspan(7.0, 9.0, color="#FEF3C7", alpha=0.55)
    ax2.axvspan(17.0, 19.0, color="#FEE2E2", alpha=0.55)
    sns.lineplot(
        data=prof_df, x="time_hr", y="pct_cong", hue="group_label", style="group_label",
        palette=palette, dashes=dashes, linewidth=2.6, ax=ax2,
    )
    ax2.set_title("B. กราฟความติดขัดละเอียดทุก 15 นาที แยกประเภทวัน", color=INK, pad=10)
    ax2.set_xlabel("เวลา (ชั่วโมง)")
    ax2.set_ylabel("สัดส่วนช่วงถนนติดขัด (%)")
    ax2.set_xticks(range(0, 25, 2))
    ax2.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], rotation=35, ha="right", fontweight="bold")
    ax2.set_ylim(0, 60)
    ax2.legend(title="", loc="upper left", frameon=True)
    ax2.text(
        8.0, 56.5, "พีคเช้า 07:45 น.\n(41.3%)", ha="center", va="top",
        fontsize=9.8, fontweight="bold", color=INK,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=AMBER, alpha=0.98, lw=1.3),
    )
    ax2.text(
        18.0, 56.5, "พีคเย็น 17:45 น.\n(ศุกร์ 49.0%)", ha="center", va="top",
        fontsize=9.8, fontweight="bold", color=INK,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=CRIMSON, alpha=0.98, lw=1.3),
    )
    sns.despine(ax=ax2)

    plt.tight_layout(rect=[0, 0.04, 1, 0.88])
    out = CHARTS_DIR / "03_temporal_heatmap_and_15m.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    summary["temporal"] = {
        "heatmap_peak_non_holiday": {"dow": dow_labels[max_pos[0]], "hour": int(max_pos[1]), "pct": round(peak_val, 2)},
        "friday_1745_pct": round(float(prof_df[prof_df["group_label"] == "วันศุกร์ (ศุกร์แห่งชาติ)"]["pct_cong"].max()), 2),
        "mon_thu_1745_pct": round(float(prof_df[prof_df["group_label"] == "วันทำงาน จ.–พฤ."]["pct_cong"].max()), 2),
    }
    return {
        "id": 3,
        "file": out.name,
        "title": "03. จังหวะเวลารถติดรายชั่วโมง & ละเอียดทุก 15 นาที",
        "insight": f"เมื่อแยกวันหยุดนักขัตฤกษ์ออก เห็นชัดเจนว่า '{dow_labels[max_pos[0]]} {max_pos[1]:02d}:00 น.' ({peak_val:.1f}%) คือช่วงที่รถติดหนักที่สุดในสัปดาห์ และพีคสุดที่ 17:45 น. (49.0%)",
    }


# ------------------------------------------------------------------------------
# CHART 04: Inbound vs Outbound Tidal Flow (Seaborn)
# ------------------------------------------------------------------------------
def chart_04_inbound_outbound_tidal(con: duckdb.DuckDBPyConnection, summary: dict) -> dict:
    tidal = con.execute("""
        SELECT
            CASE WHEN t.inbound_outbound = 'inbound' THEN 'ขาเข้าเมือง (Inbound)' ELSE 'ขาออกเมือง (Outbound)' END AS direction,
            t.hour + t.minute / 60.0 AS time_hr,
            AVG(CASE WHEN t.is_congested THEN 1.0 ELSE 0.0 END) * 100 AS pct_cong,
            AVG(t.speed_kmh) AS avg_speed
        FROM gold.itic_seg15_targets t
        JOIN silver.calendar_daily c ON t.date = c.date
        WHERE c.is_working_day = TRUE AND t.inbound_outbound IN ('inbound', 'outbound')
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).df()

    inb = tidal[tidal["direction"].str.contains("Inbound")].set_index("time_hr")
    outb = tidal[tidal["direction"].str.contains("Outbound")].set_index("time_hr")
    diff = inb["pct_cong"] - outb["pct_cong"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6.2))
    add_header_and_footer(
        fig,
        "04. รูปแบบการไหลตามทิศทางวันทำงาน: ขาเข้าเมือง (Inbound) vs ขาออกเมือง (Outbound)",
        "ช่วงเช้า 07–09 น. ขาออกติดกว่า (+5.3 pp) แต่ช่วงเย็น 17–19 น. พลิกกลับเป็นขาเข้าติดกว่าถึง +6.1 pp สะท้อนการเดินทางข้ามศูนย์กลางเมือง",
    )

    sns.lineplot(
        data=tidal, x="time_hr", y="pct_cong", hue="direction",
        palette={"ขาเข้าเมือง (Inbound)": CRIMSON, "ขาออกเมือง (Outbound)": TEAL},
        linewidth=2.8, ax=ax1,
    )
    ax1.set_title("A. สัดส่วนรถติด (%) ราย 15 นาที เปรียบเทียบสองทิศทาง", color=INK, pad=10)
    ax1.set_xlabel("เวลา (ชั่วโมง)")
    ax1.set_ylabel("สัดส่วนช่วงถนนติดขัด (%)")
    ax1.set_xticks(range(0, 25, 2))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], rotation=35, ha="right", fontweight="bold")
    ax1.legend(title="", loc="upper left")
    sns.despine(ax=ax1)

    ax2.axhline(0, color=INK, lw=1.4)
    ax2.fill_between(diff.index, diff.values, 0, where=(diff.values >= 0), color=CRIMSON, alpha=0.35, label="ขาเข้าติดกว่า (Inbound > Outbound)")
    ax2.fill_between(diff.index, diff.values, 0, where=(diff.values < 0), color=TEAL, alpha=0.35, label="ขาออกติดกว่า (Outbound > Inbound)")
    ax2.plot(diff.index, diff.values, color=INK, lw=2.2)
    ax2.set_title("B. ส่วนต่างความติดขัดสุทธิ (Inbound - Outbound, pp)", color=INK, pad=10)
    ax2.set_xlabel("เวลา (ชั่วโมง)")
    ax2.set_ylabel("ส่วนต่าง (percentage points)")
    ax2.set_xticks(range(0, 25, 2))
    ax2.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], rotation=35, ha="right", fontweight="bold")
    ax2.legend(loc="upper left")
    sns.despine(ax=ax2)

    plt.tight_layout(rect=[0, 0.04, 1, 0.88])
    out = CHARTS_DIR / "04_inbound_outbound_tidal.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    summary["tidal_flow"] = {
        "am_inbound_pct": round(float(inb.loc[7.0:9.0, "pct_cong"].mean()), 2),
        "am_outbound_pct": round(float(outb.loc[7.0:9.0, "pct_cong"].mean()), 2),
        "pm_inbound_pct": round(float(inb.loc[17.0:19.0, "pct_cong"].mean()), 2),
        "pm_outbound_pct": round(float(outb.loc[17.0:19.0, "pct_cong"].mean()), 2),
    }
    return {
        "id": 4,
        "file": out.name,
        "title": "04. ทิศทางรถติดวันทำงาน: ขาเข้า vs ขาออกเมือง",
        "insight": "ความไม่สมมาตรระหว่างทิศทางสูงถึง 5–6 pp ในชั่วโมงเร่งด่วน ทำให้ตัวแปรทิศทาง (inbound/outbound) มีความสำคัญต่อการวางแผนเส้นทาง",
    }


# ------------------------------------------------------------------------------
# CHART 05: 122-Day Monsoon Timeline (Congestion + Rain + Traffy Floods)
# ------------------------------------------------------------------------------
def chart_05_monsoon_timeline(con: duckdb.DuckDBPyConnection, summary: dict) -> dict:
    daily = con.execute("""
        SELECT
            CAST(date AS DATE) AS dt,
            AVG(CASE WHEN itic_active_segments > 0 THEN itic_pct_congested END) * 100 AS pct_cong,
            AVG(CASE WHEN itic_active_segments > 0 THEN itic_avg_speed_kmh END) AS avg_speed,
            SUM(precip_mm) / (50.0 * 4.0) AS daily_precip_mm,
            SUM(traffy_flood_15m) AS daily_flood_reports,
            MAX(CASE WHEN is_holiday THEN 1 ELSE 0 END) AS is_holiday
        FROM gold.merged_district_15m
        GROUP BY 1 ORDER BY 1
    """).df()
    daily["ma7"] = daily["pct_cong"].rolling(7, min_periods=1).mean()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16.5, 9.8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0, 1.15]})
    add_header_and_footer(
        fig,
        "05. แนวโน้มตลอด 122 วัน (มิ.ย.–ก.ย. 2022): รถติด vs ฝนสะสม vs วิกฤตน้ำท่วม Traffy Fondue",
        "หลังตัดสล็อตเซ็นเซอร์ล่ม วันน้ำท่วมใหญ่ 21 ก.ค. (1,333 เรื่อง, พายุถล่มช่วงดึก 21–03 น.) รถติดพุ่งถึง 22.2% และพีคสุดทั้งปีคือ 8 ก.ย. (25.5%)",
    )

    sns.lineplot(data=daily, x="dt", y="pct_cong", color="#64748B", lw=1.4, label="ค่าเฉลี่ยรายวัน (%)", ax=ax1)
    sns.lineplot(data=daily, x="dt", y="ma7", color=NAVY, lw=2.8, label="ค่าเฉลี่ยเคลื่อนที่ 7 วัน", ax=ax1)
    hol = daily[daily["is_holiday"] == 1]
    ax1.scatter(hol["dt"], hol["pct_cong"], color=AMBER, s=65, zorder=5, edgecolor=INK, lw=0.8, label="วันหยุดนักขัตฤกษ์")
    top3 = daily.nlargest(3, "pct_cong")
    ax1.scatter(top3["dt"], top3["pct_cong"], color=CRIMSON, s=85, zorder=6, edgecolor="white", lw=1.2, label="3 วันที่รถติดหนักที่สุด")
    for _, r in top3.iterrows():
        ax1.annotate(
            f"{r['dt'].strftime('%d %b')} ({r['pct_cong']:.1f}%)",
            xy=(r["dt"], r["pct_cong"]), xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=9.8, fontweight="bold", color=INK,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=CRIMSON, alpha=0.98, lw=1.3),
        )
    ax1.set_title("A. สัดส่วนรถติดเฉลี่ยรายวันทั้ง 50 เขต กทม. (เฉพาะสล็อตที่เซ็นเซอร์ทำงานปกติ)", color=INK, pad=6)
    ax1.set_ylabel("รถติดเฉลี่ย (%)")
    ax1.set_ylim(0, daily["pct_cong"].max() * 1.28)
    ax1.legend(loc="upper left", ncol=4)
    sns.despine(ax=ax1)

    ax2.bar(daily["dt"], daily["daily_precip_mm"], color="#0284C7", alpha=0.9, width=0.8)
    ax2.set_title("B. ปริมาณฝนสะสมเฉลี่ยรายวัน (Open-Meteo, mm/วัน)", color=INK, pad=6)
    ax2.set_ylabel("ฝนสะสม (mm)")
    sns.despine(ax=ax2)

    ax3.bar(daily["dt"], daily["daily_flood_reports"], color=CRIMSON, alpha=0.88, width=0.8)
    pf = daily.loc[daily["daily_flood_reports"].idxmax()]
    ax3.annotate(
        f"พายุฝนคืน 20–21 Jul: น้ำท่วม {int(pf['daily_flood_reports']):,} เรื่อง! (รถติด 22.2%)",
        xy=(pf["dt"], pf["daily_flood_reports"]), xytext=(20, -8), textcoords="offset points",
        fontsize=10, fontweight="bold", color=INK,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor=CRIMSON, alpha=0.98, lw=1.4),
        arrowprops=dict(arrowstyle="-|>", color=CRIMSON, lw=1.5),
    )
    ax3.set_title("C. จำนวนเรื่องแจ้งน้ำท่วมจากประชาชน (Traffy Fondue, เรื่อง/วัน)", color=INK, pad=6)
    ax3.set_ylabel("รายงานน้ำท่วม (เรื่อง)")
    ax3.set_xlabel("วันที่ (2022)")
    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    sns.despine(ax=ax3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.895])
    out = CHARTS_DIR / "05_monsoon_timeline_122d.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    summary["worst_days"] = [
        {"date": r["dt"].strftime("%Y-%m-%d"), "pct_cong": round(float(r["pct_cong"]), 2),
         "precip_mm": round(float(r["daily_precip_mm"]), 2), "flood_reports": int(r["daily_flood_reports"])}
        for _, r in daily.nlargest(5, "pct_cong").iterrows()
    ]
    return {
        "id": 5,
        "file": out.name,
        "title": "05. แนวโน้มรายวัน 122 วัน: รถติด vs ฝน vs น้ำท่วม Traffy",
        "insight": "หลังแก้บั๊กสล็อตเซ็นเซอร์ล่ม (itic_active_segments = 0) วันน้ำท่วมสูงสุด 21 ก.ค. (1,333 เรื่อง, พายุถล่มตอนดึก) รถติดพุ่งขึ้นมาที่ 22.2% และสูงสุดของปีคือ 8 ก.ย. (25.5%)",
    }


# ------------------------------------------------------------------------------
# CHART 06: Top Districts & Clean Chronic Bottlenecks
# ------------------------------------------------------------------------------
def chart_06_districts_and_bottlenecks(con: duckdb.DuckDBPyConnection, summary: dict) -> dict:
    dist_df = con.execute("""
        SELECT district,
               AVG(CASE WHEN itic_active_segments > 0 THEN itic_pct_congested END) * 100 AS pct_cong,
               AVG(CASE WHEN itic_active_segments > 0 THEN itic_avg_speed_kmh END) AS avg_speed,
               SUM(traffy_flood_15m) AS total_floods
        FROM gold.merged_district_15m
        GROUP BY 1 ORDER BY pct_cong DESC
    """).df()

    dist_show = pd.concat([dist_df.head(12), dist_df.tail(3)], ignore_index=True)

    seg_df = con.execute("""
        SELECT
            t.seg_id,
            COALESCE(d.road_name_th, 'ไม่ระบุถนน') AS road,
            COALESCE(d.from_place_th, '') AS from_p,
            COALESCE(d.to_place_th, '') AS to_p,
            d.district,
            AVG(CASE WHEN t.is_congested THEN 1.0 ELSE 0.0 END) * 100 AS pct_cong,
            AVG(t.speed_kmh) AS avg_speed
        FROM gold.itic_seg15_targets t
        JOIN silver.dim_segment d ON t.seg_id = d.seg_id
        WHERE t.from_loc != t.to_loc
        GROUP BY 1, 2, 3, 4, 5
        HAVING COUNT(*) >= 5000
        ORDER BY pct_cong DESC
        LIMIT 10
    """).df()
    seg_df["clean_label"] = seg_df.apply(
        lambda r: (f"{r['road']} [{r['district']}]\n({r['from_p'][:20]} -> {r['to_p'][:20]})"
                   if r["from_p"] else f"{r['road']} [{r['district']}]"),
        axis=1,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.5, 7.8))
    add_header_and_footer(
        fig,
        "06. อันดับเขตที่รถติดที่สุด–คล่องตัวที่สุด และ Top 10 จุดคอขวดเรื้อรังของกรุงเทพฯ",
        "กทม. ชั้นใน (คลองเตย 31.3%, ราชเทวี 30.6%) ติดหนักกว่าชั้นนอก (หนองจอก 8.5%) ถึง 3.7 เท่า | ถนนรางน้ำ & ด่านประชาชื่น ติดขัดเกิน 80%",
    )

    cmap = plt.get_cmap("RdYlGn_r")
    norm_d = mcolors.Normalize(vmin=6, vmax=34)
    dist_colors = [cmap(norm_d(v)) for v in dist_show["pct_cong"]]
    sns.barplot(
        data=dist_show, y="district", x="pct_cong", palette=dist_colors,
        hue="district", edgecolor=INK, linewidth=0.6, legend=False, ax=ax1,
    )
    ax1.axhline(11.5, color=INK, linestyle="--", lw=1.5)
    ax1.text(
        41.5, 12.15, "3 เขตชั้นนอกที่คล่องตัวที่สุด",
        fontsize=9.8, color=INK, fontweight="bold", ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#DCFCE7", edgecolor=GREEN, lw=1.1),
    )
    ax1.set_title("A. 12 เขตที่รถติดที่สุด vs 3 เขตที่คล่องตัวที่สุด (จาก 50 เขต)", color=INK, pad=10)
    ax1.set_xlabel("สัดส่วนช่วงถนนติดขัดเฉลี่ย (%)")
    ax1.set_ylabel("")
    ax1.set_xlim(0, 43)
    ax1.tick_params(axis="y", labelsize=10.8)
    for i, r in dist_show.iterrows():
        ax1.text(
            r["pct_cong"] + 0.6, i, f"{r['pct_cong']:.1f}% ({r['avg_speed']:.0f} km/h)",
            va="center", fontsize=10, fontweight="bold", color=INK, path_effects=WHITE_HALO,
        )
    sns.despine(ax=ax1)

    norm_s = mcolors.Normalize(vmin=50, vmax=85)
    seg_colors = [plt.get_cmap("YlOrRd")(norm_s(v)) for v in seg_df["pct_cong"]]
    sns.barplot(
        data=seg_df, y="clean_label", x="pct_cong", palette=seg_colors,
        hue="clean_label", edgecolor=INK, linewidth=0.6, legend=False, ax=ax2,
    )
    ax2.set_title("B. Top 10 ช่วงถนนคอขวดเรื้อรังของกรุงเทพฯ", color=INK, pad=10)
    ax2.set_xlabel("สัดส่วนเวลาที่ติดขัดตลอด 24 ชม. (%)")
    ax2.set_ylabel("")
    ax2.set_xlim(0, 98)
    ax2.tick_params(axis="y", labelsize=9.8)
    for i, r in seg_df.iterrows():
        ax2.text(
            r["pct_cong"] + 0.8, i, f"{r['pct_cong']:.1f}% ({r['avg_speed']:.0f} km/h)",
            va="center", fontsize=10, fontweight="bold", color=INK, path_effects=WHITE_HALO,
        )
    sns.despine(ax=ax2)

    plt.tight_layout(rect=[0, 0.04, 1, 0.88])
    out = CHARTS_DIR / "06_top_districts_and_bottlenecks.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    summary["top_districts"] = dist_df.head(5).round(2).to_dict(orient="records")
    summary["top_bottlenecks_clean"] = seg_df[["seg_id", "road", "district", "pct_cong", "avg_speed"]].round(2).to_dict(orient="records")
    return {
        "id": 6,
        "file": out.name,
        "title": "06. อันดับเขตและจุดคอขวดวิกฤตของกรุงเทพฯ",
        "insight": "กทม. ชั้นในติดหนักกว่าชั้นนอกถึง 3.7 เท่า (คลองเตย 31.3% vs หนองจอก 8.5%) โดยมีจุดคอขวดอันดับ 1 คือ ถนนรางน้ำ (80.9%) และทางพิเศษศรีรัช ด่านประชาชื่น (80.2%)",
    }


# ------------------------------------------------------------------------------
# CHART 07: Causal Chain — Rain Intensity (Rush vs Off-Peak) -> Flood Factors -> Traffic
# ------------------------------------------------------------------------------
def chart_07_rain_and_traffy_impact(con: duckdb.DuckDBPyConnection, summary: dict) -> dict:
    df_d = con.execute("""
        SELECT rain_bin, is_rainy_gt_1mm, is_rush_hour, train_flood_per_sq_km,
               traffy_flood_15m, itic_pct_congested, itic_avg_speed_kmh
        FROM gold.merged_district_15m
        WHERE itic_pct_congested IS NOT NULL AND itic_active_segments > 0
    """).df()

    # Panel A: Rain Intensity split by Rush Hour vs Off-Peak (resolves Simpson's Paradox!)
    bin_map = {
        "0_none": "ไม่ตก\n(0)",
        "0_0.5_trace": "ปรอยๆ\n(0–0.5)",
        "0.5_2": "เบา\n(0.5–2)",
        "2_5": "ปานกลาง\n(2–5)",
        "5_10": "หนัก\n(5–10)",
        "10_20": "หนักมาก\n(10–20)",
    }
    order_a = list(bin_map.values())
    df_d["rain_lbl"] = df_d["rain_bin"].map(bin_map)
    df_d["period_lbl"] = np.where(df_d["is_rush_hour"], "ชั่วโมงเร่งด่วน (Rush Hour)", "นอกเร่งด่วน (Off-Peak)")
    rb = df_d.groupby(["rain_lbl", "period_lbl"], as_index=False)["itic_pct_congested"].mean()
    rb["pct_cong"] = rb["itic_pct_congested"] * 100

    # Panel B: Rain x Flood Risk (with clear explanation of Median 8.11/km2)
    med_flood = df_d["train_flood_per_sq_km"].median()
    df_d["flood_group"] = np.where(
        df_d["train_flood_per_sq_km"] >= med_flood,
        "เขตเสี่ยงท่วมสูง\n(>= 8.1 เรื่อง/km²)",
        "เขตเสี่ยงท่วมต่ำ\n(< 8.1 เรื่อง/km²)",
    )
    df_d["weather_grp"] = np.where(df_d["is_rainy_gt_1mm"], "ฝนตก (> 1 mm)", "ฝนไม่ตก (<= 1 mm)")
    inter = df_d.groupby(["flood_group", "weather_grp"], as_index=False)["itic_pct_congested"].mean()
    inter["pct_cong"] = inter["itic_pct_congested"] * 100

    # Panel C: Real-time Traffy Flood 15m
    df_d["rt_flood"] = np.where(
        df_d["traffy_flood_15m"] == 0, "ไม่มีแจ้ง\n(0 เรื่อง)",
        np.where(df_d["traffy_flood_15m"] == 1, "แจ้งน้ำท่วม\n(1 เรื่อง)", "น้ำท่วมหนัก\n(>=2 เรื่อง)"),
    )
    order_c = ["ไม่มีแจ้ง\n(0 เรื่อง)", "แจ้งน้ำท่วม\n(1 เรื่อง)", "น้ำท่วมหนัก\n(>=2 เรื่อง)"]
    rt = df_d.groupby("rt_flood", as_index=False).agg(
        pct_cong=("itic_pct_congested", lambda s: s.mean() * 100),
        avg_speed=("itic_avg_speed_kmh", "mean"),
    )
    rt["rt_flood"] = pd.Categorical(rt["rt_flood"], categories=order_c, ordered=True)
    rt = rt.sort_values("rt_flood")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.5, 6.4))
    add_header_and_footer(
        fig,
        "07. ห่วงโซ่ผลกระทบ: ความแรงฝน (A) -> ปะทะเขตเสี่ยงน้ำท่วม (B) -> เกิดน้ำท่วมขังจริงฉุดรถติดพุ่ง (C)",
        "เมื่อแยกช่วงเวลา ยิ่งฝนแรงรถยิ่งติดชัดเจน (พีคสุด 39.2% ตอนฝนหนัก >10 mm) -> ทวีคูณในเขตเสี่ยงน้ำท่วม -> ดันรถติดพุ่งถึง 33.7% เมื่อมีแจ้งน้ำท่วมสด",
    )

    # Panel A: Grouped bar showing monotonic increase in both Rush Hour & Off-Peak!
    sns.barplot(
        data=rb, x="rain_lbl", y="pct_cong", hue="period_lbl", order=order_a,
        hue_order=["ชั่วโมงเร่งด่วน (Rush Hour)", "นอกเร่งด่วน (Off-Peak)"],
        palette={"ชั่วโมงเร่งด่วน (Rush Hour)": CRIMSON, "นอกเร่งด่วน (Off-Peak)": TEAL},
        edgecolor=INK, linewidth=0.6, ax=ax1,
    )
    ax1.set_title("A. ยิ่งฝนแรง รถยิ่งติด (ทั้งเร่งด่วน & นอกเร่งด่วน)", color=INK, pad=10)
    ax1.set_xlabel("ระดับความแรงฝน (mm/ชม.)")
    ax1.set_ylabel("สัดส่วนช่วงถนนติดขัด (%)")
    ax1.set_ylim(0, 46)
    ax1.legend(title="", loc="upper left", fontsize=9.5)
    for c in ax1.containers:
        ax1.bar_label(c, fmt="%.0f%%", padding=3, fontsize=8.8, fontweight="bold", color=INK)
    sns.despine(ax=ax1)

    # Panel B
    sns.barplot(
        data=inter, x="flood_group", y="pct_cong", hue="weather_grp",
        order=["เขตเสี่ยงท่วมต่ำ\n(< 8.1 เรื่อง/km²)", "เขตเสี่ยงท่วมสูง\n(>= 8.1 เรื่อง/km²)"],
        hue_order=["ฝนไม่ตก (<= 1 mm)", "ฝนตก (> 1 mm)"],
        palette={"ฝนไม่ตก (<= 1 mm)": TEAL, "ฝนตก (> 1 mm)": CRIMSON},
        edgecolor=INK, linewidth=0.6, ax=ax2,
    )
    ax2.set_title("B. ปฏิสัมพันธ์ ฝน × พื้นที่เสี่ยงน้ำท่วม", color=INK, pad=10)
    ax2.set_xlabel("กลุ่มความเสี่ยงน้ำท่วมของเขต (Traffy/km²)")
    ax2.set_ylabel("สัดส่วนช่วงถนนติดขัด (%)")
    ax2.set_ylim(0, 36)
    ax2.legend(title="", loc="upper left", fontsize=9.5)
    for c in ax2.containers:
        ax2.bar_label(c, fmt="%.1f%%", padding=4, fontsize=10, fontweight="bold", color=INK)
    sns.despine(ax=ax2)

    # Panel C
    sns.barplot(
        data=rt, x="rt_flood", y="pct_cong", palette=[TEAL, AMBER, CRIMSON],
        hue="rt_flood", edgecolor=INK, linewidth=0.6, legend=False, ax=ax3,
    )
    ax3.set_title("C. ผลของน้ำท่วมขังจริง (แจ้ง Traffy สด 15น.)", color=INK, pad=10)
    ax3.set_xlabel("จำนวนแจ้งน้ำท่วมสดในเขต (ต่อ 15 นาที)")
    ax3.set_ylabel("สัดส่วนช่วงถนนติดขัด (%)")
    ax3.set_ylim(0, 42)
    for i, r in enumerate(rt.itertuples()):
        ax3.text(
            i, r.pct_cong + 0.8, f"{r.pct_cong:.1f}%\n({r.avg_speed:.1f} km/h)",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK, path_effects=WHITE_HALO,
        )
    sns.despine(ax=ax3)

    plt.tight_layout(rect=[0, 0.04, 1, 0.88])
    out = CHARTS_DIR / "07_rain_and_traffy_flood_impact.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    summary["weather_tests"] = {
        "rain_vs_dry": welch_test(
            df_d.loc[df_d["is_rainy_gt_1mm"], "itic_pct_congested"].to_numpy(),
            df_d.loc[~df_d["is_rainy_gt_1mm"], "itic_pct_congested"].to_numpy(),
        ),
        "traffy_flood_gt0_vs_0": welch_test(
            df_d.loc[df_d["traffy_flood_15m"] > 0, "itic_pct_congested"].to_numpy(),
            df_d.loc[df_d["traffy_flood_15m"] == 0, "itic_pct_congested"].to_numpy(),
        ),
    }
    return {
        "id": 7,
        "file": out.name,
        "title": "07. ห่วงโซ่ผลกระทบ: ความแรงฝน -> เขตเสี่ยงน้ำท่วม -> น้ำท่วมขังสด Traffy",
        "insight": "เมื่อแยกชั่วโมงเร่งด่วนกับนอกเร่งด่วน ยิ่งฝนตกแรงรถยิ่งติดเพิ่มขึ้นแบบทางเดียว (พีค 39.2% ที่ฝน 10–20 mm/ชม.) และเมื่อเกิดน้ำท่วมขังจนมีแจ้ง Traffy >=2 เรื่อง/15น. รถติดพุ่งถึง 33.7%",
    }


# ------------------------------------------------------------------------------
# CHART 08: Calendar/Payday Ranking + High-Contrast Correlation Heatmap
# ------------------------------------------------------------------------------
def chart_08_calendar_and_correlation(con: duckdb.DuckDBPyConnection, summary: dict) -> dict:
    df_c = con.execute("""
        WITH payday_weeks AS (
            SELECT DISTINCT YEARWEEK(CAST(date AS DATE)) AS yw
            FROM silver.calendar_daily WHERE is_payday = TRUE
        )
        SELECT
            m.itic_pct_congested, m.itic_avg_speed_kmh, m.hour,
            m.precip_mm, m.precip_3h_mm, m.traffy_flood_15m,
            m.train_flood_per_sq_km, m.train_traffic_per_sq_km, m.temp_c,
            CAST(m.is_rush_hour AS INT) AS is_rush_hour,
            c.day_type, c.is_working_day, c.is_friday, c.is_holiday,
            c.is_weekend, c.is_payday, c.is_gov_payday, c.is_payday_25, c.is_month_end_workday,
            (YEARWEEK(CAST(c.date AS DATE)) IN (SELECT yw FROM payday_weeks)) AS is_payday_week
        FROM gold.merged_district_15m m
        JOIN silver.calendar_daily c ON m.date = c.date
        WHERE m.itic_pct_congested IS NOT NULL AND m.itic_active_segments > 0
    """).df()

    eve = df_c[df_c["hour"].between(16, 19)]
    eve_wd = eve[eve["is_working_day"]]
    cal_rows = [
        ("เย็นศุกร์ ในสัปดาห์เงินเดือนออก (จ.-พฤ. Payday)", eve_wd[eve_wd["is_friday"] & eve_wd["is_payday_week"] & (~eve_wd["is_payday"])]["itic_pct_congested"].mean() * 100, DARK_RED),
        ("เย็นวันทำงานสิ้นเดือน (Month-End)", eve_wd[eve_wd["is_month_end_workday"]]["itic_pct_congested"].mean() * 100, CRIMSON),
        ("เย็นวันศุกร์ทั้งหมด (รวมทุกศุกร์ทำงาน)", eve_wd[eve_wd["is_friday"]]["itic_pct_congested"].mean() * 100, "#EA580C"),
        ("เย็นวันทำงานอื่นที่เป็น Payday (จ.–พฤ.)", eve_wd[(~eve_wd["is_friday"]) & eve_wd["is_payday"]]["itic_pct_congested"].mean() * 100, AMBER),
        ("เย็นวันทำงานปกติ (จ.–พฤ. ไม่ใช่ Payday)", eve_wd[(~eve_wd["is_friday"]) & (~eve_wd["is_payday"])]["itic_pct_congested"].mean() * 100, NAVY),
        ("เย็นวันเสาร์–อาทิตย์", eve[eve["day_type"] == "weekend"]["itic_pct_congested"].mean() * 100, TEAL),
        ("เย็นวันหยุดนักขัตฤกษ์", eve[eve["day_type"] == "holiday"]["itic_pct_congested"].mean() * 100, GREEN),
    ]
    cal_df = pd.DataFrame(cal_rows, columns=["scenario", "pct_cong", "color"]).sort_values("pct_cong", ascending=False)

    corr_cols = [
        "itic_pct_congested", "itic_avg_speed_kmh", "is_rush_hour",
        "train_traffic_per_sq_km", "train_flood_per_sq_km",
        "precip_3h_mm", "precip_mm", "traffy_flood_15m",
        "is_weekend", "is_holiday",
    ]
    corr_labels = [
        "% รถติด", "ความเร็วเฉลี่ย", "ชั่วโมงเร่งด่วน",
        "จราจรหนาแน่น (เขต)", "เสี่ยงน้ำท่วม (เขต)",
        "ฝนสะสม 3 ชม.", "ฝนชั่วโมงปัจจุบัน", "แจ้งน้ำท่วมสด 15น.",
        "เสาร์–อาทิตย์", "วันหยุดนักขัตฤกษ์",
    ]
    corr_mat = df_c[corr_cols].corr()
    corr_mat.index = corr_labels
    corr_mat.columns = corr_labels

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.5, 7.6), gridspec_kw={"width_ratios": [1.05, 1.15]})
    add_header_and_footer(
        fig,
        "08. พฤติกรรมช่วงเย็นตามปฏิทิน (ศุกร์สัปดาห์เงินเดือนออก & วันหยุด) และเมทริกซ์สหสัมพันธ์",
        "เย็นวันศุกร์ในสัปดาห์ที่เงินเดือนออกพุ่งสูงสุดถึง 45.3% | ฝนสะสม 3 ชม. (r = +0.18) ส่งผลต่อรถติดแรงกว่าฝนปัจจุบัน (+0.12) ถึง +45%",
    )

    sns.barplot(
        data=cal_df, y="scenario", x="pct_cong",
        palette=list(cal_df["color"]), hue="scenario", edgecolor=INK, linewidth=0.6, legend=False, ax=ax1,
    )
    ax1.set_title("A. เปรียบเทียบรถติดช่วงเย็น (16:00–19:45 น.) แยกตามประเภทวัน", color=INK, pad=10)
    ax1.set_xlabel("สัดส่วนช่วงถนนติดขัดเฉลี่ย (%)")
    ax1.set_ylabel("")
    ax1.set_xlim(0, 53)
    ax1.tick_params(axis="y", labelsize=10.2)
    for i, r in enumerate(cal_df.itertuples()):
        ax1.text(
            r.pct_cong + 0.6, i, f"{r.pct_cong:.1f}%",
            va="center", fontsize=10.5, fontweight="bold", color=INK, path_effects=WHITE_HALO,
        )
    sns.despine(ax=ax1)

    # Right: Render correlation heatmap WITHOUT automatic low-contrast annotations, then draw high-contrast text manually!
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    sns.heatmap(
        corr_mat, mask=mask, cmap="RdBu_r", vmin=-0.85, vmax=0.85, center=0,
        annot=False, linewidths=0.8, linecolor="white", square=True,
        cbar_kws={"label": "Pearson r", "shrink": 0.82}, ax=ax2,
    )
    n = len(corr_labels)
    for i in range(n):
        for j in range(i + 1):
            val = float(corr_mat.iloc[i, j])
            if abs(val) >= 0.58:
                txt_col = "white"
                effects = DARK_HALO
            else:
                txt_col = INK
                effects = WHITE_HALO
            ax2.text(
                j + 0.5, i + 0.5, f"{val:.2f}",
                ha="center", va="center", fontsize=9.8, fontweight="bold",
                color=txt_col, path_effects=effects,
            )

    ax2.set_title("B. เมทริกซ์สหสัมพันธ์ของ 10 ปัจจัยหลัก", color=INK, pad=10)
    ax2.set_xticklabels(corr_labels, rotation=38, ha="right", fontsize=10, fontweight="bold", color=INK)
    ax2.set_yticklabels(corr_labels, rotation=0, fontsize=10, fontweight="bold", color=INK)

    plt.tight_layout(rect=[0, 0.04, 1, 0.88])
    out = CHARTS_DIR / "08_calendar_payday_and_correlation.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    summary["calendar_evening_ranking"] = cal_df[["scenario", "pct_cong"]].round(2).to_dict(orient="records")
    summary["top_correlations"] = {k: round(float(v), 4) for k, v in corr_mat["% รถติด"].drop("% รถติด").items()}
    return {
        "id": 8,
        "file": out.name,
        "title": "08. อิทธิพลศุกร์สัปดาห์เงินเดือนออก วันหยุด และสหสัมพันธ์ 10 ปัจจัยหลัก",
        "insight": "เย็นวันศุกร์ในสัปดาห์ที่เงินเดือนออกพุ่งสูงสุดถึง 44.8% ตามด้วยวันทำงานสิ้นเดือน (41.8%) และศุกร์ทั่วไป (41.0%) สูงกว่าเย็นวันทำงานปกติ (37.4%) ชัดเจน",
    }


# ------------------------------------------------------------------------------
# CHART 09: Individual Feature Distributions Grid (12 Features + Log1p Transform)
# ------------------------------------------------------------------------------
def chart_09_statistical_distributions(con: duckdb.DuckDBPyConnection, summary: dict) -> dict:
    # Sample 150k rows from 23.84M segment table for fast, exact KDE + Histogram rendering
    df_seg = con.execute("""
        SELECT speed_kmh, free_flow_kmh, LEAST(ratio, 1.6) AS ratio
        FROM gold.itic_seg15_targets
        USING SAMPLE 150000 ROWS (reservoir, 42)
    """).df()

    df_len = con.execute("SELECT length_m FROM silver.dim_segment WHERE length_m > 0").df()

    df_d = con.execute("""
        SELECT itic_pct_congested * 100 AS cong_pct,
               itic_avg_speed_kmh AS dist_speed,
               temp_c, precip_mm, precip_3h_mm, precip_6h_mm,
               traffy_total_15m, traffy_flood_15m, traffy_road_15m,
               train_flood_per_sq_km, train_traffic_per_sq_km, area_sq_km
        FROM gold.merged_district_15m
        WHERE itic_active_segments > 0
    """).df()

    df_dist_static = con.execute("""
        SELECT area_sq_km, train_flood_per_sq_km, train_traffic_per_sq_km,
               train_flood_reports, train_road_reports
        FROM gold.traffy_district_static_train
    """).df()

    # Define 12 individual feature panels:
    # Row 1-2: 6 Naturally Bounded / Near-Symmetric Features (Raw Scale)
    # Row 3-4: 6 Right-Skewed Features (Log1p Transformed: log(1 + x), showing Raw Skew -> Log1p Skew!)
    panels = [
        # (series, title, xlabel, color, is_log, raw_series_for_skew)
        (df_seg["speed_kmh"].dropna(), "1. ความเร็วรถจริง (speed_kmh)", "km/h (Raw Scale)", TEAL, False, None),
        (df_seg["free_flow_kmh"].dropna(), "2. ความเร็วรถโล่ง (free_flow_kmh)", "km/h (Raw Scale)", NAVY, False, None),
        (df_seg["ratio"].dropna(), "3. สัดส่วนความเร็ว (Speed Ratio)", "speed / free_flow (Raw)", "#0284C7", False, None),
        (df_d["cong_pct"].dropna(), "4. สัดส่วนรถติดระดับเขต (% Congested)", "% รถติด (Raw Scale)", CRIMSON, False, None),
        (df_d["dist_speed"].dropna(), "5. ความเร็วเฉลี่ยระดับเขต (dist_speed)", "km/h (Raw Scale)", GREEN, False, None),
        (df_d["temp_c"].dropna(), "6. อุณหภูมิรายชั่วโมง (temp_c)", "°C (Raw Scale)", AMBER, False, None),
        # Row 3 & 4: Log1p Transformed features
        (
            np.log1p(df_len["length_m"]),
            "7. ความยาวช่วงถนน [Log Transform]",
            "log(1 + length_m)",
            NAVY, True, df_len["length_m"],
        ),
        (
            np.log1p(df_dist_static["area_sq_km"]),
            "8. ขนาดพื้นที่เขต [Log Transform]",
            "log(1 + area_sq_km)",
            TEAL, True, df_dist_static["area_sq_km"],
        ),
        (
            np.log1p(df_dist_static["train_flood_per_sq_km"]),
            "9. ความหนาแน่นน้ำท่วมเขต [Log Transform]",
            "log(1 + flood_per_sq_km)",
            CRIMSON, True, df_dist_static["train_flood_per_sq_km"],
        ),
        (
            np.log1p(df_dist_static["train_traffic_per_sq_km"]),
            "10. ความหนาแน่นจราจรเขต [Log Transform]",
            "log(1 + traffic_per_sq_km)",
            "#EA580C", True, df_dist_static["train_traffic_per_sq_km"],
        ),
        (
            np.log1p(df_d.loc[df_d["precip_3h_mm"] > 0, "precip_3h_mm"]),
            "11. ฝนสะสม 3 ชม. เมื่อมีฝน [Log Transform]",
            "log(1 + precip_3h_mm)",
            "#0284C7", True, df_d.loc[df_d["precip_3h_mm"] > 0, "precip_3h_mm"],
        ),
        (
            np.log1p(df_d.loc[df_d["precip_6h_mm"] > 0, "precip_6h_mm"]),
            "12. ฝนสะสม 6 ชม. เมื่อมีฝน [Log Transform]",
            "log(1 + precip_6h_mm)",
            DARK_RED, True, df_d.loc[df_d["precip_6h_mm"] > 0, "precip_6h_mm"],
        ),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(18.8, 15.2))
    add_header_and_footer(
        fig,
        "09. การกระจายตัวของแต่ละฟีเจอร์ (12 Individual Feature Distributions) และผลของการทำ Log Transform",
        "แถว 1–2: ฟีเจอร์ความเร็วและอุณหภูมิมีการกระจายใกล้เคียงปกติ | แถว 3–4: ฟีเจอร์ที่เบ้ขวาจัด เมื่อแปลงด้วย log(1+x) ค่าความเบ้ (Skewness) ลดลงเข้าใกล้ 0 อย่างชัดเจน",
    )

    feat_stats = []
    for ax, (s, title, xlabel, col, is_log, raw_s) in zip(axes.flat, panels):
        sns.histplot(
            s, bins=28, kde=True, stat="density", color=col, alpha=0.55,
            edgecolor=INK, linewidth=0.45, line_kws={"linewidth": 2.2}, ax=ax,
        )
        mean_v = float(s.mean())
        med_v = float(s.median())
        skew_v = float(s.skew())
        ax.axvline(mean_v, color=DARK_RED, linestyle="--", lw=1.8, label=f"Mean: {mean_v:.2f}")
        ax.axvline(med_v, color=INK, linestyle="-", lw=1.8, label=f"Med: {med_v:.2f}")

        if is_log and raw_s is not None:
            raw_skew = float(raw_s.skew())
            badge_txt = f"Raw Skew: +{raw_skew:.2f}\nLog1p Skew: {skew_v:+.2f}"
            badge_bg = "#FEF2F2"
            badge_edge = CRIMSON
        else:
            badge_txt = f"Mean: {mean_v:.1f} | Med: {med_v:.1f}\nSkewness: {skew_v:+.2f}"
            badge_bg = "#F8FAFC"
            badge_edge = NAVY

        ax.text(
            0.97, 0.93, badge_txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=9.2, fontweight="bold", color=INK,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=badge_bg, edgecolor=badge_edge, alpha=0.96, lw=1.1),
        )
        ax.set_title(title, color=INK, fontsize=12, pad=7)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        sns.despine(ax=ax)

        feat_stats.append({
            "feature": title,
            "is_log1p": is_log,
            "mean": round(mean_v, 3),
            "median": round(med_v, 3),
            "skewness": round(skew_v, 3),
            "raw_skewness": round(float(raw_s.skew()), 3) if raw_s is not None else round(skew_v, 3),
        })

    plt.tight_layout(rect=[0, 0.025, 1, 0.91])
    out = CHARTS_DIR / "09_statistical_distributions_and_skewness.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    summary["individual_feature_distributions"] = feat_stats
    return {
        "id": 9,
        "file": out.name,
        "title": "09. การกระจายตัวของแต่ละฟีเจอร์ (12 Features) & การทำ Log Transform",
        "insight": "การแปลง log(1+x) ช่วยลดความเบ้ขวา (Skewness) ของตัวแปรพื้นที่ ถนน น้ำท่วม และฝนสะสมลงอย่างชัดเจน เช่น area_sq_km (+3.24 -> +0.31), flood_per_sq_km (+1.00 -> -0.20), precip_6h (+2.04 -> +0.52)",
    }


# ------------------------------------------------------------------------------
# CHART 10: Flood Factors Regression & District Area Denominator Bias
# ------------------------------------------------------------------------------
def chart_10_flood_factors_and_area_bias(con: duckdb.DuckDBPyConnection, summary: dict) -> dict:
    df_f = con.execute("""
        WITH seg_agg AS (
            SELECT district, SUM(length_m)/1000.0 AS road_km
            FROM silver.dim_segment GROUP BY 1
        ),
        m_agg AS (
            SELECT district,
                   MAX(area_sq_km) AS area_sq_km,
                   MAX(weather_point) AS weather_point,
                   SUM(traffy_flood_15m) AS total_flood,
                   SUM(traffy_road_15m) AS total_road,
                   MAX(train_flood_per_sq_km) AS flood_per_sqkm
            FROM gold.merged_district_15m GROUP BY 1
        )
        SELECT m.*, COALESCE(s.road_km, 1.0) AS road_km,
               m.total_flood / COALESCE(s.road_km, 1.0) AS flood_per_road_km
        FROM m_agg m LEFT JOIN seg_agg s USING (district)
    """).df()

    med_density = float(df_f["flood_per_sqkm"].median())
    df_f["tier_lbl"] = np.where(
        df_f["flood_per_sqkm"] >= med_density,
        "จัดเป็น 'ท่วมสูง' (>= 8.1/km²)",
        "ถูกจัดเป็น 'ท่วมต่ำ' (< 8.1/km²)",
    )
    r_road = float(df_f["total_road"].corr(df_f["total_flood"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.2, 6.6))
    add_header_and_footer(
        fig,
        "10. วิเคราะห์การกระจายตัวของปัจจัยน้ำท่วม (Road Defects r = +0.67) และอคติขนาดพื้นที่เขต (Area Bias)",
        "ปัญหาโครงสร้างถนน (traffy_road) สัมพันธ์กับปริมาณน้ำท่วมสูงสุด (r = +0.67) | เขตพื้นที่ใหญ่ (>120 km²) เช่น บางขุนเทียน–ลาดกระบัง ถูกจัดเป็นเสี่ยงต่ำเพราะตัวหารพื้นที่ใหญ่",
    )

    # Panel A: Seaborn regplot (Road complaints vs Flood complaints)
    sns.regplot(
        data=df_f, x="total_road", y="total_flood",
        scatter_kws={"s": 70, "color": CRIMSON, "alpha": 0.88, "edgecolor": INK},
        line_kws={"color": NAVY, "lw": 2.4, "label": f"Linear Fit (Pearson r = +{r_road:.2f})"},
        ax=ax1,
    )
    for _, r in df_f.nlargest(7, "total_flood").iterrows():
        ax1.annotate(
            f"{r['district']} ({int(r['total_flood'])})",
            xy=(r["total_road"], r["total_flood"]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=9.5, fontweight="bold", color=INK, path_effects=WHITE_HALO,
        )
    ax1.set_title("A. ความสัมพันธ์ระหว่างเรื่องแจ้งปัญหาถนน vs น้ำท่วม (50 เขต)", color=INK, pad=10)
    ax1.set_xlabel("จำนวนเรื่องร้องเรียนปัญหาถนน/ท่อริมถนน (traffy_road, เรื่อง)")
    ax1.set_ylabel("จำนวนเรื่องร้องเรียนน้ำท่วมรวม (traffy_flood, เรื่อง)")
    ax1.legend(loc="upper left")
    sns.despine(ax=ax1)

    # Panel B: Area vs Total Flood showing Denominator Bias
    sns.scatterplot(
        data=df_f, x="area_sq_km", y="total_flood", hue="tier_lbl",
        palette={"จัดเป็น 'ท่วมสูง' (>= 8.1/km²)": CRIMSON, "ถูกจัดเป็น 'ท่วมต่ำ' (< 8.1/km²)": TEAL},
        s=85, alpha=0.9, edgecolor=INK, linewidth=0.9, ax=ax2,
    )
    # Plot the threshold line: total_flood_train ≈ 8.11 * area_sq_km (scaled to full 122d period ~1.4x)
    x_line = np.linspace(0, 245, 100)
    ax2.plot(x_line, x_line * med_density * 1.35, color=NAVY, linestyle="--", lw=1.8, label="เส้นแบ่งเกณฑ์ความหนาแน่น (8.1 เรื่อง/km²)")

    highlight_dists = ["ประเวศ", "บางกะปิ", "สวนหลวง", "วังทองหลาง", "บางขุนเทียน", "ลาดกระบัง", "คลองสามวา", "หนองจอก"]
    for _, r in df_f[df_f["district"].isin(highlight_dists)].iterrows():
        ax2.annotate(
            f"{r['district']} ({int(r['total_flood'])} เรื่อง)",
            xy=(r["area_sq_km"], r["total_flood"]),
            xytext=(6, 5), textcoords="offset points",
            fontsize=9.3, fontweight="bold", color=INK, path_effects=WHITE_HALO,
        )
    ax2.set_title("B. อคติตัวหารพื้นที่เขต (Area Bias: เขตใหญ่ถูกปัดเป็น 'ท่วมต่ำ')", color=INK, pad=10)
    ax2.set_xlabel("ขนาดพื้นที่เขต (area_sq_km, ตร.กม. — Skewness +3.24)")
    ax2.set_ylabel("จำนวนเรื่องร้องเรียนน้ำท่วมรวม (เรื่อง)")
    ax2.legend(title="", loc="upper right", fontsize=9.5)
    sns.despine(ax=ax2)

    plt.tight_layout(rect=[0, 0.04, 1, 0.88])
    out = CHARTS_DIR / "10_flood_factors_and_area_bias.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    summary["flood_factor_correlations"] = {
        "road_complaints_vs_total_flood_r": round(r_road, 3),
        "area_vs_flood_per_sqkm_r": round(float(df_f["area_sq_km"].corr(df_f["flood_per_sqkm"])), 3),
        "area_vs_flood_per_road_km_r": round(float(df_f["area_sq_km"].corr(df_f["flood_per_road_km"])), 3),
    }
    return {
        "id": 10,
        "file": out.name,
        "title": "10. การกระจายตัวของปัจจัยน้ำท่วม & อคติขนาดพื้นที่เขต (Area Bias)",
        "insight": "ปัญหาโครงสร้างถนน (traffy_road) สัมพันธ์กับน้ำท่วมสูงสุด (r = +0.67) และกราฟ B พิสูจน์ว่า บางขุนเทียน (507 เรื่อง) กับลาดกระบัง (432 เรื่อง) ถูกจัดเป็นเสี่ยงต่ำเพราะพื้นที่เขตใหญ่เกิน 120 ตร.กม.",
    }


# ------------------------------------------------------------------------------
# CHART 11: Data Quality, Speed Jump Outliers & Train/Valid/Test Distribution Shift
# ------------------------------------------------------------------------------
def chart_11_quality_outliers_and_split_shift(con: duckdb.DuckDBPyConnection, summary: dict) -> dict:
    comp_seg = con.execute("""
        SELECT seg_id,
               AVG(completeness_all) * 100 AS comp_all,
               AVG(completeness_peak) * 100 AS comp_peak
        FROM silver.quality_completeness_seg_day
        GROUP BY 1
    """).df()

    jump_hist = con.execute("""
        SELECT CAST(FLOOR(GREATEST(LEAST(speed_kmh - prev_15m_speed_kmh, 80), -80) / 5.0) * 5 AS DOUBLE) AS bin,
               COUNT(*) / 1e6 AS cnt_m
        FROM gold.itic_seg15_targets
        WHERE speed_kmh IS NOT NULL AND prev_15m_speed_kmh IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).df()

    split_df = con.execute("""
        SELECT
            CASE split
                WHEN 'train' THEN '1. Train\n(มิ.ย.–23 ส.ค.)'
                WHEN 'valid' THEN '2. Valid\n(24 ส.ค.–6 ก.ย.)'
                ELSE '3. Test\n(7–30 ก.ย.)'
            END AS split_lbl,
            AVG(CASE WHEN itic_active_segments > 0 THEN itic_pct_congested END) * 100 AS cong_pct,
            AVG(CASE WHEN is_rainy_gt_1mm THEN 1.0 ELSE 0.0 END) * 100 AS rainy_pct,
            AVG(traffy_flood_15m) * 100 AS flood_rate
        FROM gold.merged_district_15m
        GROUP BY 1 ORDER BY 1
    """).df()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.8, 6.4))
    add_header_and_footer(
        fig,
        "11. การกระจายความครบถ้วนของข้อมูล (Completeness), Outliers การกระโดดความเร็ว และ Distribution Shift",
        "ความครบถ้วนช่วงพีคสูงกว่าทั้งวัน (Median 64% vs 53%) | ความเร็วกระโดดเกิน ±60 km/h มีเพียง 0.57% | ชุด Test เดือน ก.ย. มีชั่วโมงฝนตกพุ่ง +75% จากชุด Train",
    )

    # Panel A: Segment completeness KDE / Histogram
    sns.histplot(comp_seg["comp_peak"], bins=25, color=NAVY, alpha=0.65, kde=True, label="ช่วงเร่งด่วน (Peak Completeness)", ax=ax1)
    sns.histplot(comp_seg["comp_all"], bins=25, color=TEAL, alpha=0.55, kde=True, label="ตลอด 24 ชม. (All-Day Completeness)", ax=ax1)
    ax1.axvline(80, color=CRIMSON, lw=2.0, linestyle="--", label="เกณฑ์คุณภาพ 80%")
    ax1.set_title("A. การกระจายความครบถ้วนข้อมูลของ 4,068 ช่วงถนน", color=INK, pad=10)
    ax1.set_xlabel("ความครบถ้วนของสล็อตเวลา (%)")
    ax1.set_ylabel("จำนวนช่วงถนน (Segments)")
    ax1.legend(loc="upper left", fontsize=9.2)
    sns.despine(ax=ax1)

    # Panel B: Speed change (delta 15m) distribution & outlier tails
    j_cols = [CRIMSON if abs(b + 2.5) >= 60 else TEAL for b in jump_hist["bin"]]
    ax2.bar(jump_hist["bin"] + 2.5, jump_hist["cnt_m"], width=4.4, color=j_cols, alpha=0.88, edgecolor=INK, lw=0.4)
    ax2.axvline(-60, color=CRIMSON, lw=2.0, linestyle="--", label="Outlier Threshold (±60 km/h = 0.57%)")
    ax2.axvline(60, color=CRIMSON, lw=2.0, linestyle="--")
    ax2.set_title("B. การกระจายค่าความเร็วเปลี่ยนแปลงใน 15 นาที (Speed Change)", color=INK, pad=10)
    ax2.set_xlabel("Speed Change = speed(t) - speed(t-15m) (km/h)")
    ax2.set_ylabel("จำนวนระเบียน (ล้านแถว)")
    ax2.legend(loc="upper right", fontsize=9.2)
    sns.despine(ax=ax2)

    # Panel C: Train vs Valid vs Test Distribution Shift
    s_melt = split_df.melt(
        id_vars="split_lbl",
        value_vars=["cong_pct", "rainy_pct"],
        var_name="metric",
        value_name="val",
    )
    s_melt["metric_lbl"] = s_melt["metric"].map({
        "cong_pct": "อัตรารถติดเฉลี่ย (%)",
        "rainy_pct": "สัดส่วนชั่วโมงฝนตก >1mm (%)",
    })
    sns.barplot(
        data=s_melt, x="split_lbl", y="val", hue="metric_lbl",
        palette={"อัตรารถติดเฉลี่ย (%)": NAVY, "สัดส่วนชั่วโมงฝนตก >1mm (%)": CRIMSON},
        edgecolor=INK, linewidth=0.6, ax=ax3,
    )
    ax3.set_title("C. Distribution Shift: ชุด Train vs Valid vs Test", color=INK, pad=10)
    ax3.set_xlabel("")
    ax3.set_ylabel("สัดส่วน (%)")
    ax3.set_ylim(0, 27)
    ax3.legend(title="", loc="upper left", fontsize=9.2)
    for c in ax3.containers:
        ax3.bar_label(c, fmt="%.1f%%", padding=3, fontsize=9.5, fontweight="bold", color=INK)
    sns.despine(ax=ax3)

    plt.tight_layout(rect=[0, 0.04, 1, 0.88])
    out = CHARTS_DIR / "11_data_quality_outliers_and_split_shift.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    summary["split_shift"] = split_df.round(2).to_dict(orient="records")
    return {
        "id": 11,
        "file": out.name,
        "title": "11. คุณภาพข้อมูล Outliers และ Distribution Shift (Train vs Valid vs Test)",
        "insight": "ชุดข้อมูล Test (ก.ย.) มีชั่วโมงฝนตกพุ่งสูงถึง 13.3% (เพิ่มขึ้น +75% จากชุด Train ที่ 7.6%) ยืนยันว่าเกิด Covariate Shift ตามฤดูกาลที่ต้องใช้ฟีเจอร์ฝนและน้ำท่วมช่วยในการพยากรณ์",
    }


def build_html_gallery(cards: list[dict], summary: dict) -> None:
    ov = summary.get("overview", {})
    card_html = []
    for c in cards:
        card_html.append(f"""
        <section class="card">
          <h2>{c['title']}</h2>
          <p class="insight">💡 <strong>Key Insight:</strong> {c['insight']}</p>
          <a href="charts/{c['file']}" target="_blank">
            <img src="charts/{c['file']}" alt="{c['title']}" loading="lazy" />
          </a>
        </section>
        """)

    html = f"""<!DOCTYPE html>
<html lang="th">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Bangkok Traffic EDA — Visual Insights Gallery</title>
  <style>
    :root {{ --navy: #0F2942; --teal: #0D9488; --bg: #F8FAFC; --card: #FFFFFF; --muted: #334155; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: 'Tahoma', 'Leelawadee UI', sans-serif; background: var(--bg); color: #0F172A; line-height: 1.6; }}
    header {{ background: var(--navy); color: white; padding: 28px 24px; text-align: center; }}
    header h1 {{ margin: 0 0 8px; font-size: 1.8rem; }}
    header p {{ margin: 0; opacity: 0.92; font-size: 1rem; }}
    .kpis {{ max-width: 1280px; margin: 20px auto 8px; padding: 0 20px; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; }}
    .kpi {{ background: var(--card); border-radius: 12px; padding: 16px 18px; border: 1px solid #CBD5E1; text-align: center; box-shadow: 0 1px 3px rgba(15,41,66,0.06); }}
    .kpi .val {{ font-size: 1.55rem; font-weight: 800; color: var(--navy); }}
    .kpi .lbl {{ font-size: 0.9rem; color: var(--muted); margin-top: 4px; font-weight: 600; }}
    main {{ max-width: 1280px; margin: 16px auto 48px; padding: 0 20px; display: flex; flex-direction: column; gap: 24px; }}
    .card {{ background: var(--card); border-radius: 14px; padding: 22px 24px; border: 1px solid #CBD5E1; box-shadow: 0 2px 8px rgba(15,41,66,0.05); }}
    .card h2 {{ margin: 0 0 8px; color: var(--navy); font-size: 1.25rem; }}
    .insight {{ margin: 0 0 16px; padding: 10px 14px; background: #F0FDFA; border-left: 4px solid var(--teal); border-radius: 6px; color: #0F172A; font-size: 0.98rem; }}
    .card img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid #CBD5E1; display: block; }}
    footer {{ text-align: center; color: var(--muted); font-size: 0.88rem; padding: 0 20px 32px; font-weight: 600; }}
  </style>
</head>
<body>
  <header>
    <h1>🚦 Bangkok Traffic Congestion — Seaborn EDA Gallery</h1>
    <p>ฐานข้อมูลหลัก: bkk_traffic.duckdb (1 มิ.ย. – 30 ก.ย. 2022 | คลิกที่กราฟเพื่อเปิดดูขนาดเต็ม)</p>
  </header>
  <div class="kpis">
    <div class="kpi"><div class="val">{ov.get('total_rows', 23840561):,}</div><div class="lbl">ระเบียนจราจร (15 นาที)</div></div>
    <div class="kpi"><div class="val">{ov.get('n_segments', 4068):,} ช่วง / {ov.get('n_districts', 50)} เขต</div><div class="lbl">ความครอบคลุม กทม.</div></div>
    <div class="kpi"><div class="val">{ov.get('avg_speed_kmh', 38.0):.1f} km/h</div><div class="lbl">ความเร็วเฉลี่ยรวม</div></div>
    <div class="kpi"><div class="val">{ov.get('pct_congested', 23.1):.1f}%</div><div class="lbl">สัดส่วนเวลารถติดเฉลี่ย</div></div>
  </div>
  <main>
    {''.join(card_html)}
  </main>
  <footer>สร้างอัตโนมัติด้วย DuckDB + Seaborn (รันซ้ำได้ด้วย <code>python eda.py</code>)</footer>
</body>
</html>
"""
    HTML_REPORT_PATH.write_text(html, encoding="utf-8")


CHART_REGISTRY = [
    (1, "แผนที่ฐาน กทม. (Baseline รูป 06): รถติดสะสม & ผลกระทบฝน", chart_01_bkk_map_baseline),
    (2, "แผนที่ชั่วโมงเร่งด่วนวันทำงาน: เช้า vs เย็น", chart_02_bkk_map_morning_vs_evening),
    (3, "Heatmap รายชั่วโมง × วัน & กราฟละเอียดทุก 15 นาที", chart_03_temporal_heatmap_and_15m),
    (4, "ทิศทางรถติดวันทำงาน: ขาเข้า vs ขาออกเมือง", chart_04_inbound_outbound_tidal),
    (5, "แนวโน้มรายวัน 122 วัน: รถติด vs ฝน vs น้ำท่วม Traffy", chart_05_monsoon_timeline),
    (6, "อันดับเขตที่รถติดที่สุด–คล่องตัวสุด & Top 10 คอขวด", chart_06_districts_and_bottlenecks),
    (7, "ห่วงโซ่ผลกระทบ: ความแรงฝน -> เขตเสี่ยงท่วม -> Traffy สด", chart_07_rain_and_traffy_impact),
    (8, "ปฏิทิน (ศุกร์สัปดาห์เงินเดือนออก) & เมทริกซ์สหสัมพันธ์ 10 ปัจจัย", chart_08_calendar_and_correlation),
    (9, "การกระจายตัวของแต่ละฟีเจอร์ (12 Features) & Log Transform", chart_09_statistical_distributions),
    (10, "การกระจายตัวของปัจจัยน้ำท่วม (Regplot) & อคติขนาดพื้นที่เขต", chart_10_flood_factors_and_area_bias),
    (11, "ความครบถ้วนข้อมูล (Completeness), Outliers & Train/Test Split Shift", chart_11_quality_outliers_and_split_shift),
]


def run_eda(selected_ids: list[int] | None = None) -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"\n❌ ไม่พบฐานข้อมูล DuckDB ที่: {DB_PATH}\n"
            f"👉 กรุณาวางไฟล์ 'bkk_traffic.duckdb' ไว้ในโฟลเดอร์ 'data/' แล้วรัน 'python eda.py' ใหม่อีกครั้งครับ\n"
        )

    setup_seaborn_style()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    if not selected_ids:
        for old_png in CHARTS_DIR.glob("*.png"):
            old_png.unlink()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    ov = con.execute("""
        SELECT COUNT(*) AS total_rows, COUNT(DISTINCT seg_id) AS n_segments,
               COUNT(DISTINCT district) AS n_districts, AVG(speed_kmh) AS avg_speed_kmh,
               AVG(CASE WHEN is_congested THEN 1.0 ELSE 0.0 END) * 100 AS pct_congested
        FROM gold.itic_seg15_targets
    """).df().iloc[0]

    summary: dict = {
        "source_db": DB_PATH.name,
        "overview": {
            "total_rows": int(ov["total_rows"]),
            "n_segments": int(ov["n_segments"]),
            "n_districts": int(ov["n_districts"]),
            "avg_speed_kmh": round(float(ov["avg_speed_kmh"]), 2),
            "pct_congested": round(float(ov["pct_congested"]), 2),
        },
    }

    cards: list[dict] = []
    targets = CHART_REGISTRY if not selected_ids else [c for c in CHART_REGISTRY if c[0] in selected_ids]
    total_n = len(CHART_REGISTRY)
    print(f"\n🚀 Running Seaborn EDA on {DB_PATH.name} ({len(targets)} charts)...\n")
    for cid, desc, fn in targets:
        print(f"  [{cid}/{total_n}] {desc}...")
        cards.append(fn(con, summary))

    con.close()
    if not selected_ids:
        SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        build_html_gallery(cards, summary)

    print(f"\n✅ เสร็จเรียบร้อย!")
    print(f"   📊 กราฟทั้งหมด: {CHARTS_DIR}")
    print(f"   🌐 หน้าเว็บดูกราฟรวม: {HTML_REPORT_PATH}")
    print(f"   📄 สรุปสถิติ JSON: {SUMMARY_PATH}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bangkok Traffic Seaborn EDA Runner (DuckDB)")
    parser.add_argument("charts", nargs="*", type=int, help="เลือกหมายเลขกราฟที่ต้องการรัน เช่น 1 2 6 (เว้นว่าง = รันทั้งหมด)")
    parser.add_argument("--list", action="store_true", help="แสดงรายการกราฟทั้งหมด")
    args = parser.parse_args()

    if args.list:
        print("\n📋 รายการกราฟ EDA ทั้งหมด (รันด้วย: python eda.py [หมายเลข]):")
        for cid, desc, _ in CHART_REGISTRY:
            print(f"  {cid}: {desc}")
        print()
        return

    run_eda(args.charts if args.charts else None)


if __name__ == "__main__":
    main()
