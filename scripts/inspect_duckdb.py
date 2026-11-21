"""Inspect and browse tables inside data/bkk_traffic.duckdb (CLI + Interactive Web UI).

Usage:
    python scripts/inspect_duckdb.py                              # ดูสารบัญทั้ง 17 ตาราง + ตัวอย่าง
    python scripts/inspect_duckdb.py silver.dim_segment --rows 15 # เลือกเปิดดูตารางที่ต้องการ
    python scripts/inspect_duckdb.py --sql "SELECT * FROM gold.traffy_district_static_train LIMIT 10"
    python scripts/inspect_duckdb.py --ui                         # เปิดหน้าเว็บ http://localhost:8765
"""
from __future__ import annotations

import argparse
import html
from http.server import BaseHTTPRequestHandler, HTTPServer
import subprocess
import sys
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
if _VENV_PY.exists() and Path(sys.executable).resolve() != _VENV_PY.resolve():
    raise SystemExit(subprocess.call([str(_VENV_PY), *sys.argv]))

import duckdb
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

DB_FILE = ROOT / "data" / "bkk_traffic.duckdb"

ALL_TABLES = [
    "bronze.raw_manifest_log",
    "bronze.itic_bkk_daily_raw",
    "silver.dim_district",
    "silver.itic_location_table",
    "silver.dim_segment",
    "silver.weather_hourly",
    "silver.calendar_daily",
    "silver.traffy_district_15m",
    "silver.traffy_district_hourly",
    "silver.traffy_district_daily",
    "silver.itic_segment_15m",
    "silver.quality_completeness_seg_day",
    "gold.traffy_district_static_train",
    "gold.district_hourly_panel",
    "gold.itic_seg15_targets",
    "gold.merged_district_15m",
    "gold.merged_segment_15m",
]


def show_catalog(con: duckdb.DuckDBPyConnection) -> None:
    print("=" * 85)
    print(f"DUCKDB WAREHOUSE: {DB_FILE} ({DB_FILE.stat().st_size / 1e6:,.2f} MB)")
    print("=" * 85)
    tbls = con.execute(
        """
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema IN ('bronze', 'silver', 'gold')
        ORDER BY CASE table_schema WHEN 'bronze' THEN 1 WHEN 'silver' THEN 2 ELSE 3 END, table_name
        """
    ).fetchall()
    print(f"{'Schema':<10} | {'Table / View Name':<35} | {'Type':<10} | {'Rows':>15}")
    print("-" * 85)
    for sch, name, ttype in tbls:
        if name == "itic_bkk_daily_raw":
            cnt_str = "~200M (122 CSVs)"
        else:
            cnt = con.execute(f"SELECT COUNT(*) FROM {sch}.{name}").fetchone()[0]
            cnt_str = f"{cnt:,}"
        print(f"{sch:<10} | {name:<35} | {ttype:<10} | {cnt_str:>15}")


def launch_web_ui(db_file: Path, port: int = 8765) -> None:
    con_init = duckdb.connect(str(db_file), read_only=True)
    db_tables = [
        f"{s}.{t}"
        for s, t in con_init.execute(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema IN ('bronze', 'silver', 'gold')
            ORDER BY CASE table_schema WHEN 'bronze' THEN 1 WHEN 'silver' THEN 2 ELSE 3 END, table_name
            """
        ).fetchall()
    ]
    con_init.close()
    default_tbl = "gold.merged_district_15m" if "gold.merged_district_15m" in db_tables else (db_tables[0] if db_tables else "")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            tbl = qs.get("table", [default_tbl])[0]
            custom_sql = qs.get("sql", [""])[0].strip()
            limit = int(qs.get("limit", ["50"])[0])

            con = duckdb.connect(str(db_file), read_only=True)
            sql_to_run = custom_sql if custom_sql else f"SELECT * FROM {tbl} LIMIT {limit}"
            err_msg = ""
            df = pd.DataFrame()
            try:
                df = con.execute(sql_to_run).df()
            except Exception as e:
                err_msg = str(e)
            con.close()

            links_html = " ".join(
                f'<a href="/?table={t}&limit={limit}" style="display:inline-block;margin:3px;padding:6px 10px;'
                f'border-radius:6px;text-decoration:none;font-size:13px;'
                f'background:{"#1d4ed8;color:#fff" if t==tbl and not custom_sql else "#e2e8f0;color:#1e293b"}">{t}</a>'
                for t in db_tables
            )
            tbl_html = df.to_html(index=False, border=0, classes="tbl") if not err_msg else f"<pre style='color:red'>{html.escape(err_msg)}</pre>"

            page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>DuckDB Viewer — {db_file.name}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 20px; background: #f8fafc; color: #0f172a; }}
.tbl {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 13px; }}
.tbl th {{ background: #1e293b; color: #fff; padding: 8px 10px; text-align: left; position: sticky; top: 0; }}
.tbl td {{ border-bottom: 1px solid #e2e8f0; padding: 6px 10px; white-space: nowrap; }}
.tbl tr:hover td {{ background: #f1f5f9; }}
.box {{ background: #fff; padding: 14px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 14px; }}
</style></head><body>
<h2>🦆 BKK Traffic — DuckDB Viewer ({db_file.name})</h2>
<div class="box"><b>คลิกเลือกตารางที่ต้องการดู:</b><br>{links_html}</div>
<div class="box">
  <form method="get" action="/">
    <b>รันคำสั่ง SQL ค้นหาข้อมูล:</b><br>
    <input type="text" name="sql" value="{html.escape(sql_to_run)}" style="width:78%;padding:8px;font-family:monospace;margin-top:6px;">
    <button type="submit" style="padding:8px 16px;background:#16a34a;color:#fff;border:none;border-radius:6px;cursor:pointer;">Run SQL</button>
  </form>
</div>
<div class="box" style="overflow-x:auto;">
  <b>ผลลัพธ์ ({len(df):,} แถว × {df.shape[1]} คอลัมน์):</b><br><br>{tbl_html}
</div>
</body></html>"""
            data = page.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    print(f"DuckDB Web Viewer ({db_file.name}) running at: http://localhost:{port}")
    print("Press Ctrl+C to stop.")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main() -> None:
    global DB_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("table", nargs="?", default="", help="Table name e.g. gold.merged_district_15m or silver.dim_segment")
    ap.add_argument("--db", type=str, default=str(DB_FILE), help="Path to DuckDB file (default: data/bkk_traffic.duckdb)")
    ap.add_argument("--rows", type=int, default=10, help="Number of rows to preview")
    ap.add_argument("--sql", type=str, default="", help="Custom SQL query to run")
    ap.add_argument("--ui", action="store_true", help="Launch interactive browser viewer at http://localhost:8765")
    ap.add_argument("--port", type=int, default=8765, help="Port for --ui")
    args = ap.parse_args()
    DB_FILE = Path(args.db)

    if args.ui:
        launch_web_ui(DB_FILE, args.port)
        return

    con = duckdb.connect(str(DB_FILE), read_only=True)
    if args.sql:
        print(f"=== SQL: {args.sql} ===")
        print(con.execute(args.sql).df().to_string(index=False))
    elif args.table:
        cnt = con.execute(f"SELECT COUNT(*) FROM {args.table}").fetchone()[0]
        print(f"=== TABLE: {args.table} (Total rows: {cnt:,}) ===")
        df = con.execute(f"SELECT * FROM {args.table} LIMIT {args.rows}").df()
        print(df.to_string(index=False))
    else:
        show_catalog(con)
    con.close()


if __name__ == "__main__":
    main()
