"""Fast, bug-fixed iTIC Historical Traffic Status parser & cleaner for Bangkok (2022-06..2022-09).

Fixes the capture-group off-by-one bug in the previous script where <Quantity> (1/2/3)
was accidentally written to `link_speed_kmh` and <QuantType> (51) caused `congestion_level`
to always be 3.

Supports running all 4 months in parallel (`--parallel`) across multiple CPU cores:
- 2022-06 reads directly from local `data/raw/itic/traffic/2022-06-main.tar`
- 2022-07, 2022-08, 2022-09 stream from `https://itic.longdo.com/opendata/traffic-status-feeds/2022/{month}-main.tar`
- Automatically resumes from the last fixed day if a stream is interrupted.
- Mirrors fixed daily CSVs to `c:\\datasetBKK\\cleandata\\itic_historical_traffic` if that directory exists.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd

from src.common import get_logger, path

log = get_logger("ingest.itic_traffic")

FAST_MSG_RE = re.compile(
    rb"<dateTime>([^<]*)</dateTime><resultOf>[^<]*</resultOf></Preamble>\s*"
    rb"<Location><Version>[^<]*</Version><Segment>"
    rb"<From><locCode>(\d+)</locCode><offset>(\d*)</offset><direction>([a-z]*)</direction></From>"
    rb"<To><locCode>(\d+)</locCode><offset>(\d*)</offset><direction>([a-z]*)</direction></To>"
    rb"</Segment></Location>\s*"
    rb"<Event><EventCode>[^<]*</EventCode><QuantType>[^<]*</QuantType><Quantity>(\d*)</Quantity>"
    rb"<UnitOfMeasure>[^<]*</UnitOfMeasure><LinkSpeed>([^<]+)</LinkSpeed></Event>"
)

CLEAN_HEADER = [
    "snapshot_time",
    "event_time",
    "from_loc",
    "from_offset",
    "from_dir",
    "to_loc",
    "to_offset",
    "to_dir",
    "link_speed_kmh",
    "congestion_level",
    "adm2_pcode",
    "adm2_name",
]

MONTH_DAYS = {
    "2022-06": 30,
    "2022-07": 31,
    "2022-08": 31,
    "2022-09": 30,
}

EXPECTED_MIN_TAR_BYTES = 6_500_000_000
MIRROR_ROOT = Path(r"c:\datasetBKK\cleandata\itic_historical_traffic")


def load_bkk_loc_lookup() -> dict[int, tuple[str, str]]:
    """Load Bangkok loc_code -> (adm2_pcode, adm2_name_en) from itic_location_table_bkk_clean.csv."""
    loc_csv = path("processed") / "csv" / "itic_location_table" / "itic_location_table_bkk_clean.csv"
    if not loc_csv.exists():
        raise FileNotFoundError(f"Missing location table: {loc_csv}")
    df = pd.read_csv(loc_csv)
    return {int(r.loc_code): (str(r.adm2_pcode), str(r.adm2_name_en)) for r in df.itertuples()}


def is_day_already_fixed(csv_path: Path) -> bool:
    """Return True only if csv_path exists AND its link_speed_kmh has real speeds (> 3 km/h)."""
    if not csv_path.exists() or csv_path.stat().st_size < 100_000:
        return False
    try:
        s = pd.read_csv(csv_path, usecols=["link_speed_kmh"], nrows=200)["link_speed_kmh"]
        return bool((s > 3).any())
    except Exception:
        return False


def get_fixed_days(month_dir: Path) -> set[str]:
    if not month_dir.exists():
        return set()
    fixed = set()
    for f in month_dir.iterdir():
        if f.name.startswith("itic_bkk_clean_") and f.name.endswith(".csv"):
            if is_day_already_fixed(f):
                fixed.add(f.name[15:25])
    return fixed


def process_month_once(month: str, lut: dict[int, tuple[str, str]], max_days: int | None = None) -> int:
    month_dir = path("processed") / "csv" / "iTic Traffic" / month
    month_dir.mkdir(parents=True, exist_ok=True)

    for f in month_dir.glob("*.tmp"):
        try:
            f.unlink()
        except OSError:
            pass

    expected_days = MONTH_DAYS.get(month, 30)
    done_days = get_fixed_days(month_dir)
    if len(done_days) >= expected_days:
        log.info("[%s] All %s days already have valid link_speed_kmh, skipping.", month, len(done_days))
        return len(done_days)

    local_tar = path("raw") / "itic" / "traffic" / f"{month}-main.tar"
    use_local = local_tar.exists() and local_tar.stat().st_size >= EXPECTED_MIN_TAR_BYTES
    url = f"https://itic.longdo.com/opendata/traffic-status-feeds/2022/{month}-main.tar"

    if use_local:
        log.info("[%s] Reading from LOCAL file %s (done days: %s/%s)", month, local_tar.name, len(done_days), expected_days)
    else:
        log.info("[%s] Streaming via curl.exe from %s (done days: %s/%s)", month, url, len(done_days), expected_days)

    current_day: str | None = None
    cur_file = None
    cur_tmp_path: Path | None = None
    cur_final_path: Path | None = None
    cur_writer = None
    proc = None

    n_total = n_bkk_raw = n_valid = n_dropped_speed = n_dup = 0
    day_valid = 0
    days_processed_this_run = 0
    t_day_start = time.time()

    def close_current_day(completed: bool = True) -> None:
        nonlocal cur_file, cur_tmp_path, cur_final_path, cur_writer, day_valid, days_processed_this_run, t_day_start, current_day
        if cur_file is not None:
            cur_file.close()
            if completed and cur_tmp_path and cur_tmp_path.exists() and cur_final_path:
                os.replace(cur_tmp_path, cur_final_path)
                if MIRROR_ROOT.exists():
                    mirror_dir = MIRROR_ROOT / month
                    mirror_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(cur_final_path, mirror_dir / cur_final_path.name)
                    except OSError:
                        pass
                if current_day:
                    done_days.add(current_day)
                days_processed_this_run += 1
                sz_mb = cur_final_path.stat().st_size / 1e6
                elapsed = time.time() - t_day_start
                log.info(
                    "[%s] saved %s (%s/%s) | rows=%s (%.1f MB) in %.1fs",
                    month,
                    cur_final_path.name,
                    len(done_days),
                    expected_days,
                    f"{day_valid:,}",
                    sz_mb,
                    elapsed,
                )
            elif not completed and cur_tmp_path and cur_tmp_path.exists():
                try:
                    cur_tmp_path.unlink()
                except OSError:
                    pass
        cur_file = cur_tmp_path = cur_final_path = cur_writer = None
        current_day = None
        day_valid = 0
        t_day_start = time.time()

    try:
        if use_local:
            tar_ctx = tarfile.open(local_tar, mode="r|")
        else:
            proc = subprocess.Popen(
                [
                    "curl.exe",
                    "-sS",
                    "--retry",
                    "5",
                    "--connect-timeout",
                    "30",
                    "--speed-time",
                    "60",
                    "--speed-limit",
                    "10240",
                    url,
                ],
                stdout=subprocess.PIPE,
                bufsize=1 << 20,
            )
            tar_ctx = tarfile.open(fileobj=proc.stdout, mode="r|")

        with tar_ctx as t:
            for m in t:
                if not m.isfile() or not m.name.endswith(".xml.gz"):
                    continue
                m_stamp = re.search(r"(\d{12})-main", m.name)
                if not m_stamp:
                    continue
                stamp = m_stamp.group(1)
                snap = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[8:10]}:{stamp[10:12]}:00"
                day = snap[:10]

                if day in done_days:
                    if current_day is not None:
                        close_current_day(completed=True)
                    continue

                if day != current_day:
                    close_current_day(completed=True)
                    if max_days is not None and days_processed_this_run >= max_days:
                        log.info("[%s] Reached max_days=%s, stopping early.", month, max_days)
                        break
                    current_day = day
                    cur_final_path = month_dir / f"itic_bkk_clean_{day}.csv"
                    cur_tmp_path = month_dir / f"itic_bkk_clean_{day}.csv.tmp"
                    cur_file = open(cur_tmp_path, "w", newline="", encoding="utf-8-sig")
                    cur_writer = csv.writer(cur_file)
                    cur_writer.writerow(CLEAN_HEADER)

                try:
                    fobj = t.extractfile(m)
                    data = gzip.decompress(fobj.read()) if fobj else b""
                except Exception as e:
                    log.warning("[%s] skip corrupt member %s: %s", month, m.name, e)
                    continue

                seen_in_snap = set()
                for g in FAST_MSG_RE.finditer(data):
                    n_total += 1
                    fl, tl = int(g.group(2)), int(g.group(5))
                    hit = lut.get(fl) or lut.get(tl)
                    if not hit:
                        continue
                    n_bkk_raw += 1

                    spd_b = g.group(9)
                    try:
                        spd = float(spd_b)
                    except ValueError:
                        n_dropped_speed += 1
                        continue
                    if spd <= 0.0 or spd > 150.0:
                        n_dropped_speed += 1
                        continue

                    foff = g.group(3).decode("ascii")
                    fdir = g.group(4).decode("ascii")
                    toff = g.group(6).decode("ascii")
                    tdir = g.group(7).decode("ascii")
                    dedup_key = (fl, tl, fdir, foff, toff)
                    if dedup_key in seen_in_snap:
                        n_dup += 1
                        continue
                    seen_in_snap.add(dedup_key)

                    qty_b = g.group(8)
                    qty = int(qty_b) if qty_b in (b"1", b"2", b"3") else (1 if spd >= 30.0 else (2 if spd >= 15.0 else 3))
                    ev_time = g.group(1).decode("ascii", "replace") or snap
                    pcode, dname = hit

                    cur_writer.writerow(
                        [
                            snap,
                            ev_time,
                            fl,
                            foff,
                            fdir,
                            tl,
                            toff,
                            tdir,
                            int(spd) if spd.is_integer() else spd,
                            qty,
                            pcode,
                            dname,
                        ]
                    )
                    n_valid += 1
                    day_valid += 1
            else:
                close_current_day(completed=True)

        if proc:
            proc.terminate()
    except Exception:
        close_current_day(completed=False)
        if proc:
            proc.kill()
        raise
    return len(done_days)


def process_month_with_retry(month: str, max_days: int | None = None) -> str:
    lut = load_bkk_loc_lookup()
    expected = MONTH_DAYS.get(month, 30)
    month_dir = path("processed") / "csv" / "iTic Traffic" / month
    while len(get_fixed_days(month_dir)) < expected:
        try:
            process_month_once(month, lut, max_days=max_days)
            if max_days is not None:
                break
        except Exception as e:
            log.warning("[%s] Stream interrupted (%s), resuming from last completed day in 5s...", month, e)
            time.sleep(5)
    return f"{month}: {len(get_fixed_days(month_dir))}/{expected} days complete"


if __name__ == "__main__":
    args = sys.argv[1:]
    parallel = "--parallel" in args
    max_d = None
    for a in args:
        if a.startswith("--max-days="):
            max_d = int(a.split("=")[1])
    months = [a for a in args if not a.startswith("--")] or ["2022-06", "2022-07", "2022-08", "2022-09"]

    if parallel and len(months) > 1:
        log.info("Launching %s months in parallel: %s", len(months), months)
        with ProcessPoolExecutor(max_workers=len(months)) as ex:
            futs = {ex.submit(process_month_with_retry, mo, max_d): mo for mo in months}
            for fut in as_completed(futs):
                log.info("FINISHED -> %s", fut.result())
    else:
        for mo in months:
            res = process_month_with_retry(mo, max_d)
            log.info("FINISHED -> %s", res)
