"""Run the BKK Traffic Data Engineering pipeline with one command.

Usage:
    python run_pipeline.py                    # run all stages end-to-end
    python run_pipeline.py --stage check      # verify folders & raw checksum manifest
    python run_pipeline.py --stage weather    # stage Open-Meteo weather table
    python run_pipeline.py --stage calendar   # build 2022 Bangkok calendar table
    python run_pipeline.py --stage traffy     # clean & aggregate Traffy Fondue + 50 districts
    python run_pipeline.py --stage panel      # build complete 50-district x 2,928-hour panel
    python run_pipeline.py --stage quality    # run all Quality Gate & Leakage checks
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Auto-switch to the project's .venv interpreter if launched with global Python
_VENV_PY = Path(__file__).resolve().parent / ".venv" / "Scripts" / "python.exe"
if _VENV_PY.exists() and Path(sys.executable).resolve() != _VENV_PY.resolve():
    raise SystemExit(subprocess.call([str(_VENV_PY), *sys.argv]))

from src.common import CFG, ROOT, get_logger, new_run_id, path, set_seed, verify_manifest
from src.ingest.calendar import ingest_calendar
from src.ingest.traffy import stage_traffy
from src.ingest.weather import download_weather, stage_weather
from src.quality.gates import verify_all_processed_tables
from src.transform.district_panel import build_district_panel

log = get_logger("pipeline")


def stage_check(run_id: str) -> None:
    """Stage 0: sanity-check config, folders and raw data snapshot checksums."""
    log.info("project=%s tz=%s seed=%s", CFG["project"], CFG["tz"], CFG["seed"])
    log.info("study=%s..%s split=%s", CFG["study"]["start"], CFG["study"]["end"], CFG["split"])
    for key in ("raw", "quarantine", "staging", "processed", "features", "audit", "quality"):
        p = path(key)
        p.mkdir(parents=True, exist_ok=True)
    problems = verify_manifest()
    if problems:
        for p in problems:
            log.error(p)
        sys.exit("raw data snapshot does not match manifest -> stop run")
    log.info("raw data snapshot checksums: OK")


def stage_weather_step(run_id: str) -> None:
    download_weather()  # skips download if raw files already exist
    stage_weather(run_id=run_id)


def stage_calendar_step(run_id: str) -> None:
    ingest_calendar(run_id=run_id)


def stage_traffy_step(run_id: str) -> None:
    stage_traffy(run_id=run_id)


def stage_panel_step(run_id: str) -> None:
    build_district_panel(run_id=run_id)


def stage_quality_step(run_id: str) -> None:
    verify_all_processed_tables(run_id=run_id)


STAGES = {
    "check": stage_check,
    "weather": stage_weather_step,
    "calendar": stage_calendar_step,
    "traffy": stage_traffy_step,
    "panel": stage_panel_step,
    "quality": stage_quality_step,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES) + ["all"], default="all")
    args = ap.parse_args()

    set_seed()
    run_id = new_run_id()
    log.info("starting pipeline run_id=%s (stage=%s)", run_id, args.stage)
    todo = STAGES.items() if args.stage == "all" else [(args.stage, STAGES[args.stage])]
    for name, fn in todo:
        log.info("=== stage: %s ===", name)
        fn(run_id)
    log.info("pipeline finished successfully (run_id=%s)", run_id)


if __name__ == "__main__":
    main()
