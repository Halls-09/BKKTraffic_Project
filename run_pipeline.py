"""Run the whole BKK Traffic data pipeline with one command.

    python run_pipeline.py                 # run every stage
    python run_pipeline.py --stage check   # only environment / data snapshot check

Stages will be added step by step (ingest -> quality -> aggregate -> features).
"""
from __future__ import annotations

import argparse
import sys

from src.common import CFG, ROOT, get_logger, new_run_id, path, set_seed, verify_manifest

log = get_logger("pipeline")


def stage_check() -> None:
    """Stage 0: sanity-check config, folders and raw data snapshot."""
    log.info("project=%s tz=%s seed=%s", CFG["project"], CFG["tz"], CFG["seed"])
    log.info("study=%s..%s split=%s", CFG["study"]["start"], CFG["study"]["end"], CFG["split"])
    for key in ("raw", "quarantine", "staging", "processed", "features", "audit", "quality"):
        p = path(key)
        p.mkdir(parents=True, exist_ok=True)
        log.info("folder ok: %s", p.relative_to(ROOT).as_posix())
    problems = verify_manifest()
    if problems:
        for p in problems:
            log.error(p)
        sys.exit("raw data snapshot does not match manifest -> stop run")
    log.info("raw data snapshot: OK")


STAGES = {
    "check": stage_check,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES) + ["all"], default="all")
    args = ap.parse_args()

    set_seed()
    run_id = new_run_id()
    log.info("run_id=%s", run_id)
    todo = STAGES.values() if args.stage == "all" else [STAGES[args.stage]]
    for fn in todo:
        log.info("=== stage: %s ===", fn.__name__)
        fn()
    log.info("done")


if __name__ == "__main__":
    main()

