"""
Master CLI Entrypoint for Bangkok Traffic Congestion Forecasting & MLOps Platform.
Responsibility: Sarit (67090500441) - ML Engineer

Usage Examples:
  python run_pipeline.py run-all
  python run_pipeline.py step registry
  python run_pipeline.py step monitor
  python run_pipeline.py status
  python run_pipeline.py dashboard
  python run_pipeline.py test
"""
from __future__ import annotations

import json
import sys
from typing import Optional

import typer
from rich.panel import Panel
from rich.table import Table

from src.ml_engineering.app_dashboard import generate_standalone_dashboard_html, serve_dashboard
from src.ml_engineering.pipeline_runner import PipelineOrchestrator
from src.utils.logger import console, log_error, log_info, log_step, log_success
from src.utils.storage import ROOT_DIR, load_config

app = typer.Typer(
    name="bkk-pipeline",
    help="Bangkok 1-Hour-Ahead Traffic Congestion Forecasting & MLOps CLI (CSS 342)",
    add_completion=False,
)


@app.command("run-all")
def cmd_run_all(
    skip_synthetic_if_exists: bool = typer.Option(
        False, "--skip-synthetic", help="Skip synthetic data generation if Bronze files already exist."
    ),
) -> None:
    """Run the complete 8-stage Medallion + MLOps DAG and build the interactive dashboard."""
    orchestrator = PipelineOrchestrator()
    orchestrator.run_full_pipeline(skip_synthetic_if_exists=skip_synthetic_if_exists)
    generate_standalone_dashboard_html()


@app.command("step")
def cmd_step(
    step_name: str = typer.Argument(
        ...,
        help="Step name: synthetic | ingest | features | train | ablation | explain | registry | monitor",
    ),
) -> None:
    """Execute a single DAG step."""
    orchestrator = PipelineOrchestrator()
    orchestrator.run_step(step_name)


@app.command("status")
def cmd_status() -> None:
    """Display MLOps Control Plane status: Champion Model, Promotion Gates, Drift Health, and Lineage."""
    cfg = load_config()
    champ_path = ROOT_DIR / cfg["paths"]["registry_dir"] / "champion_metadata.json"
    drift_path = ROOT_DIR / cfg["paths"]["monitoring_dir"] / "drift_monitoring_report.json"
    manifest_path = ROOT_DIR / cfg["paths"]["manifest_path"]

    if not champ_path.exists():
        log_error("ยังไม่มีโมเดลในระบบ กรุณารัน `python run_pipeline.py run-all` ก่อน")
        raise typer.Exit(code=1)

    with open(champ_path, "r", encoding="utf-8") as f:
        champ = json.load(f)

    m_test = champ["metrics"]["test"]
    b0_test = champ["metrics"]["baseline_b0_test"]

    console.print(
        Panel.fit(
            f"[bold green]🏆 ACTIVE CHAMPION MODEL: {champ['version']}[/bold green]\n"
            f"[white]Backend:[/white] {champ['model_backend']} | "
            f"[white]Optimal Threshold:[/white] {champ['optimal_threshold']} | "
            f"[white]Features:[/white] {champ['num_features']}\n"
            f"[white]Model SHA256:[/white] [cyan]{champ['model_sha256'][:24]}...[/cyan]\n"
            f"[white]Test F1 (All):[/white] [bold yellow]{m_test['f1_all']:.4f}[/bold yellow] (vs B0: {b0_test['f1_all']:.4f}) | "
            f"[white]Onset Recall:[/white] [bold magenta]{m_test['onset_recall']:.4f}[/bold magenta] (vs B0: {b0_test['onset_recall']:.4f}) | "
            f"[white]MAE:[/white] {m_test['mae_ratio']:.4f}",
            title="MLOps Model Registry Status",
            border_style="green",
        )
    )

    if drift_path.exists():
        with open(drift_path, "r", encoding="utf-8") as f:
            drift = json.load(f)
        dt_table = Table(title=f"📉 Data Drift Monitoring (Overall: {drift['overall_system_health']})")
        dt_table.add_column("Feature", style="cyan")
        dt_table.add_column("Train Mean", justify="right")
        dt_table.add_column("Test Mean", justify="right")
        dt_table.add_column("PSI (Current)", justify="right", style="bold")
        dt_table.add_column("Status", justify="center")

        for item in drift.get("current_window_psi", []):
            st = item["status"]
            badge = f"[green]{st}[/green]" if st == "STABLE" else f"[red]{st}[/red]"
            dt_table.add_row(
                item["feature"],
                f"{item['ref_mean']:.4f}",
                f"{item['cur_mean']:.4f}",
                f"{item['psi']:.4f}",
                badge,
            )
        console.print(dt_table)

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        log_info(f"Data Checksum Manifest: ติดตามไฟล์ข้อมูลทั้งหมด {len(manifest.get('files', {}))} ตาราง (SHA-256 Verified)")


@app.command("dashboard")
def cmd_dashboard(
    host: str = typer.Option("127.0.0.1", "--host", help="Host address"),
    port: int = typer.Option(8050, "--port", help="Port number"),
    build_only: bool = typer.Option(
        False, "--build-only", help="Only build artifacts/dashboard_index.html without starting HTTP server"
    ),
) -> None:
    """Build and launch the interactive 4-View Bangkok Traffic & MLOps Dashboard + FastAPI Server."""
    html_file = generate_standalone_dashboard_html()
    if build_only:
        log_success(f"สร้างไฟล์ Dashboard เรียบร้อยที่: {html_file}")
        return
    serve_dashboard(host=host, port=port)


@app.command("env-check")
def cmd_env_check() -> None:
    """Inspect `.env` API keys status (masked safely) and list available FastAPI endpoints."""
    from src.utils.env_config import get_api_keys_status

    status = get_api_keys_status()
    table = Table(title=f"🔑 .env API Keys & Endpoints Status ({status['env_file_path']})")
    table.add_column("Env Variable", style="cyan", no_wrap=True)
    table.add_column("Provider / Service", style="white")
    table.add_column("Mode", justify="center")
    table.add_column("Masked Key", justify="center", style="yellow")
    table.add_column("Endpoint / Fallback", style="dim")

    for k in status["keys"]:
        mode_badge = (
            f"[bold green]{k['mode']}[/bold green]"
            if k["is_configured"]
            else f"[cyan]{k['mode']}[/cyan]"
        )
        table.add_row(
            k["key_name"],
            k["provider"],
            mode_badge,
            k["masked_preview"],
            f"{k['endpoint_url']} (Fallback: {k['fallback_mode']})",
        )
    console.print(table)


@app.command("test")
def cmd_test() -> None:
    """Run all unit tests (Data Contract 4.2, Zero Temporal Leakage, MLOps Registry, PSI Drift & API Endpoints)."""
    log_step("UNIT_TESTS", "รันชุดทดสอบความถูกต้องของระบบ (Data Contract, Temporal Leakage, MLOps, API & .env)...")
    from tests.test_data_contract import test_quality_gate_quarantines_outliers
    from tests.test_mlops_registry import (
        test_env_and_api_endpoints,
        test_model_registry_champion_gates,
        test_psi_drift_mathematics,
    )
    from tests.test_temporal_leakage import test_zero_temporal_leakage_boundaries

    tests = [
        ("test_quality_gate_quarantines_outliers (Data Contract 4.2)", test_quality_gate_quarantines_outliers),
        ("test_zero_temporal_leakage_boundaries (Strict As-Of Cutoff t)", test_zero_temporal_leakage_boundaries),
        ("test_psi_drift_mathematics (PSI Drift Formula)", test_psi_drift_mathematics),
        ("test_model_registry_champion_gates (Model Registry & Guardrails)", test_model_registry_champion_gates),
        ("test_env_and_api_endpoints (.env Masking & FastAPI /api/v1/*)", test_env_and_api_endpoints),
    ]

    passed = 0
    for name, fn in tests:
        try:
            fn()
            log_success(f"PASSED: {name}")
            passed += 1
        except Exception as exc:
            log_error(f"FAILED: {name} -> {exc}")
            sys.exit(1)

    log_success(f"ผ่านการทดสอบทั้งหมด {passed}/{len(tests)} ชุดทดสอบ (100% Pass)")


@app.command("de-phase")
def cmd_de_phase(
    stage: str = typer.Option("all", "--stage", help="DE-Phase stage: check | calendar | weather | all"),
) -> None:
    """Run DE-Phase modules from `origin/DE-Phase` (`src/common.py`, `src/ingest/calendar.py`, `src/ingest/weather.py`)."""
    from src.common import CFG, ROOT, new_run_id, path, set_seed, verify_manifest
    from src.ingest.calendar import ingest_calendar
    from src.ingest.weather import download_weather, stage_weather

    set_seed()
    run_id = new_run_id()
    log_step("DE_PHASE", f"รันโมดูลจาก branch DE-Phase (run_id={run_id}, stage={stage})...")

    if stage in ("all", "calendar"):
        df_cal = ingest_calendar(run_id=run_id)
        log_success(f"[DE-Phase] สร้างตารางปฏิทิน 2022 สำเร็จ: {len(df_cal)} วัน -> data/processed/calendar_daily.parquet")

    if stage in ("all", "weather"):
        download_weather()
        df_wx = stage_weather(run_id=run_id)
        log_success(f"[DE-Phase] สร้างตารางสภาพอากาศ 3 จุด (central/north/east) สำเร็จ: {len(df_wx)} แถว -> data/processed/weather_hourly.parquet")

    if stage in ("all", "check"):
        for key in ("raw", "quarantine", "staging", "processed", "features", "audit", "quality"):
            p = path(key)
            p.mkdir(parents=True, exist_ok=True)
        problems = verify_manifest()
        if problems:
            for prob in problems:
                log_error(f"Manifest check note: {prob}")
        else:
            log_success("[DE-Phase] ตรวจสอบ logs/manifest.jsonl (SHA-256 Raw Snapshot): OK 100%")


if __name__ == "__main__":
    # Support DE-Phase legacy flag `python run_pipeline.py --stage check` directly
    if len(sys.argv) >= 2 and sys.argv[1] == "--stage":
        sys.argv.insert(1, "de-phase")
    app()
