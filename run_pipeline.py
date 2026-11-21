"""
Master MLOps CLI Entrypoint for Bangkok Traffic 1-Hour-Ahead Forecasting Platform.
Role: ML Engineer (Pipeline Orchestration, Model Registry, Drift Monitoring, Dashboard & API)

Usage:
  python run_pipeline.py run-all             # Run full end-to-end MLOps DAG + build dashboard
  python run_pipeline.py step <name>         # Run single stage: prepare|train|explain|forecast|registry|monitor|charts|dashboard
  python run_pipeline.py status              # Show Champion Model, Promotion Gates, PSI Drift & Lineage
  python run_pipeline.py rollback v1.0.0     # Roll back Champion model to a specific registered version
  python run_pipeline.py dashboard           # Build & launch FastAPI + Interactive 4-Tab Dashboard (port 8050)
  python run_pipeline.py test                # Run automated MLOps, Temporal Leakage & API tests
"""
from __future__ import annotations

import json
import sys

import typer
from rich.panel import Panel
from rich.table import Table

from src.ml_engineering.api_server import serve_dashboard
from src.ml_engineering.app_dashboard import generate_standalone_dashboard_html
from src.ml_engineering.data_contract_adapter import ingest_external_teammate_dataset
from src.ml_engineering.pipeline_runner import PipelineOrchestrator
from src.ml_engineering.registry import ModelRegistry
from src.utils.logger import console, log_error, log_step, log_success
from src.utils.storage import ROOT_DIR, load_config

app = typer.Typer(
    name="bkk-mlops",
    help="Bangkok 1-Hour-Ahead Traffic Forecasting & MLOps Control Plane (CSS 342)",
    add_completion=False,
)


@app.command("run-all")
def cmd_run_all(
    force_prepare: bool = typer.Option(False, "--force-prepare", help="Rebuild real iTIC dataset from zips"),
) -> None:
    """Execute the full MLOps DAG: train -> explain -> forecast -> registry -> monitor -> charts -> dashboard."""
    orchestrator = PipelineOrchestrator()
    result = orchestrator.run_full_pipeline(force_prepare=force_prepare)
    if result["status"] != "SUCCESS":
        raise typer.Exit(code=1)


@app.command("step")
def cmd_step(
    step_name: str = typer.Argument(
        ...,
        help="Stage: prepare | contract | train | ablation | explain | forecast | online-ml | registry | monitor | charts | dashboard | model-card",
    ),
) -> None:
    """Run a single pipeline stage."""
    orchestrator = PipelineOrchestrator()
    orchestrator.run_step(step_name)


@app.command("ingest-external")
def cmd_ingest_external(
    external_path: str = typer.Argument(..., help="Path to new Data Engineer feature table (.parquet, .csv, or .pkl)"),
    run_dag: bool = typer.Option(True, "--run-dag/--no-run-dag", help="Automatically run MLOps pipeline after passing Data Contract 4.2"),
) -> None:
    """Validate and ingest a new dataset from Data Engineer teammate via Data Contract 4.2 Adapter."""
    ingest_external_teammate_dataset(external_path)
    if run_dag:
        orchestrator = PipelineOrchestrator()
        orchestrator.run_full_pipeline(force_prepare=False)


@app.command("status")
def cmd_status() -> None:
    """Inspect MLOps Control Plane: Active Champion, Promotion Gates, PSI Drift & Latest Lineage."""
    cfg = load_config()
    champ_path = ROOT_DIR / cfg["paths"]["registry_dir"] / "champion_metadata.json"
    drift_path = ROOT_DIR / cfg["paths"]["monitoring_dir"] / "drift_monitoring_report.json"

    if not champ_path.exists():
        log_error("ยังไม่มี Champion Model ใน Registry กรุณารัน `python run_pipeline.py run-all` ก่อน")
        raise typer.Exit(code=1)

    champ = json.loads(champ_path.read_text(encoding="utf-8"))
    m_test = champ["metrics"]["test"]
    b0_test = champ["metrics"]["baseline_b0_test"]

    console.print(
        Panel.fit(
            f"[bold green]ACTIVE CHAMPION: {champ['version']}[/bold green] ({champ['stage']})\n"
            f"Backend: {champ['model_backend']} | Threshold: {champ['optimal_threshold']} | Features: {champ['num_features']}\n"
            f"Winning Window: {champ.get('winning_window_strategy')}\n"
            f"Model SHA-256: [cyan]{champ['model_sha256'][:24]}...[/cyan]\n"
            f"Test F1: [bold yellow]{m_test['f1_all']:.4f}[/bold yellow] (vs B0: {b0_test['f1_all']:.4f}) | "
            f"Onset Recall: [bold magenta]{m_test['onset_recall']:.4f}[/bold magenta] | "
            f"MAE: {m_test['mae']:.4f}",
            title="Model Registry Status",
            border_style="green",
        )
    )

    if drift_path.exists():
        drift = json.loads(drift_path.read_text(encoding="utf-8"))
        table = Table(title=f"Drift Monitoring (Overall Health: {drift['overall_system_health']})")
        table.add_column("Feature", style="cyan")
        table.add_column("Train Mean", justify="right")
        table.add_column("Test Mean", justify="right")
        table.add_column("PSI", justify="right", style="bold")
        table.add_column("Status", justify="center")
        for r in drift["current_window_psi"] + [drift["prediction_drift"]]:
            color = "green" if r["status"] == "STABLE" else ("yellow" if r["status"] == "WARNING" else "red")
            table.add_row(
                r["feature"],
                f"{r['ref_mean']:.4f}",
                f"{r['cur_mean']:.4f}",
                f"{r['psi']:.4f}",
                f"[{color}]{r['status']}[/{color}]",
            )
        console.print(table)


@app.command("rollback")
def cmd_rollback(
    target_version: str = typer.Argument(..., help="Registered version ID, e.g. v1.0.0"),
    reason: str = typer.Option("Manual CLI rollback", "--reason", help="Audit log reason"),
) -> None:
    """Roll back Active Champion to a previously registered model version (SHA-256 verified)."""
    reg = ModelRegistry()
    reg.rollback_to_version(target_version, reason=reason)


@app.command("dashboard")
def cmd_dashboard(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8050, "--port"),
    build_only: bool = typer.Option(False, "--build-only", help="Generate HTML without starting HTTP server"),
) -> None:
    """Build and serve the Interactive 4-Tab Traffic & MLOps Dashboard + FastAPI API."""
    html_path = generate_standalone_dashboard_html()
    if build_only:
        log_success(f"สร้างไฟล์ Dashboard สำเร็จที่ {html_path}")
        return
    serve_dashboard(host=host, port=port)


@app.command("test")
def cmd_test() -> None:
    """Run all automated MLOps, Zero-Leakage, PSI Drift, Registry & FastAPI tests."""
    log_step("UNIT_TESTS", "รันชุดทดสอบ MLOps (Zero Temporal Leakage, PSI Drift, Registry Gates, Rollback, FastAPI)...")
    from tests.test_mlops import (
        test_fastapi_endpoints,
        test_proposal_acceptance_gates_and_monitoring,
        test_psi_drift_math,
        test_registry_gates_and_rollback,
        test_zero_temporal_leakage_real_data,
    )

    suite = [
        ("Zero Temporal Leakage & Data Contract 4.2 (Real iTIC Split Boundaries)", test_zero_temporal_leakage_real_data),
        ("PSI Drift Mathematical Accuracy", test_psi_drift_math),
        ("Model Registry Promotion Gates & Audited Rollback", test_registry_gates_and_rollback),
        ("FastAPI Serving & MLOps Control Plane Endpoints", test_fastapi_endpoints),
        ("Proposal Acceptance Gates (3-Fold Robustness, Daytime 06-23h, H1-H5, Layer-1 SLA)", test_proposal_acceptance_gates_and_monitoring),
    ]
    for name, fn in suite:
        try:
            fn()
            log_success(f"PASSED: {name}")
        except Exception as exc:
            log_error(f"FAILED: {name} -> {exc}")
            sys.exit(1)
    log_success(f"ผ่านครบทุกชุดทดสอบ {len(suite)}/{len(suite)} (100% Pass)")


if __name__ == "__main__":
    app()
