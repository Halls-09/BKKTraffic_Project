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
    """Build and launch the interactive 4-View Bangkok Traffic & MLOps Dashboard."""
    html_file = generate_standalone_dashboard_html()
    if build_only:
        log_success(f"สร้างไฟล์ Dashboard เรียบร้อยที่: {html_file}")
        return
    serve_dashboard(host=host, port=port)


@app.command("test")
def cmd_test() -> None:
    """Run all unit tests (Data Contract 4.2, Zero Temporal Leakage, MLOps Registry & PSI Drift)."""
    log_step("UNIT_TESTS", "รันชุดทดสอบความถูกต้องของระบบ (Data Contract, Temporal Leakage, MLOps)...")
    from tests.test_data_contract import test_quality_gate_quarantines_outliers
    from tests.test_mlops_registry import test_model_registry_champion_gates, test_psi_drift_mathematics
    from tests.test_temporal_leakage import test_zero_temporal_leakage_boundaries

    tests = [
        ("test_quality_gate_quarantines_outliers (Data Contract 4.2)", test_quality_gate_quarantines_outliers),
        ("test_zero_temporal_leakage_boundaries (Strict As-Of Cutoff t)", test_zero_temporal_leakage_boundaries),
        ("test_psi_drift_mathematics (PSI Drift Formula)", test_psi_drift_mathematics),
        ("test_model_registry_champion_gates (Model Registry & Guardrails)", test_model_registry_champion_gates),
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


if __name__ == "__main__":
    app()
