"""
End-to-End Pipeline Orchestrator & Lineage Tracker.
Responsibility: Sarit (67090500441) - ML Engineer

Orchestrates the 8-stage MLOps DAG:
1. synthetic  -> Generate realistic Bangkok Bronze dataset (if needed)
2. ingest     -> Bronze -> Quality Gate 4.2 -> Silver Layer (iTIC, Weather, Traffy)
3. features   -> Silver -> Gold Feature Store (8 Groups, Strict As-Of Cutoff t)
4. train      -> Train Baselines (B0, B1) & LightGBM/HistGBR (B2) + Threshold Tuning
5. ablation   -> Run B3 Feature Group Ablation Study
6. explain    -> Run SHAP / Attribution & Hypotheses H1-H5 Verification
7. registry   -> Register Challenger & Execute Champion Promotion Gates
8. monitor    -> Compute PSI Data Drift & Rolling Concept Drift Monitoring
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from rich.panel import Panel
from rich.table import Table

from src.data_engineering.build_features import run_build_features
from src.data_engineering.ingest_itic import run_ingest_itic
from src.data_engineering.ingest_traffy import run_ingest_traffy
from src.data_engineering.ingest_weather import run_ingest_weather_and_incidents
from src.data_engineering.synthetic_generator import generate_synthetic_bronze_data
from src.data_science.ablation_runner import run_ablation_study
from src.data_science.explain_shap import run_explain_shap
from src.data_science.train_lgbm import run_train_and_evaluate
from src.ml_engineering.monitoring_drift import run_monitoring_drift
from src.ml_engineering.registry import run_model_registry_step
from src.utils.logger import console, log_error, log_step, log_success
from src.utils.storage import ROOT_DIR, load_config


def _step_ingest_all() -> dict[str, Any]:
    res_itic = run_ingest_itic()
    res_weather = run_ingest_weather_and_incidents()
    res_traffy = run_ingest_traffy()
    return {**res_itic, **res_weather, **res_traffy}


DAG_STEPS: dict[str, tuple[str, str, Callable[[], Any]]] = {
    "synthetic": ("Data Engineering", "สร้างข้อมูลจำลอง iTIC + Weather + Traffy (Bronze)", generate_synthetic_bronze_data),
    "ingest": ("Data Engineering", "ทำความสะอาดข้อมูล + Quality Gate 4.2 -> Silver Layer", _step_ingest_all),
    "features": ("Data Engineering", "สร้าง Gold Feature Store ทั้ง 8 กลุ่ม (Zero Leakage)", run_build_features),
    "train": ("Data Science", "เทรนโมเดล B0, B1, B2 + จูน Threshold บนชุด Validation", run_train_and_evaluate),
    "ablation": ("Data Science", "ทดลอง B3 Ablation Study ตัดทีละกลุ่มฟีเจอร์", run_ablation_study),
    "explain": ("Data Science", "วิเคราะห์ปัจจัย SHAP และตรวจสอบสมมุติฐาน H1–H5", run_explain_shap),
    "registry": ("ML Engineering", "Model Registry & Champion/Challenger Promotion Gate", run_model_registry_step),
    "monitor": ("ML Engineering", "ตรวจจับ Data Drift (PSI) & Rolling Performance Monitoring", run_monitoring_drift),
}


class PipelineOrchestrator:
    def __init__(self) -> None:
        self.cfg = load_config()
        self.lineage_dir = ROOT_DIR / self.cfg["paths"]["lineage_dir"]
        self.lineage_dir.mkdir(parents=True, exist_ok=True)

    def run_step(self, step_name: str) -> dict[str, Any]:
        if step_name not in DAG_STEPS:
            raise ValueError(f"Unknown pipeline step '{step_name}'. Valid steps: {list(DAG_STEPS.keys())}")
        role, desc, fn = DAG_STEPS[step_name]
        t0 = time.perf_counter()
        output = fn()
        elapsed = round(time.perf_counter() - t0, 3)
        return {
            "step": step_name,
            "role": role,
            "description": desc,
            "status": "SUCCESS",
            "duration_sec": elapsed,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary": output if isinstance(output, dict) else {},
        }

    def run_full_pipeline(self, skip_synthetic_if_exists: bool = False) -> dict[str, Any]:
        run_id = f"RUN_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        console.print(
            Panel.fit(
                f"[bold cyan]🚦 BANGKOK TRAFFIC FORECASTING — END-TO-END MLOPS PIPELINE[/bold cyan]\n"
                f"[white]Run ID:[/white] [bold yellow]{run_id}[/bold yellow] | "
                f"[white]Architecture:[/white] Medallion (Bronze -> Silver -> Gold -> Registry -> Monitoring)",
                border_style="cyan",
            )
        )

        t_start = time.perf_counter()
        step_records = []
        overall_status = "SUCCESS"

        for step_name in DAG_STEPS:
            bronze_seg = ROOT_DIR / self.cfg["paths"]["bronze_dir"] / "bronze_itic_segments.parquet"
            if step_name == "synthetic" and skip_synthetic_if_exists and bronze_seg.exists():
                continue

            try:
                rec = self.run_step(step_name)
                step_records.append(rec)
            except Exception as exc:
                overall_status = "FAILED"
                log_error(f"Pipeline failed at step '{step_name}': {exc}")
                step_records.append({
                    "step": step_name,
                    "status": "FAILED",
                    "error": str(exc),
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                })
                raise

        total_sec = round(time.perf_counter() - t_start, 3)
        lineage_record = {
            "run_id": run_id,
            "status": overall_status,
            "total_duration_sec": total_sec,
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
            "steps": step_records,
        }

        with open(self.lineage_dir / "latest_run.json", "w", encoding="utf-8") as f:
            json.dump(lineage_record, f, indent=2, ensure_ascii=False)

        history_path = self.lineage_dir / "pipeline_run_history.json"
        history = []
        if history_path.exists():
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
        history.append(lineage_record)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history[-25:], f, indent=2, ensure_ascii=False)

        self.print_summary_table(lineage_record)
        return lineage_record

    def print_summary_table(self, lineage_record: dict[str, Any]) -> None:
        table = Table(title=f"📊 สรุปผลการรัน Pipeline ({lineage_record['run_id']})", show_lines=True)
        table.add_column("ลำดับ", justify="center", style="cyan")
        table.add_column("Step", style="bold magenta")
        table.add_column("บทบาท (Role)", style="white")
        table.add_column("รายละเอียด", style="white")
        table.add_column("สถานะ", justify="center")
        table.add_column("เวลา (วินาที)", justify="right", style="yellow")

        for idx, s in enumerate(lineage_record["steps"], 1):
            st_badge = "[bold green]✔ SUCCESS[/bold green]" if s["status"] == "SUCCESS" else "[bold red]✖ FAILED[/bold red]"
            table.add_row(
                str(idx),
                s["step"],
                s.get("role", "-"),
                s.get("description", "-"),
                st_badge,
                f"{s.get('duration_sec', 0.0):.2f}s",
            )
        console.print(table)
        log_success(f"รัน End-to-End Pipeline เสร็จสมบูรณ์ในเวลา {lineage_record['total_duration_sec']:.2f} วินาที")
