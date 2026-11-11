"""
End-to-End Pipeline Orchestrator & Lineage Tracker.
Responsibility: Sarit (67090500441) - ML Engineer

Orchestrates the 10-stage MLOps DAG:
1. synthetic     -> Generate realistic Bangkok Bronze dataset
2. ingest        -> Bronze -> Quality Gate 4.2 -> Silver Layer (iTIC, Weather, Traffy)
3. features      -> Silver -> Gold Feature Store (8 Groups, Strict As-Of Cutoff t)
4. train         -> Train Baselines (B0, B1) & LightGBM/HistGBR (B2) + Threshold Tuning
5. ablation      -> Run B3 Feature Group Ablation Study
6. explain       -> Run SHAP / Attribution & Hypotheses H1-H5 Verification
7. experiments   -> Run Section 11 Experiments (Horizon 15/30/45/60m, 3-Fold Robustness, Traffy Event Study 95% CI, Hourly MAE)
8. registry      -> Run 3-Challenger Tournament (Expanding/Rolling/Same-Season) & Section 10 Promotion Gates
9. monitor       -> Run 5-Layer MLOps Monitoring (Circuit-Breaker, PSI Drift, Rolling F1, Rollback Readiness)
10. deliverables -> Auto-generate DATA_DICTIONARY.md & MODEL_CARD.md (Section 12)
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
from src.data_science.experiments_suite import run_proposal_experiments
from src.data_science.explain_shap import run_explain_shap
from src.data_science.train_lgbm import run_train_and_evaluate
from src.ml_engineering.deliverables_generator import generate_section12_deliverables
from src.ml_engineering.monitoring_drift import run_monitoring_drift
from src.ml_engineering.registry import run_model_registry_step
from src.utils.logger import console, log_error, log_success
from src.utils.storage import ROOT_DIR, load_config


def _step_ingest_all() -> dict[str, Any]:
    from src.common import new_run_id
    from src.ingest.calendar import ingest_calendar
    from src.ingest.weather import download_weather, stage_weather

    de_run_id = new_run_id()
    df_cal = ingest_calendar(run_id=de_run_id)
    download_weather()
    df_wx_de = stage_weather(run_id=de_run_id)

    res_itic = run_ingest_itic()
    res_weather = run_ingest_weather_and_incidents()
    res_traffy = run_ingest_traffy()
    return {
        "de_calendar_days": len(df_cal),
        "de_weather_3points_rows": len(df_wx_de),
        **res_itic,
        **res_weather,
        **res_traffy,
    }


DAG_STEPS: dict[str, tuple[str, str, Callable[[], Any]]] = {
    "synthetic": ("Data Engineering", "สร้างข้อมูลจำลอง iTIC + Weather + Traffy (Bronze)", generate_synthetic_bronze_data),
    "ingest": ("Data Engineering", "ทำความสะอาดข้อมูล + Quality Gate 4.2 -> Silver Layer", _step_ingest_all),
    "features": ("Data Engineering", "สร้าง Gold Feature Store ทั้ง 8 กลุ่ม (Zero Leakage)", run_build_features),
    "train": ("Data Science", "เทรนโมเดล B0, B1, B2 + จูน Threshold บนชุด Validation", run_train_and_evaluate),
    "ablation": ("Data Science", "ทดลอง B3 Ablation Study ตัดทีละกลุ่มฟีเจอร์", run_ablation_study),
    "explain": ("Data Science", "วิเคราะห์ปัจจัย SHAP และตรวจสอบสมมุติฐาน H1–H5", run_explain_shap),
    "experiments": ("Data Science", "การทดลองข้อ 11 (Horizon 15/30/60m, 3-Fold, Event Study, Hourly MAE)", run_proposal_experiments),
    "registry": ("ML Engineering", "3-Challenger Tournament & Champion Promotion Gate (ข้อ 10)", run_model_registry_step),
    "monitor": ("ML Engineering", "ระบบเฝ้าระวัง 5 ชั้น (Circuit-Breaker, PSI, Rolling F1, Rollback)", run_monitoring_drift),
    "deliverables": ("ML Engineering", "สร้าง DATA_DICTIONARY.md และ MODEL_CARD.md อัตโนมัติ (ข้อ 12)", generate_section12_deliverables),
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
                f"[bold cyan]🚦 BANGKOK TRAFFIC FORECASTING — END-TO-END MLOPS PIPELINE (10-STAGE DAG)[/bold cyan]\n"
                f"[white]Run ID:[/white] [bold yellow]{run_id}[/bold yellow] | "
                f"[white]Architecture:[/white] Medallion + 3-Challenger Registry + 5-Layer Monitoring",
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
        log_success(f"รัน End-to-End Pipeline ทั้ง {len(lineage_record['steps'])} ขั้นตอนสำเร็จในเวลา {lineage_record['total_duration_sec']:.2f} วินาที")
