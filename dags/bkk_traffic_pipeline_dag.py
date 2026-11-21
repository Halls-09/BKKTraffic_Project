"""Apache Airflow Medallion DAG: Sources -> Bronze -> Silver -> Gold -> ML.

อ้างอิง Proposal ข้อ 7 (Data Engineering และสถาปัตยกรรม):
1. [BRONZE LAYER]  : Verify raw data snapshot checksums (`manifest.jsonl`) & land raw files
                     (iTIC XML.gz -> daily raw CSVs, Open-Meteo JSON, Traffy JSON, Calendar CSV)
2. [SILVER LAYER]  : Clean, deduplicate, remove PII, standardize timezone (`Asia/Bangkok`),
                     aggregate iTIC & Traffy to 15-minute intervals, build `dim_segment` &
                     `quality_completeness_seg_day` in DuckDB (`silver.*`)
3. [GOLD LAYER]    : Compute Train-only `free_flow_kmh` (P90) and `traffy_district_static_train`,
                     compute `ratio`, `target_ratio` (t+60 min), `new_onset`, and merge all datasets
                     at 15-minute resolution into `gold.merged_district_15m` & `gold.merged_segment_15m`
                     inside `data/bkk_traffic.duckdb` + run Quality & Leakage Gate assertions
4. [ML LAYER]      : Handoff to Data Scientist (Apichet: EDA, Baselines B0/B1, LightGBM, SHAP)
                     and ML Engineer (Sarit: Model Registry, Monitoring, Dashboard)
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

PROJECT_ROOT = Path("/opt/airflow/project") if Path("/opt/airflow/project/config.yaml").exists() else Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
except ImportError:
    class DAG:  # type: ignore[no-redef]
        def __init__(self, dag_id: str, **kwargs):
            self.dag_id = dag_id
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class PythonOperator:  # type: ignore[no-redef]
        def __init__(self, task_id: str, python_callable, **kwargs):
            self.task_id = task_id
            self.python_callable = python_callable
            self.downstream: list = []

        def set_downstream(self, other):
            self.downstream.append(other)
            return other

        def __rshift__(self, other):
            if isinstance(other, (list, tuple)):
                for o in other:
                    self.set_downstream(o)
            else:
                self.set_downstream(other)
            return other

        def __rrshift__(self, other):
            if isinstance(other, (list, tuple)):
                for o in other:
                    o.set_downstream(self)
            return self


from src.common import get_logger, set_seed, verify_manifest
from src.ingest.calendar import ingest_calendar
from src.ingest.itic_traffic import process_month_with_retry
from src.ingest.traffy import stage_traffy
from src.ingest.weather import download_weather, stage_weather
from src.quality.gates import verify_all_processed_tables
from src.transform.district_panel import build_district_panel
from src.warehouse.duckdb_loader import build_duckdb_warehouse

log = get_logger("airflow.medallion_dag")


# ---------------------------------------------------------------------------
# 1. BRONZE TASKS (Raw Landing & Checksum Verification)
# ---------------------------------------------------------------------------
def bronze_verify_raw_sources() -> None:
    set_seed()
    problems = verify_manifest()
    if problems:
        raise RuntimeError(f"Bronze manifest checksum failure: {problems}")
    download_weather()


def bronze_extract_itic_daily() -> None:
    for mo in ["2022-06", "2022-07", "2022-08", "2022-09"]:
        process_month_with_retry(mo)


# ---------------------------------------------------------------------------
# 2. SILVER TASKS (Cleaning, Deduplication, Governance PII Removal, Standardization)
# ---------------------------------------------------------------------------
def silver_clean_weather() -> None:
    stage_weather(run_id="airflow_silver")


def silver_clean_calendar() -> None:
    ingest_calendar(run_id="airflow_silver")


def silver_clean_traffy() -> None:
    stage_traffy(run_id="airflow_silver")


# ---------------------------------------------------------------------------
# 3. GOLD TASKS (DuckDB Medallion Warehouse: 15-min Aggregation, Target, Merge & Quality Gate)
# ---------------------------------------------------------------------------
def gold_build_duckdb_medallion_15m() -> None:
    build_district_panel(run_id="airflow_gold")
    build_duckdb_warehouse(run_id="airflow_gold")


def gold_quality_and_leakage_gate() -> None:
    verify_all_processed_tables(run_id="airflow_gold")


# ---------------------------------------------------------------------------
# 4. ML HANDOFF TASK (Ready for EDA, Baseline B0/B1, LightGBM & SHAP)
# ---------------------------------------------------------------------------
def ml_handoff_ready_notification() -> None:
    log.info(
        "GOLD -> ML HANDOFF READY! DuckDB tables `gold.merged_district_15m` and "
        "`gold.merged_segment_15m` are validated and ready for Apichet (EDA & LightGBM)."
    )


default_args = {
    "owner": "thanakorn_data_engineer",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": dt.timedelta(minutes=5),
}

with DAG(
    dag_id="bkk_traffic_medallion_bronze_silver_gold_ml",
    description="Sources -> Bronze -> Silver -> Gold (DuckDB 15-min Merged Warehouse) -> ML Handoff",
    default_args=default_args,
    start_date=dt.datetime(2022, 6, 1),
    schedule="@daily",
    catchup=False,
    tags=["bkk_traffic", "css342", "medallion", "bronze_silver_gold", "duckdb"],
) as dag:

    # Layer 1: Bronze
    t_bronze_sources = PythonOperator(
        task_id="bronze_verify_raw_sources",
        python_callable=bronze_verify_raw_sources,
    )
    t_bronze_itic = PythonOperator(
        task_id="bronze_extract_itic_daily",
        python_callable=bronze_extract_itic_daily,
    )

    # Layer 2: Silver
    t_silver_weather = PythonOperator(
        task_id="silver_clean_weather",
        python_callable=silver_clean_weather,
    )
    t_silver_calendar = PythonOperator(
        task_id="silver_clean_calendar",
        python_callable=silver_clean_calendar,
    )
    t_silver_traffy = PythonOperator(
        task_id="silver_clean_traffy",
        python_callable=silver_clean_traffy,
    )

    # Layer 3: Gold (DuckDB 15-minute Warehouse & Quality/Leakage Gates)
    t_gold_duckdb = PythonOperator(
        task_id="gold_build_duckdb_medallion_15m",
        python_callable=gold_build_duckdb_medallion_15m,
    )
    t_gold_gate = PythonOperator(
        task_id="gold_quality_and_leakage_gate",
        python_callable=gold_quality_and_leakage_gate,
    )

    # Layer 4: ML Handoff
    t_ml_ready = PythonOperator(
        task_id="ml_handoff_ready_for_eda_and_lightgbm",
        python_callable=ml_handoff_ready_notification,
    )

    # Medallion Pipeline Dependency Flow:
    # Sources -> Bronze -> Silver -> Gold (DuckDB) -> Quality Gate -> ML
    t_bronze_sources >> [t_bronze_itic, t_silver_weather, t_silver_calendar, t_silver_traffy] >> t_gold_duckdb
    t_gold_duckdb >> t_gold_gate >> t_ml_ready
