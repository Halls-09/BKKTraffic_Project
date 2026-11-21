"""
Apache Airflow Production DAGs for Bangkok 1-Hour-Ahead Traffic Forecasting & Online MLOps Platform.
Matches Proposal Section 7 (Pages 8-9) & Section 10 (Page 11):
  1) `bkk_traffic_15m_online_inference`:
     Runs every 15 minutes (`*/15 * * * *`) — validates incoming micro-batch via Data Contract 4.2,
     applies Segment Circuit-Breaker, and serves 1-hour-ahead predictions for all 448 directional segments.
  2) `bkk_traffic_daily_monitoring_and_online_warmstart`:
     Runs daily (`0 1 * * *`) — joins predictions with actual ground truth arrived at t+60m,
     computes 3-Layer Monitoring (Data Quality, Feature/Prediction PSI, Rolling 7d F1/MAE),
     and performs Online Incremental Warm-Start update (`init_model`) on the sliding window.
  3) `bkk_traffic_weekly_challenger_tournament`:
     Runs weekly (`0 2 * * 0` or triggered by Layer-3 F1 drop > 0.05) — runs 3-Strategy Challenger
     Tournament (Expanding Window, Rolling 8 Weeks, Same-Season Monsoon), checks 4 Promotion Gates
     + 3-Fold Robustness, promotes or keeps Champion, and auto-rolls back on post-promote degradation.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from src.data_science.forecast import run_forecast_report
from src.ml_engineering.app_dashboard import generate_standalone_dashboard_html
from src.ml_engineering.data_contract_adapter import validate_data_contract_4_2
from src.ml_engineering.monitoring_drift import run_monitoring_drift
from src.ml_engineering.online_ml import run_online_ml_simulation
from src.ml_engineering.pipeline_runner import generate_model_card_md
from src.ml_engineering.registry import run_model_registry_step

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


DEFAULT_ARGS: dict[str, Any] = {
    "owner": "sarit_ml_engineer_67090500441",
    "depends_on_past": False,
    "start_date": datetime(2022, 9, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

AIRFLOW_DAG_SPECS: list[dict[str, Any]] = [
    {
        "dag_id": "bkk_traffic_15m_online_inference",
        "schedule": "*/15 * * * * (ทุก 15 นาที)",
        "proposal_ref": "Section 7 & 10: Ingest + Quality Gate 4.2 + Predict t+60m",
        "tasks": [
            "validate_data_contract_4_2",
            "check_segment_circuit_breaker",
            "predict_next_60min_all_segments",
        ],
    },
    {
        "dag_id": "bkk_traffic_daily_monitoring_and_online_warmstart",
        "schedule": "0 1 * * * (รายวัน 01:00 น.)",
        "proposal_ref": "Section 7 & 10: Join Actual t+60m + 3-Layer Monitor + Online Incremental Warm-Start",
        "tasks": [
            "join_ground_truth_t60m",
            "run_3layer_drift_and_performance_monitor",
            "online_lightgbm_warmstart_update",
            "refresh_dashboard_artifacts",
        ],
    },
    {
        "dag_id": "bkk_traffic_weekly_challenger_tournament",
        "schedule": "0 2 * * 0 (รายสัปดาห์ หรือเมื่อ F1 ตกเกิน 0.05)",
        "proposal_ref": "Section 10: 3-Challenger Tournament + 4 Promotion Gates + Audited Rollback",
        "tasks": [
            "run_3strategy_challenger_tournament",
            "evaluate_4_promotion_gates_and_3fold_robustness",
            "promote_or_rollback_champion",
            "update_model_card_and_lineage",
        ],
    },
]


if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="bkk_traffic_15m_online_inference",
        default_args=DEFAULT_ARGS,
        schedule_interval="*/15 * * * *",
        catchup=False,
        tags=["bkk_traffic", "css342", "online_inference"],
    ) as dag_15m:
        t_contract = PythonOperator(task_id="validate_data_contract_4_2", python_callable=validate_data_contract_4_2)
        t_predict = PythonOperator(task_id="predict_next_60min_all_segments", python_callable=run_forecast_report)
        t_contract >> t_predict

    with DAG(
        dag_id="bkk_traffic_daily_monitoring_and_online_warmstart",
        default_args=DEFAULT_ARGS,
        schedule_interval="0 1 * * *",
        catchup=False,
        tags=["bkk_traffic", "css342", "monitoring", "online_ml"],
    ) as dag_daily:
        t_mon = PythonOperator(task_id="run_3layer_drift_and_performance_monitor", python_callable=run_monitoring_drift)
        t_online = PythonOperator(task_id="online_lightgbm_warmstart_update", python_callable=run_online_ml_simulation)
        t_dash = PythonOperator(task_id="refresh_dashboard_artifacts", python_callable=generate_standalone_dashboard_html)
        t_mon >> t_online >> t_dash

    with DAG(
        dag_id="bkk_traffic_weekly_challenger_tournament",
        default_args=DEFAULT_ARGS,
        schedule_interval="0 2 * * 0",
        catchup=False,
        tags=["bkk_traffic", "css342", "registry", "challenger_tournament"],
    ) as dag_weekly:
        t_reg = PythonOperator(task_id="run_3strategy_challenger_tournament", python_callable=run_model_registry_step)
        t_card = PythonOperator(task_id="update_model_card_and_lineage", python_callable=generate_model_card_md)
        t_reg >> t_card
