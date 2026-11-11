# ==============================================================================
# Makefile - Reproducible End-to-End Pipeline Orchestration
# Usage:
#   make all          -> Run full pipeline (synthetic -> ingest -> quality -> features -> train -> monitor)
#   make ingest       -> Generate/load Bronze & promote to Silver via Quality Gate
#   make features     -> Build Gold Feature Store (8 Feature Groups, Zero Leakage)
#   make train        -> Train Baselines (B0, B1) + LightGBM/HistGBR (B2) + Ablation (B3) + SHAP
#   make registry     -> Evaluate Challenger vs Champion and promote if guardrails pass
#   make monitor      -> Run PSI Data Drift & Rolling 7-Day Concept Drift monitoring
#   make dashboard    -> Launch interactive Bangkok Traffic Dashboard
#   make test         -> Run Data Contract, Temporal Leakage, and MLOps unit tests
# ==============================================================================

PYTHON := python

.PHONY: all synthetic ingest features train ablation explain registry monitor dashboard test status clean

all:
	$(PYTHON) run_pipeline.py run-all

synthetic:
	$(PYTHON) run_pipeline.py step synthetic

ingest:
	$(PYTHON) run_pipeline.py step ingest

features:
	$(PYTHON) run_pipeline.py step features

train:
	$(PYTHON) run_pipeline.py step train

ablation:
	$(PYTHON) run_pipeline.py step ablation

explain:
	$(PYTHON) run_pipeline.py step explain

registry:
	$(PYTHON) run_pipeline.py step registry

monitor:
	$(PYTHON) run_pipeline.py step monitor

status:
	$(PYTHON) run_pipeline.py status

dashboard:
	$(PYTHON) run_pipeline.py dashboard

test:
	$(PYTHON) run_pipeline.py test
