"""
Model Registry & Automated Champion/Challenger Promotion Engine.
Responsibility: Sarit (67090500441) - ML Engineer

Features:
- Semantic model versioning (v1.0.0, v1.0.1, ...)
- Full artifact lineage (SHA-256 of model binary + SHA-256 of Gold dataset)
- Multi-Gate Champion Promotion evaluation:
  1. Primary Metric Gate: Test F1 >= Baseline B0/B1 + min_improvement
  2. Guardrail Gate 1: Onset Recall >= min_onset_recall (0.55)
  3. Guardrail Gate 2: MAE Ratio <= max_mae_ratio (0.15)
  4. Challenger vs Champion Gate: Challenger F1 >= Champion F1 - 0.005
- Instant rollback support to any historical version
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.utils.logger import log_info, log_step, log_success, log_warning
from src.utils.storage import ROOT_DIR, compute_sha256, load_config


class ModelRegistry:
    def __init__(self, registry_dir: Path | None = None) -> None:
        cfg = load_config()
        self.cfg = cfg
        self.registry_dir = registry_dir or (ROOT_DIR / cfg["paths"]["registry_dir"])
        self.models_dir = self.registry_dir / "models"
        self.index_path = self.registry_dir / "registry_index.json"
        self.champion_path = self.registry_dir / "champion_metadata.json"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> dict[str, Any]:
        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "project": self.cfg["project"]["name"],
            "champion_version": None,
            "versions": [],
        }

    def _save_index(self, index_data: dict[str, Any]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)

    def _next_version_id(self, index_data: dict[str, Any]) -> str:
        n = len(index_data.get("versions", []))
        return f"v1.0.{n}"

    def register_and_evaluate_challenger(
        self,
        candidate_joblib_path: Path | None = None,
        eval_summary_path: Path | None = None,
    ) -> dict[str, Any]:
        """
        Registers the latest trained Challenger model, evaluates all Promotion Gates,
        and promotes it to Champion if all guardrails pass.
        """
        log_step("MODEL_REGISTRY", "ลงทะเบียนโมเดล Challenger และตรวจสอบเกณฑ์ Champion Promotion Gate...")
        cand_path = candidate_joblib_path or (self.registry_dir / "candidates" / "latest_challenger.joblib")
        eval_path = eval_summary_path or (ROOT_DIR / self.cfg["paths"]["evaluation_dir"] / "evaluation_summary.json")
        gold_path = ROOT_DIR / self.cfg["paths"]["gold_dir"] / "gold_features.parquet"

        if not cand_path.exists() or not eval_path.exists():
            raise FileNotFoundError("Missing challenger artifact or evaluation_summary.json. Run training step first.")

        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        index_data = self._load_index()
        version_id = self._next_version_id(index_data)
        versioned_model_file = self.models_dir / f"model_{version_id}.joblib"
        shutil.copy2(cand_path, versioned_model_file)

        model_sha256 = compute_sha256(versioned_model_file)
        gold_sha256 = compute_sha256(gold_path) if gold_path.exists() else "unknown"

        b0_test = eval_data["models"]["B0_Persistence"]["test"]
        b1_test = eval_data["models"]["B1_Historical_Average"]["test"]
        b2_val = eval_data["models"]["B2_LightGBM"]["val"]
        b2_test = eval_data["models"]["B2_LightGBM"]["test"]

        rules = self.cfg["registry"]["promotion_rules"]
        best_baseline_f1 = max(b0_test["f1_all"], b1_test["f1_all"])
        f1_gain_over_baseline = round(b2_test["f1_all"] - best_baseline_f1, 4)

        # Evaluate Promotion Gates
        gate_baseline = f1_gain_over_baseline >= rules["min_f1_improvement_over_baseline"]
        gate_onset = b2_test["onset_recall"] >= rules["min_onset_recall"]
        gate_mae = b2_test["mae_ratio"] <= rules["max_mae_ratio"]

        current_champ_ver = index_data.get("champion_version")
        current_champ_f1 = -1.0
        if current_champ_ver:
            for v in index_data["versions"]:
                if v["version"] == current_champ_ver:
                    current_champ_f1 = v["metrics"]["test"]["f1_all"]
                    break

        gate_beats_champion = b2_test["f1_all"] >= (current_champ_f1 - 0.002)
        all_gates_passed = bool(gate_baseline and gate_onset and gate_mae and gate_beats_champion)

        promotion_checks = {
            "gate_1_beats_baseline_f1": {
                "passed": bool(gate_baseline),
                "challenger_f1": b2_test["f1_all"],
                "best_baseline_f1": best_baseline_f1,
                "f1_gain": f1_gain_over_baseline,
                "required_gain": rules["min_f1_improvement_over_baseline"],
            },
            "gate_2_onset_recall_guardrail": {
                "passed": bool(gate_onset),
                "challenger_onset_recall": b2_test["onset_recall"],
                "baseline_b0_onset_recall": b0_test["onset_recall"],
                "min_required": rules["min_onset_recall"],
            },
            "gate_3_mae_plausibility": {
                "passed": bool(gate_mae),
                "challenger_mae": b2_test["mae_ratio"],
                "max_allowed": rules["max_mae_ratio"],
            },
            "gate_4_non_regression_vs_champion": {
                "passed": bool(gate_beats_champion),
                "previous_champion_version": current_champ_ver,
                "previous_champion_f1": current_champ_f1 if current_champ_f1 >= 0 else None,
                "challenger_f1": b2_test["f1_all"],
            },
        }

        new_stage = "Champion" if all_gates_passed else "Staging_Rejected"
        if all_gates_passed:
            for v in index_data["versions"]:
                if v.get("stage") == "Champion":
                    v["stage"] = "Archived"
            index_data["champion_version"] = version_id

        version_record = {
            "version": version_id,
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": new_stage,
            "model_backend": eval_data["model_backend"],
            "artifact_path": str(versioned_model_file.relative_to(ROOT_DIR)).replace("\\", "/"),
            "model_sha256": model_sha256,
            "gold_data_sha256": gold_sha256,
            "optimal_threshold": eval_data["optimal_threshold_val"],
            "num_features": eval_data["num_features"],
            "metrics": {
                "val": b2_val,
                "test": b2_test,
                "baseline_b0_test": b0_test,
                "baseline_b1_test": b1_test,
            },
            "promotion_decision": {
                "promoted_to_champion": all_gates_passed,
                "gates": promotion_checks,
            },
        }

        index_data["versions"].append(version_record)
        self._save_index(index_data)

        if all_gates_passed:
            champion_bundle_path = self.models_dir / "champion_model.joblib"
            shutil.copy2(versioned_model_file, champion_bundle_path)
            with open(self.champion_path, "w", encoding="utf-8") as f:
                json.dump(version_record, f, indent=2, ensure_ascii=False)
            log_info(
                f"รุ่นโมเดล {version_id} ผ่านทุกเกณฑ์ Guardrail! "
                f"(F1={b2_test['f1_all']:.4f} [+{f1_gain_over_baseline:.4f} เหนือ Baseline], "
                f"Onset Recall={b2_test['onset_recall']:.4f}, MAE={b2_test['mae_ratio']:.4f})"
            )
            log_success(f"เลื่อนสถานะ {version_id} เป็น [Champion] ใน Model Registry เรียบร้อย")
        else:
            log_warning(f"รุ่นโมเดล {version_id} ไม่ผ่านเกณฑ์บางข้อ คงสถานะไว้ที่ Staging_Rejected")

        return version_record

    def load_champion_model(self) -> dict[str, Any]:
        champ_file = self.models_dir / "champion_model.joblib"
        if not champ_file.exists():
            raise FileNotFoundError("No Champion model found in registry.")
        return joblib.load(champ_file)

    def rollback_to_version(self, target_version: str) -> dict[str, Any]:
        index_data = self._load_index()
        target_rec = None
        for v in index_data["versions"]:
            if v["version"] == target_version:
                target_rec = v
                break
        if target_rec is None:
            raise ValueError(f"Version {target_version} not found in registry.")

        for v in index_data["versions"]:
            if v.get("stage") == "Champion":
                v["stage"] = "Archived"
        target_rec["stage"] = "Champion"
        index_data["champion_version"] = target_version
        self._save_index(index_data)

        src_file = ROOT_DIR / target_rec["artifact_path"]
        shutil.copy2(src_file, self.models_dir / "champion_model.joblib")
        with open(self.champion_path, "w", encoding="utf-8") as f:
            json.dump(target_rec, f, indent=2, ensure_ascii=False)
        return target_rec


def run_model_registry_step() -> dict[str, Any]:
    registry = ModelRegistry()
    return registry.register_and_evaluate_challenger()
