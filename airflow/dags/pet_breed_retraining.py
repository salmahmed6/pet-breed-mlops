"""Airflow DAG for drift-triggered Pet Breed model retraining."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from airflow.providers.standard.operators.python import PythonOperator, ShortCircuitOperator

from airflow import DAG

PROJECT_ROOT = Path("/opt/airflow/project")
REPORT_DIR = PROJECT_ROOT / "reports" / "retraining"
CANDIDATE_DIR = PROJECT_ROOT / "artifacts" / "retraining" / "candidate"
CANDIDATE_CHECKPOINT = CANDIDATE_DIR / "resnet18_best.pt"
CANDIDATE_REPORT = REPORT_DIR / "candidate_evaluation.json"
QUALITY_DECISION = REPORT_DIR / "quality_gate_decision.json"
TRAINING_CONFIG = PROJECT_ROOT / "configs" / "training_airflow.yaml"
MLFLOW_CONFIG = PROJECT_ROOT / "configs" / "mlflow.yaml"


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def check_drift(**context) -> bool:
    """Run the monitoring scorecard and return whether retraining is required."""
    _run(
        [
            sys.executable,
            "-m",
            "scripts.drift_trigger",
            "--decision-file",
            str(REPORT_DIR / "drift_decision.json"),
        ]
    )
    decision = json.loads((REPORT_DIR / "drift_decision.json").read_text(encoding="utf-8"))
    context["ti"].xcom_push(key="drift_decision", value=decision)
    return bool(decision["retraining_required"])


def retrain() -> None:
    _run(
        [
            sys.executable,
            "-m",
            "scripts.run_retraining",
            "--model",
            "resnet18",
            "--config",
            str(TRAINING_CONFIG),
            "--output-dir",
            str(CANDIDATE_DIR),
            "--report",
            str(REPORT_DIR / "candidate_training.json"),
        ]
    )


def evaluate_candidate() -> None:
    _run(
        [
            sys.executable,
            "-m",
            "scripts.evaluate_candidate",
            "--model",
            "resnet18",
            "--checkpoint",
            str(CANDIDATE_CHECKPOINT),
            "--config",
            str(TRAINING_CONFIG),
            "--output",
            str(CANDIDATE_REPORT),
        ]
    )


def quality_gate() -> None:
    _run(
        [
            sys.executable,
            "-m",
            "scripts.quality_gate",
            "--baseline",
            str(PROJECT_ROOT / "reports" / "quality_baseline.json"),
            "--candidate",
            str(CANDIDATE_REPORT),
            "--decision",
            str(QUALITY_DECISION),
            "--mlflow-config",
            str(MLFLOW_CONFIG),
        ]
    )


with DAG(
    dag_id="pet_breed_drift_retraining",
    description="Drift -> candidate retraining -> evaluation -> quality gate -> promotion",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "pet-breed", "retraining"],
) as dag:
    drift_check = ShortCircuitOperator(
        task_id="check_drift",
        python_callable=check_drift,
    )

    retraining = PythonOperator(
        task_id="retrain_candidate",
        python_callable=retrain,
    )

    evaluate = PythonOperator(
        task_id="evaluate_candidate",
        python_callable=evaluate_candidate,
    )

    gate = PythonOperator(
        task_id="quality_gate",
        python_callable=quality_gate,
    )

    drift_check >> retraining >> evaluate >> gate
