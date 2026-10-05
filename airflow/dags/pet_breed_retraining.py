"""Airflow DAG for drift-triggered Pet Breed model retraining."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.trigger_rule import TriggerRule


PROJECT_ROOT = Path("/opt/airflow/project")
REPORT_DIR = PROJECT_ROOT / "reports" / "retraining"


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
            "--output-dir",
            "artifacts/retraining/candidate",
            "--report",
            "reports/retraining/candidate_evaluation.json",
        ]
    )


def quality_gate() -> None:
    _run(
        [
            sys.executable,
            "-m",
            "scripts.quality_gate",
            "--baseline",
            "reports/quality_baseline.json",
            "--candidate",
            "reports/retraining/candidate_evaluation.json",
            "--minimum-delta",
            "0.0",
        ]
    )


with DAG(
    dag_id="pet_breed_drift_retraining",
    description="Drift -> candidate retraining -> evaluation -> quality gate",
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
        python_callable=lambda: Path(
            PROJECT_ROOT / "reports/retraining/candidate_evaluation.json"
        ).exists(),
    )

    gate = PythonOperator(
        task_id="quality_gate",
        python_callable=quality_gate,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    drift_check >> retraining >> evaluate >> gate
