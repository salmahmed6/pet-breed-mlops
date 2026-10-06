"""Validation tests for the Airflow retraining DAG."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

DAG_PATH = Path("airflow/dags/pet_breed_retraining.py")


def test_dag_source_is_valid_python() -> None:
    ast.parse(DAG_PATH.read_text(encoding="utf-8"))


def test_dag_import_and_structure() -> None:
    pytest.importorskip("airflow.models")

    from airflow.models import DagBag

    dagbag = DagBag(dag_folder=str(DAG_PATH.parent), include_examples=False)
    assert not dagbag.import_errors
    dag = dagbag.get_dag("pet_breed_drift_retraining")
    assert dag is not None
    assert dag.task_ids == [
        "check_drift",
        "retrain_candidate",
        "evaluate_candidate",
        "quality_gate",
    ]
    assert dag.get_task("check_drift").downstream_task_ids == {"retrain_candidate"}
    assert dag.get_task("retrain_candidate").downstream_task_ids == {"evaluate_candidate"}
    assert dag.get_task("evaluate_candidate").downstream_task_ids == {"quality_gate"}


def test_dag_uses_absolute_paths_and_airflow_3_operators() -> None:
    source = DAG_PATH.read_text(encoding="utf-8")
    assert "airflow.providers.standard.operators.python" in source
    assert "TriggerRule" not in source
    assert 'Path("/opt/airflow/project")' in source
    assert "str(CANDIDATE_CHECKPOINT)" in source
    assert "str(CANDIDATE_REPORT)" in source


def test_dag_uses_airflow_resource_profile_and_separate_reports() -> None:
    source = DAG_PATH.read_text(encoding="utf-8")
    assert "training_airflow.yaml" in source
    assert "candidate_training.json" in source
    assert "candidate_evaluation.json" in source


def test_retraining_does_not_fit_test_split() -> None:
    source = Path("scripts/run_retraining.py").read_text(encoding="utf-8")
    assert '"fit_split": "train"' in source
    assert '"protected_split": "test"' in source
