"""Validation tests for the Airflow retraining DAG."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


DAG_PATH = Path("airflow/dags/pet_breed_retraining.py")


def test_dag_source_is_valid_python() -> None:
    ast.parse(DAG_PATH.read_text(encoding="utf-8"))


def test_dag_import_and_structure() -> None:
    airflow = pytest.importorskip("airflow")
    del airflow

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


def test_retraining_does_not_fit_test_split() -> None:
    source = Path("scripts/run_retraining.py").read_text(encoding="utf-8")
    assert '"fit_split": "train"' in source
    assert '"protected_split": "test"' in source
