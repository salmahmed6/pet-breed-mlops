from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.quality_gate import (
    QUALITY_THRESHOLD,
    compare_quality,
    load_production_top_1,
    load_top_1,
    promote_registered_version,
)


def write_report(path: Path, top_1: float) -> None:
    path.write_text(json.dumps({"top_1_accuracy": top_1}), encoding="utf-8")


def test_quality_gate_accepts_candidate_at_production_baseline() -> None:
    decision = compare_quality(0.80, 0.80)

    assert decision["accepted"] is True
    assert decision["delta"] == pytest.approx(0.0)
    assert decision["required_candidate_top_1_accuracy"] == pytest.approx(0.79)


def test_quality_gate_accepts_exact_minus_one_percent_boundary() -> None:
    decision = compare_quality(0.80, 0.79)

    assert QUALITY_THRESHOLD == -0.01
    assert decision["accepted"] is True
    assert decision["candidate_top_1_accuracy"] == pytest.approx(0.79)
    assert decision["required_candidate_top_1_accuracy"] == pytest.approx(0.79)


def test_exact_one_percent_drop_is_accepted() -> None:
    result = compare_quality(
        production_top_1=0.1,
        candidate_top_1=0.09,
    )

    assert result["accepted"] is True
    assert result["delta"] == pytest.approx(-0.01)


def test_quality_gate_rejects_below_minus_one_percent_boundary() -> None:
    decision = compare_quality(0.80, 0.789999)

    assert decision["accepted"] is False
    assert decision["delta"] == pytest.approx(-0.010001)


def test_quality_gate_rejects_large_quality_drop() -> None:
    decision = compare_quality(0.80, 0.70)

    assert decision["accepted"] is False
    assert decision["required_candidate_top_1_accuracy"] == pytest.approx(0.79)


def test_quality_gate_requires_fixed_threshold() -> None:
    with pytest.raises(ValueError, match="requires minimum_delta=-0.01"):
        compare_quality(0.80, 0.80, minimum_delta=0.0)


def test_quality_gate_loads_production_baseline_from_evaluation_artifact(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.json"
    write_report(baseline, 0.83)

    accuracy, source = load_production_top_1(baseline)

    assert accuracy == pytest.approx(0.83)
    assert source == f"artifact:{baseline}"


def test_quality_gate_reports_missing_candidate_report(tmp_path: Path) -> None:
    missing_candidate = tmp_path / "candidate.json"

    with pytest.raises(FileNotFoundError, match="Quality gate report not found"):
        load_top_1(missing_candidate)


def test_quality_gate_rejects_invalid_accuracy(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    write_report(baseline, 1.01)

    with pytest.raises(ValueError, match="between 0 and 1"):
        load_production_top_1(baseline)


class FakeMlflowClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def set_model_version_tag(
        self,
        model_name: str,
        version: str,
        key: str,
        value: str,
    ) -> None:
        self.calls.append(("tag", model_name, version, key, value))

    def set_registered_model_alias(
        self,
        model_name: str,
        alias: str,
        version: str,
    ) -> None:
        self.calls.append(("alias", model_name, alias, version))


def test_accepted_candidate_is_promoted_to_production() -> None:
    client = FakeMlflowClient()

    promote_registered_version(
        client,
        registered_model_name="PetBreedClassifier",
        version="7",
        production_top_1=0.80,
        candidate_top_1=0.79,
    )

    assert ("tag", "PetBreedClassifier", "7", "quality_gate_decision", "accepted") in client.calls
    assert ("tag", "PetBreedClassifier", "7", "top_1_accuracy", "0.79000000") in client.calls
    assert (
        "tag",
        "PetBreedClassifier",
        "7",
        "production_baseline_top_1_accuracy",
        "0.80000000",
    ) in client.calls
    assert ("alias", "PetBreedClassifier", "Production", "7") in client.calls
