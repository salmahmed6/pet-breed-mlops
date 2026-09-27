from pathlib import Path

import pytest

from scripts.quality_gate import check_quality


def write_report(path: Path, top_1: float) -> None:
    path.write_text(f'{{"top_1_accuracy": {top_1}}}', encoding="utf-8")


def test_quality_gate_passes_at_baseline(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    write_report(baseline, 0.80)
    write_report(candidate, 0.80)

    check_quality(baseline, candidate)


def test_quality_gate_passes_above_baseline(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    write_report(baseline, 0.80)
    write_report(candidate, 0.81)

    check_quality(baseline, candidate)


def test_quality_gate_rejects_drop_below_baseline(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    write_report(baseline, 0.80)
    write_report(candidate, 0.79)

    with pytest.raises(SystemExit, match="Model quality gate failed"):
        check_quality(baseline, candidate)
