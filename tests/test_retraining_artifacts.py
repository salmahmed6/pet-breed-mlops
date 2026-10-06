from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pet_breed_mlops.data.loaders import create_dataloaders
from scripts.run_retraining import _promote_run


def test_promote_run_replaces_previous_candidate_only_after_completion(tmp_path: Path) -> None:
    output_dir = tmp_path / "candidate"
    output_dir.mkdir()
    stale_checkpoint = output_dir / "resnet18_best.pt"
    stale_checkpoint.write_bytes(b"stale")

    run_dir = tmp_path / ".candidate-run-test"
    run_dir.mkdir()
    completed_checkpoint = run_dir / "resnet18_best.pt"
    completed_checkpoint.write_bytes(b"completed")
    completed_history = run_dir / "resnet18_history.json"
    completed_history.write_text("{}", encoding="utf-8")

    _promote_run(run_dir, output_dir)

    promoted_checkpoint = output_dir / "resnet18_best.pt"
    promoted_history = output_dir / "resnet18_history.json"
    assert promoted_checkpoint.read_bytes() == b"completed"
    assert promoted_history.read_text(encoding="utf-8") == "{}"
    assert not run_dir.exists()
    assert not (output_dir.parent / ".candidate.previous").exists()
    assert promoted_checkpoint.read_bytes() != b"stale"


def test_promote_run_keeps_existing_candidate_on_failure(tmp_path: Path) -> None:
    output_dir = tmp_path / "candidate"
    output_dir.mkdir()
    stale_checkpoint = output_dir / "resnet18_best.pt"
    stale_checkpoint.write_bytes(b"stale")
    failed_run = tmp_path / ".candidate-run-missing"

    with pytest.raises(FileNotFoundError):
        _promote_run(failed_run, output_dir)

    assert stale_checkpoint.read_bytes() == b"stale"


def test_airflow_training_configuration_is_resource_safe() -> None:
    config = yaml.safe_load(Path("configs/training_airflow.yaml").read_text(encoding="utf-8"))

    assert config["data"]["num_workers"] == 0
    assert config["training"]["batch_size"] == 8
    assert config["training"]["epochs"] == 5
    assert config["data"]["image_size"] == 224


def test_dataloaders_disable_worker_and_pinning_overhead() -> None:
    loaders = create_dataloaders(
        manifest_path="data/processed/manifest.json",
        image_size=224,
        batch_size=8,
        num_workers=0,
    )

    for loader in loaders:
        assert loader.batch_size == 8
        assert loader.num_workers == 0
        assert loader.pin_memory is False
        assert loader.persistent_workers is False
