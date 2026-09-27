from pathlib import Path

import yaml


def test_batch_inference_script_exists():
    assert Path("scripts/batch_inference.py").exists()


def test_load_test_config_records_required_dimensions():
    config = yaml.safe_load(
        Path("configs/load_test.yaml").read_text(encoding="utf-8")
    )
    assert (
        config["concurrency"] > 0
        and config["duration_seconds"] > 0
        and config["batch_size"] > 0
    )
    assert config["endpoint"] and config["payload"]
