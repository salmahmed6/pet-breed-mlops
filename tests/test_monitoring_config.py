import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_monitoring_stack_configs() -> None:
    prometheus = yaml.safe_load((ROOT / "configs/prometheus.yml").read_text())
    alerts = yaml.safe_load((ROOT / "configs/prometheus/alerts.yml").read_text())
    alertmanager = yaml.safe_load((ROOT / "configs/alertmanager.yml").read_text())

    assert prometheus["rule_files"] == ["/etc/prometheus/alerts.yml"]
    rule = alerts["groups"][0]["rules"][0]
    assert rule["alert"] == "PetBreedDriftDetected"
    assert "pet_breed_drift_flag" in rule["expr"]
    assert alertmanager["receivers"][0]["webhook_configs"][0]["url"].endswith("/retrain")


def test_dashboard_contains_required_panels() -> None:
    dashboard = json.loads(
        (ROOT / "configs/grafana/dashboards/pet-breed-monitoring.json").read_text()
    )
    titles = {panel["title"] for panel in dashboard["panels"]}
    assert {
        "Request Rate",
        "Error Rate",
        "Inference Latency P95",
        "Latest Prediction Confidence",
        "Abstentions",
        "Drift Score",
        "Drift Flags",
    }.issubset(titles)
