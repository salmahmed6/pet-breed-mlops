# Grafana Dashboard and Drift Alerting

## Local architecture

BentoML / metrics exporter -> Prometheus -> Grafana
                                 |
                                 +-> Alert rule -> Alertmanager -> /retrain webhook

The stack is local-only and consumes the metrics from Issue #44.

## Start

1. Start the metrics exporter:
   python -m scripts.export_prometheus

2. Start the local retraining webhook:
   python -m scripts.retraining_webhook

3. Start Prometheus, Alertmanager and Grafana:
   docker compose -f docker-compose.monitoring.yml up -d

Open Prometheus at http://localhost:9090, Alertmanager at http://localhost:9093, and Grafana at http://localhost:3000.

The Grafana dashboard is provisioned automatically as "Pet Breed MLOps Monitoring". A fresh local Grafana container uses admin/admin unless changed.

## Drift alert

The Issue #43 scorecard converts the configured detector thresholds in configs/monitoring.yaml into pet_breed_drift_flag values. Prometheus alerts when:

    max(pet_breed_drift_flag) >= 1

This keeps the detector threshold in the existing monitoring configuration instead of introducing an undocumented second threshold. The alert waits 30 seconds before firing.

Alertmanager forwards the alert to:

    http://host.docker.internal:9001/retrain

The local webhook records the Alertmanager payload at reports/monitoring/retraining_alert.json.

## Test the retraining path

With the webhook running:

    Invoke-RestMethod -Uri http://127.0.0.1:9001/retrain -Method Post -ContentType "application/json" -Body '{"alerts":[{"status":"firing","labels":{"alertname":"PetBreedDriftDetected"}}]}'

Then inspect:

    Get-Content reports/monitoring/retraining_alert.json

For a real alert demonstration, keep the exporter and webhook running, start the Docker stack, and inspect the Alerts pages in Prometheus and Alertmanager. The drift values come from the actual local scorecard.

## Dashboard panels

The provisioned dashboard contains request rate, error rate, p95 latency, latest confidence, abstentions, drift score, and drift flags. Panels query Prometheus directly.
