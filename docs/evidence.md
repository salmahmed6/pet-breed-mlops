# Final End-to-End Evidence

Issue #48 is the final integration/evidence closeout for the Track C Pet Breed MLOps project. It does not duplicate Sprint 1-5 implementation work. This document records the reproducible workflow, evidence locations, and the verification state that must be completed from the local environment.

## 1. Final workflow

```text
DVC-managed data
  -> monitoring scorecard
  -> drift decision / alert
  -> Airflow retraining
  -> candidate evaluation
  -> MLflow quality gate
  -> Production alias promotion OR rejection
```

The Airflow DAG implements the orchestration chain:

```text
check_drift -> retrain_candidate -> evaluate_candidate -> quality_gate
```

Monitoring also has an alert path:

```text
metrics exporter -> Prometheus -> Alertmanager -> /retrain webhook
```

## 2. Reproducible local verification

### Prerequisites

- Python project dependencies installed.
- DVC data available locally.
- `artifacts/models/resnet18_best.pt` available.
- Docker Desktop running for the Airflow and monitoring stacks.
- MLflow configured through `configs/mlflow.yaml` for a real promotion run.

### Monitoring data and drift evidence

From the repository root:

```powershell
python -m scripts.run_drift_scorecard
python -m scripts.prepare_monitoring_data
python -m scripts.run_evidently_report
```

Verify these generated artifacts exist and were created by the current run:

- `reports/monitoring/drift_scorecard.json`
- `reports/monitoring/drift_scorecard.csv`
- `reports/monitoring/reference_monitoring.csv`
- `reports/monitoring/current_monitoring.csv`
- `reports/monitoring/evidently_drift.html`

### Monitoring and alert path

Terminal 1:

```powershell
python -m scripts.export_prometheus
```

Terminal 2:

```powershell
python -m scripts.retraining_webhook
```

Terminal 3:

```powershell
docker compose -f docker-compose.monitoring.yml up -d
```

Verify Prometheus, Alertmanager, and Grafana are reachable, inspect the drift alert, and confirm the webhook creates:

```text
reports/monitoring/retraining_alert.json
```

A manually posted alert may validate the webhook receiver, but it must not be described as proof that the real drift detector fired. Real alert evidence requires the exporter, Prometheus rule, and Alertmanager path to operate from the generated monitoring scorecard.

### Airflow retraining flow

From the repository root:

```powershell
docker compose -f docker-compose.airflow.yml build
docker compose -f docker-compose.airflow.yml up -d

docker compose -f docker-compose.airflow.yml exec airflow python -m airflow dags list
docker compose -f docker-compose.airflow.yml exec airflow python -m airflow dags show pet_breed_drift_retraining
docker compose -f docker-compose.airflow.yml exec airflow python -m airflow dags trigger pet_breed_drift_retraining
docker compose -f docker-compose.airflow.yml exec airflow python -m airflow dags list-runs
```

For the selected run, inspect task states/logs and verify these artifacts:

- `reports/retraining/drift_decision.json`
- `reports/retraining/candidate_evaluation.json`
- `reports/retraining/quality_gate_decision.json`
- `artifacts/retraining/candidate/resnet18_best.pt`

The retraining implementation records that fitting uses `train`, model selection uses `val`, and `test` remains protected.

### Quality-gate and registry verification

Run the quality gate using the generated candidate evaluation:

```powershell
python -m scripts.quality_gate --baseline reports/quality_baseline.json --candidate reports/retraining/candidate_evaluation.json --decision reports/retraining/quality_gate_decision.json --mlflow-config configs/mlflow.yaml
```

Verify the decision contains:

- production top-1 accuracy;
- candidate top-1 accuracy;
- required candidate accuracy;
- candidate delta;
- fixed threshold `-0.01`;
- baseline source;
- final accepted/rejected decision.

For an accepted candidate, verify MLflow registered a new version and moved the `Production` alias. For a rejected candidate, verify the existing `Production` alias did not change.

## 3. Final evidence index

| Area | Evidence location | Verification status |
|---|---|---|
| DVC/data pipeline | `dvc.yaml`, `dvc.lock`, `data/raw/oxford-iiit-pet.dvc` | Verify locally with `dvc status` / `dvc dag` |
| Training / evaluation | `reports/` and `artifacts/models/` | Use actual generated reports/checkpoints |
| MLflow experiments | `reports/mlflow_experiment_report.md` and local MLflow UI | Existing project evidence; refresh UI evidence when needed |
| Calibration / abstention | `reports/calibration/`, `docs/calibration.md` | Verify generated calibration report |
| Serving | `docs/serving.md`, serving tests | Verify local endpoint/contract |
| Optimization | `reports/optimization/`, `docs/optimization.md` | Verify measured benchmark artifacts; hardware-dependent results require local execution |
| Load testing | `reports/serving/`, `docs/load_testing.md` | Verify actual Locust/batch reports |
| Drift monitoring | `reports/monitoring/`, `docs/monitoring*.md` | Verify actual generated scorecard + Evidently report |
| Alerting | `configs/prometheus/alerts.yml`, `configs/alertmanager.yml`, `reports/monitoring/retraining_alert.json` | Verify real alert delivery |
| Retraining | `airflow/dags/pet_breed_retraining.py`, `reports/retraining/` | Verify a real Airflow run |
| Quality gate | `reports/quality_gate_decision.json`, `docs/quality_gate_promotion.md` | Verify real candidate decision and MLflow registry state |
| Peer review | External reviewer evidence / screenshots | Pending until available |

## 4. Current repository evidence already committed

The repository already contains generated or documented evidence for monitoring, calibration, optimization, and the previous Airflow validation/evidence package. These artifacts should be reused rather than regenerated into duplicate Sprint 1-5 implementations.

The existing Airflow evidence package is under `docs/issue-46-airflow/`, including the evidence report document and supporting Markdown.

## 5. What cannot be claimed before the local run

Do not claim that the final end-to-end demonstration succeeded solely because the scripts and tests exist. The following require actual local verification:

- one complete drift-triggered Airflow run;
- the resulting candidate evaluation;
- the resulting quality-gate decision;
- actual MLflow registry/Production alias state;
- fresh Grafana/Prometheus screenshots if required by the final submission;
- peer-review participation evidence when available.

This repository deliberately treats missing runtime evidence as unverified rather than inventing metrics or screenshots.

## 6. Final checklist

- [ ] End-to-end drift -> alert -> retraining -> quality gate -> registry flow executed.
- [ ] Real generated artifacts linked and inspected.
- [ ] README/runbook commands executed from a clean pull.
- [ ] DVC, MLflow, CI/CD, serving, optimization, load testing, monitoring, and retraining states verified.
- [ ] Required screenshots/reports linked.
- [ ] Peer-review evidence added when available.
- [ ] Issue #48 acceptance criteria can be checked from actual run output.
