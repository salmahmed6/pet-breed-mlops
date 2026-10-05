# Airflow Drift-Triggered Retraining

## Purpose

Sprint 5 Issue #46 adds an Airflow workflow that connects monitoring drift detection to candidate retraining and a quality gate.

The workflow is:

1. Run the existing drift scorecard.
2. Persist a drift decision.
3. Skip retraining when no drift is detected.
4. Retrain a candidate model when drift is detected.
5. Persist candidate validation metrics and checkpoint metadata.
6. Run the existing quality gate against `reports/quality_baseline.json`.

## Data protection

The candidate training path uses the existing trainer. The trainer fits only on the `train` split and uses `val` for model selection. The `test` split remains protected from fitting.

The drift scorecard may inspect the test split for monitoring comparison; this is not a fitting operation and does not make test samples available to the optimizer.

## Local execution

Build and start the isolated Airflow environment:

```powershell
docker compose -f docker-compose.airflow.yml build
docker compose -f docker-compose.airflow.yml up -d
```

Open Airflow at http://localhost:8080.

Check the DAG:

```powershell
docker compose -f docker-compose.airflow.yml exec airflow airflow dags list
docker compose -f docker-compose.airflow.yml exec airflow airflow dags show pet_breed_drift_retraining
```

Trigger a manual run:

```powershell
docker compose -f docker-compose.airflow.yml exec airflow airflow dags trigger pet_breed_drift_retraining
```

Inspect the run:

```powershell
docker compose -f docker-compose.airflow.yml exec airflow airflow dags list-runs -d pet_breed_drift_retraining
docker compose -f docker-compose.airflow.yml logs --tail=200 airflow
```

## Evidence artifacts

A successful drift-triggered run should produce:

- `reports/monitoring/drift_scorecard.json`
- `reports/retraining/drift_decision.json`
- `reports/retraining/candidate_evaluation.json`
- `artifacts/retraining/candidate/resnet18_best.pt`

The DAG is manually triggerable and can be invoked by the monitoring integration in a later orchestration layer.

## Validation

In the local project environment:

```powershell
pytest -q tests/test_airflow_dag.py
```

The AST and policy tests run without Airflow installed; the full DagBag validation is enabled automatically when Airflow is available.

## Notes

The Docker image is intentionally isolated from the project's `.venv`. Apache Airflow's official Docker image is used as the orchestration runtime, while the project ML dependencies are installed into that image.
