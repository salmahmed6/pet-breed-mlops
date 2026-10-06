# Sprint 5 Issue #46 — Airflow Retraining Evidence Report

## PR

- PR: #57
- Title: [S5][E1] Implement Airflow drift-triggered retraining DAG
- Target: dev
- Merge commit: a3f57271e1a6390dd431cd5299d9b329cee7e272

## Completed work

- Added isolated Airflow 3.3.2 runtime and Docker Compose setup.
- Added the `pet_breed_drift_retraining` DAG.
- Implemented drift check → candidate retraining → candidate evaluation → quality gate.
- Connected retraining decisions to the existing Sprint 4 drift scorecard.
- Added candidate retraining/evaluation scripts.
- Added DAG validation and retraining-artifact tests.
- Added CPU/resource-safe training configuration.
- Kept train for fitting, validation for selection, and test protected from fitting.
- Added the Airflow runbook.

## Real execution

A real manual Airflow run was triggered:

`manual__2026-10-06T11:37:23.234757+00:00`

The drift-check task completed successfully and detected drift. Candidate retraining then started.

The retraining task was terminated because the local Docker environment ran out of memory. The recorded result was `SIGKILL: 9` and Docker inspection showed `OOMKilled=true`.

A partial checkpoint existed, but it is not treated as successful candidate evidence. Evaluation and quality-gate stages were not reached.

## Acceptance status

| Acceptance criterion | Status |
|---|---|
| Airflow DAG importable / validated | Completed |
| Drift condition can trigger retraining | Completed |
| Candidate artifacts from a complete real run | Partial — retraining was OOM-killed |
| Protected test split | Implemented |
| Full end-to-end DAG execution | Not completed due to local memory limitation |

## Screenshot coverage

The five screenshots captured during the real Airflow execution cover:

1. Airflow DAG Runs.
2. Drift Check Running.
3. Drift Check Succeeded / Retraining Running.
4. Retraining Still Running.
5. Retraining Failed; Evaluation and Quality Gate Not Executed.

The screenshots are embedded in the DOCX report generated during the project work session.
