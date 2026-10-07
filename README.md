# Pet Breed Classification MLOps

An end-to-end **MLOps project for 37-class pet breed image classification** using the **Oxford-IIIT Pet dataset**.

This repository documents the complete journey from data and model development to reproducibility, experiment tracking, production serving, optimization, monitoring, automated retraining, quality gates, and CI/CD.

The goal was not only to train a model, but to build the **engineering system around the model** so it can be reproduced, evaluated, served, monitored, and safely updated.

---
# # 📌 Project Overview
# ## Problem

Build a production-oriented machine learning system that classifies pet images into **37 breed classes**.
# ## Dataset

- Oxford-IIIT Pet Dataset
- 37 pet breed classes
- Image classification task
- DVC-managed data and pipeline artifacts
# ## Final MLOps lifecycle

~~~text
Data

DVC versioning & reproducible pipeline

Training & evaluation

MLflow experiment tracking

Calibration & confidence-based abstention

Model serving

ONNX / OpenVINO / TensorRT optimization

Load testing

Drift monitoring

Prometheus / Grafana / Alertmanager

Drift-triggered retraining

Candidate evaluation

Quality Gate

MLflow Model Registry

Production promotion
~~~

---
# What We Built

The project was developed incrementally, with each stage adding another part of a complete MLOps lifecycle.
# # 1. Project & ML Foundations

The project started with a reproducible Python ML codebase:

- Python 3.11+
- PyTorch / Torchvision
- Scikit-learn
- Pydantic and YAML-based configuration
- Structured source code under **src/**
- Automated tests under **tests/**
- Reusable commands under **scripts/**
- Centralized configuration under **configs/**

The structure was designed so experimentation and production-oriented code could coexist without becoming a collection of notebooks and one-off scripts.

---
# # 2. Data Versioning & Reproducibility

**DVC** was introduced to make the data and preprocessing pipeline reproducible.

The DVC pipeline covers:

1. Dataset preparation
2. Train/validation/test split generation
3. Manifest creation
4. Corrupted-data generation for robustness and monitoring experiments

Key files:

- **dvc.yaml**
- **dvc.lock**
- **.dvc/**
- **data/**

The important principle is that data transformations are treated as versioned pipeline stages rather than manual local steps.

📖 [Dataset documentation](docs/dataset.md)

---
# # 3. Model Training & Evaluation

The core classification pipeline uses PyTorch models and reproducible training/evaluation scripts.

Supported model work includes:

- ResNet18
- MobileNetV3 Small
- ResNet50

The training lifecycle separates:

- **train** → model fitting
- **validation** → model selection and quality decisions
- **test** → protected final evaluation

Evaluation includes:

- Top-1 accuracy
- Macro F1
- Loss
- Per-class analysis where applicable

Generated model artifacts and reports are kept separate from source code.

---
# # 4. Experiment Tracking with MLflow

**MLflow** was introduced to make model development traceable.

The project tracks:

- experiments
- parameters
- metrics
- model artifacts
- candidate versions
- production model versions

MLflow is also used later for model registry and production promotion.

📖 [Experiments documentation](docs/experiments.md)

---
# # 5. Model Calibration & Abstention

A production classifier should not blindly trust every prediction.

The project therefore includes:

- confidence analysis
- calibration
- confidence thresholds
- abstention behavior

The objective is to provide a safer inference path when the model is not sufficiently confident.

📖 [Calibration documentation](docs/calibration.md)

---
# # 6. Production Model Serving

The trained model was moved from a training artifact into a serving workflow.

The serving work covers:

- inference contracts
- request/response behavior
- model loading
- inference endpoints
- serving tests
- deployment-oriented structure

The project also explores production inference infrastructure and serving runtimes.

📖 [Serving documentation](docs/serving.md)

---
# # 7. Model Optimization

The project goes beyond a PyTorch checkpoint and explores deployment optimization.

The optimization stage includes:

- ONNX export
- ONNX Runtime
- OpenVINO
- TensorRT-oriented deployment work
- comparison of optimized inference paths

The purpose is to understand the trade-offs between:

- model accuracy
- inference latency
- throughput
- deployment hardware/runtime

📖 [Optimization documentation](docs/optimization.md)

---
# # 8. Triton & Production Inference

The serving stage was extended toward production inference infrastructure, including **NVIDIA Triton Inference Server** concepts and deployment paths.

This stage focuses on understanding how a trained model becomes a managed inference service rather than simply calling a Python function.

---
# # 9. Load Testing & Performance Evidence

Inference performance is tested using reproducible workloads rather than manually reported numbers.

The project includes:

- batch inference benchmarking
- Locust load testing
- configurable users/concurrency
- request failure tracking
- requests/second
- latency percentiles
- generated performance reports

Example:

~~~powershell
python -m scripts.batch_inference --checkpoint artifacts/models/resnet18_best.pt --batch-size 32 --max-batches 50 --output reports/serving/batch_inference.json
~~~

📖 [Load testing documentation](docs/load_testing.md)

---
# Monitoring & Observability

A production ML system must be monitored after deployment.

The monitoring layer covers several signals.
# # Data / Feature Drift

The project generates monitoring data and drift scorecards using:

- Evidently
- embedding drift
- pixel/data drift
- confidence drift indicators

Generated evidence is stored under **reports/monitoring/**.

📖 [Drift monitoring documentation](docs/monitoring_drift.md)
# # Metrics & Alerting

The monitoring stack uses:

- Prometheus
- Grafana
- Alertmanager
- Prometheus metrics exporter
- retraining webhook

The intended alert path is:

~~~text
Monitoring metrics

Prometheus

Alert rule

Alertmanager

Retraining webhook
~~~

📖 [Monitoring documentation](docs/monitoring.md)  
📖 [Evidently + Prometheus documentation](docs/monitoring_evidently_prometheus.md)  
📖 [Grafana alerting documentation](docs/grafana_alerting.md)

---
# Automated Retraining

The project closes the ML lifecycle loop.

When drift indicates that the deployed model may no longer be reliable:

~~~text
Drift detected

Retraining required

Train candidate

Evaluate candidate

Quality Gate

Accept → Production
Reject → Keep current Production model
~~~

The orchestration layer uses **Apache Airflow**.

The DAG follows:

~~~text
check_drift

retrain_candidate

evaluate_candidate

quality_gate
~~~

📖 [Airflow retraining documentation](docs/airflow_retraining.md)

---
# Model Quality Gate & Promotion

A new model must pass an explicit quality decision before becoming production.

The quality gate records:

- production accuracy
- candidate accuracy
- required candidate accuracy
- minimum allowed delta
- final decision
- MLflow model version
- production alias state

Promotion follows:

~~~text
Candidate

Evaluation

Quality Gate
   ├── Rejected → keep current Production

   └── Accepted → register/promote candidate
~~~

📖 [Quality gate & promotion documentation](docs/quality_gate_promotion.md)

---
# Final Verified Candidate Run

The final candidate workflow was executed locally during the project closeout.

The candidate ResNet18 run reached:

| Metric | Validation result |
|---|---:|
| Top-1 Accuracy | **91.85%** |
| Macro F1 | **91.63%** |

The quality gate accepted the candidate and promoted **MLflow model version 7** to the **Production** alias in the verified MLflow environment.
# ## Important baseline note

The configured quality-gate baseline in this final verification was the Sprint 2 smoke-test baseline:

~~~text
Top-1 accuracy = 0.10
~~~

Therefore, the large numerical delta reported by the gate represents improvement over that **deliberately limited smoke-test baseline**, not a claim of an equivalent production-model improvement.

The repository keeps this distinction explicit so generated evidence is not misrepresented.

---
# Final End-to-End Workflow

The final integration stage connects the individual project components into one lifecycle:

~~~text
DVC-managed data

Monitoring / Drift Scorecard

Drift Decision

Airflow Retraining

Candidate Model

Candidate Evaluation

MLflow Quality Gate


│                               │
Reject                        Accept
│                               │
Keep Production        Register / Promote

                         Production Alias
~~~

📖 [Final end-to-end evidence & verification](docs/final_end_to_end_evidence.md)

---
# Repository Structure

~~~text
pet-breed-mlops/

├── .github/
│   └── workflows/          # Independent CI/CD workflows

├── airflow/                # Airflow DAGs and orchestration
├── artifacts/              # Generated model/retraining artifacts
├── configs/                # Training, MLflow, monitoring and runtime config
├── data/                   # DVC-managed data and processed artifacts
├── docker/                 # Deployment/container configuration
├── docs/                   # Project documentation and evidence
├── models/                 # Model-related assets
├── reports/                # Generated evaluation/monitoring evidence
├── scripts/                # Reproducible project commands
├── src/                    # Main Python package
├── tests/                  # Automated tests and fixtures

├── dvc.yaml                # DVC pipeline
├── dvc.lock                # Locked DVC pipeline state
├── Dockerfile
├── docker-compose.airflow.yml
├── docker-compose.monitoring.yml
├── locustfile.py
├── pyproject.toml
└── README.md
~~~

---
# Technology Stack

| Area | Technologies |
|---|---|
| Language | Python 3.11+ |
| Deep Learning | PyTorch, Torchvision |
| ML | Scikit-learn |
| Data Versioning | DVC |
| Experiment Tracking | MLflow |
| Model Serving | BentoML, Triton-oriented serving |
| Optimization | ONNX, ONNX Runtime, OpenVINO, TensorRT |
| Monitoring | Evidently, Prometheus, Grafana |
| Alerting | Alertmanager |
| Orchestration | Apache Airflow |
| Load Testing | Locust |
| Containers | Docker, Docker Compose |
| Testing | Pytest |
| Code Quality | Ruff, Mypy |
| CI/CD | GitHub Actions |

---
# Local Setup
# # Requirements

- Python 3.11–3.13
- Git
- Docker Desktop for Airflow and monitoring components
- DVC-managed dataset available locally
- MLflow configuration for registry/promotion workflows
# # Install

~~~bash
git clone https://github.com/salmahmed6/pet-breed-mlops.git
cd pet-breed-mlops

python -m venv .venv
~~~
# ## Windows

~~~powershell
.\\.venv\\Scripts\\Activate.ps1
~~~
# ## Linux/macOS

~~~bash
source .venv/bin/activate
~~~

Install the project and development dependencies:

~~~bash
python -m pip install -e ".[dev]"
~~~

For load testing:

~~~bash
python -m pip install -e ".[loadtest]"
~~~

---
# Reproducibility Commands
# # DVC

~~~bash
dvc status
dvc dag
dvc repro
~~~
# # Candidate Evaluation

~~~bash
python -m scripts.evaluate_candidate --model resnet18 --checkpoint artifacts/retraining/candidate/resnet18_best.pt --config configs/training.yaml --output reports/retraining/candidate_evaluation.json --split val
~~~
# # Quality Gate

~~~bash
python -m scripts.quality_gate --baseline reports/quality_baseline.json --candidate reports/retraining/candidate_evaluation.json --decision reports/retraining/quality_gate_decision.json --mlflow-config configs/mlflow.yaml
~~~
# # Drift Decision

~~~bash
python -m scripts.drift_trigger --decision-file reports/retraining/drift_decision.json
~~~

> On Windows PowerShell, use the PowerShell backtick for multi-line command continuation, or run the command on one line.

---
# Quality Checks

The project keeps linting, formatting, testing, and quality-gate validation as separate CI checks.

Run locally:

~~~bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q -m "not data"
python -m mypy src
~~~

The CI pipeline is intentionally split into independent workflows so failures are isolated and easy to identify.

Current workflows include:

- **Lint**
- **Format**
- **Tests**
- **Quality Gate**
- **Docker Build**
- **Publish Image**

The publish workflow runs for **main** and publishes the container image to GitHub Container Registry.

---
# Docker & Infrastructure

The repository includes Docker support for:

- model/application serving
- Airflow retraining orchestration
- monitoring infrastructure

Airflow:

~~~bash
docker compose -f docker-compose.airflow.yml up -d
~~~

Monitoring:

~~~bash
docker compose -f docker-compose.monitoring.yml up -d
~~~

The exact verification steps are documented in the relevant files under **docs/**.

---
# Documentation

The **docs/** directory contains detailed technical documentation:

- [Architecture](docs/architecture.md)
- [Dataset & DVC](docs/dataset.md)
- [Experiments / MLflow](docs/experiments.md)
- [Calibration](docs/calibration.md)
- [Serving](docs/serving.md)
- [Optimization](docs/optimization.md)
- [Load Testing](docs/load_testing.md)
- [Monitoring](docs/monitoring.md)
- [Drift Monitoring](docs/monitoring_drift.md)
- [Evidently & Prometheus](docs/monitoring_evidently_prometheus.md)
- [Grafana & Alerting](docs/grafana_alerting.md)
- [Airflow Retraining](docs/airflow_retraining.md)
- [Quality Gate & Promotion](docs/quality_gate_promotion.md)
- [Final End-to-End Evidence](docs/final_end_to_end_evidence.md)

---
# Project Journey

The project was intentionally developed as a progression through the MLOps lifecycle.
# ## Stage 1 — Foundation

Established the Python project structure, configuration, testing strategy, and reproducible development workflow.
# ## Stage 2 — Data & ML Pipeline

Added dataset versioning, deterministic data preparation, training, evaluation, and experiment tracking.
# ## Stage 3 — Production ML

Moved from model development toward production concerns: calibration, confidence handling, serving contracts, optimized runtimes, and inference performance.
# ## Stage 4 — Observability

Added monitoring, drift detection, metrics, dashboards, alerting, and evidence collection.
# ## Stage 5 — Automation & Integration

Connected drift detection to automated retraining, candidate evaluation, quality gates, MLflow registry promotion, and independent CI/CD checks.

The final result is a project that treats the model as one component of a larger **continuous ML system**.

---
# Evidence & Reproducibility Principles

A core rule throughout this project is:

> **Do not claim an experiment, metric, deployment state, alert, or promotion unless it can be reproduced or verified from actual evidence.**

For that reason:

- generated metrics are stored as artifacts;
- performance numbers are not invented;
- hardware-dependent optimization results require actual execution;
- MLflow registry state must be verified in the configured environment;
- monitoring screenshots must come from the real monitoring stack;
- final end-to-end claims are separated from individual component verification.

This makes the repository useful not only as a demonstration, but as an auditable MLOps implementation.

---
# Final Status

The project contains the complete implementation path for:

- [x] Data versioning with DVC
- [x] Reproducible data pipeline
- [x] Model training and evaluation
- [x] MLflow experiment tracking
- [x] Calibration and abstention
- [x] Model serving
- [x] Inference optimization
- [x] Triton-oriented production serving
- [x] Load testing
- [x] Drift detection
- [x] Prometheus/Grafana monitoring
- [x] Alerting and retraining webhook
- [x] Airflow retraining orchestration
- [x] Candidate evaluation
- [x] Automated quality gate
- [x] MLflow model promotion
- [x] Automated tests
- [x] Independent CI/CD workflows
- [x] Docker build pipeline
- [x] Final integration/evidence documentation

Runtime evidence that depends on a particular local or infrastructure environment is explicitly documented as a verification step rather than represented as completed without evidence.

---

---
