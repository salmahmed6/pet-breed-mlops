# MLflow Experiment Tracking Report

## Issue

**#23 — Add MLflow experiment tracking and model registry**

This report documents the MLflow tracking and model-registry implementation for the Track C Pet Breed Classification project.

## Experiment

- Experiment name: `pet-breed-classification`
- Tracking backend: SQLite (`sqlite:///mlflow.db`)
- Registered model: `PetBreedClassifier`
- Dataset: frozen Oxford-IIIT Pet manifest produced by the Sprint 1 DVC pipeline
- Training runs use the reproducible training pipeline from Issue #22.

## Required experiment coverage

Five MLflow experiment runs were completed across the three required backbones:

| Run | Backbone | Learning rate | Validation Top-1 | Validation Macro-F1 | Registered version |
|---|---|---:|---:|---:|---:|
| 1 | ResNet-18 | 0.001 | 0.1562 | 0.0729 | 2 |
| 2 | ResNet-50 | 0.001 | 0.2125 | 0.1336 | 3 |
| 3 | MobileNetV3-Small | 0.001 | 0.0875 | 0.0478 | 4 |
| 4 | ResNet-18 | 0.0005 | 0.1500 | 0.0521 | 5 |
| 5 | ResNet-50 | 0.0005 | 0.0688 | 0.0554 | 6 |

An additional initial integration run created model version 1.

> **Important:** These five runs were intentionally executed as limited-batch infrastructure-validation runs (1 epoch, 5 batches). Their metrics are not final model-quality results and must not be presented as the project's final benchmark.

## Logged parameters and metrics

Each tracked run records:

- backbone
- seed
- image size
- batch size
- epochs
- learning rate
- weight decay
- max-batches setting
- train loss
- train top-1 accuracy
- train macro-F1
- validation loss
- validation top-1 accuracy
- validation macro-F1

Calibration metrics such as ECE, temperature, coverage, and selective accuracy are intentionally deferred to Issue #24, where calibration and abstention are implemented.

## Artifacts

Each run stores:

- training history JSON
- model checkpoint
- registered PyTorch model

The registered model is named:

`PetBreedClassifier`

Model versions 1–6 were created during the verification and experiment runs.

## MLflow UI evidence

The MLflow UI was verified locally at `http://127.0.0.1:5000`.

### Evidence 1 — MLflow experiment

The `pet-breed-classification` experiment was verified in the MLflow UI. A screenshot of the experiment page was captured during local verification.

### Evidence 2 — Training runs

The Training Runs page was verified and showed the required five experiment runs across ResNet-18, ResNet-50, and MobileNetV3-Small. The registered model versions were visible in the Models column. A screenshot was captured during local verification.

### Evidence 3 — Model registry

The `PetBreedClassifier` registry was verified, including model version 6 and its source run. A screenshot was captured during local verification.

The original UI screenshots are retained as project evidence from the verification session. They are also included in the final report package prepared alongside this PR.

## Reproducibility

MLflow configuration is stored in:

`configs/mlflow.yaml`

The local tracking backend uses SQLite rather than MLflow's deprecated filesystem tracking backend.

A dedicated test verifies that MLflow parameters and metrics can be written and retrieved from the configured tracking backend.

## Verification

The single-run end-to-end verification completed successfully:

- MLflow experiment creation
- run creation
- parameter logging
- metric logging
- artifact logging
- PyTorch model logging
- `PetBreedClassifier` registration
- model version creation

The five-run experiment matrix then completed successfully.

## Scope boundary

Issue #23 covers experiment tracking and model registry. Probability calibration and abstention are handled separately by Issue #24, including:

- temperature scaling
- reliability diagrams
- ECE
- calibrated confidence
- abstention threshold
- coverage
- selective accuracy
