# Track C — Pet Breed Classification MLOps Architecture

## 1. Project Overview

This project implements an end-to-end MLOps platform for **pet breed classification** using the Oxford-IIIT Pet dataset.

The system takes a pet image as input and produces:

- Top-3 breed predictions
- Calibrated confidence
- Abstention decision
- Predicted species (cat/dog)
- Model version

The project focuses on taking a computer vision model from reproducible data preparation and training through optimization, production serving, monitoring, and automated retraining.

---

## 2. High-Level Architecture

```mermaid
flowchart TD

    A[Oxford-IIIT Pet Dataset] --> B[DVC Data Versioning]

    B --> C[Fixed Train / Validation / Test Splits]
    C --> D[Dataset Manifest]
    D --> E[Dataset Validation]

    E --> F[Corruption Suite]
    F --> G[Corrupted Test Sets]

    C --> H[PyTorch Dataset / DataLoader]

    H --> I[Model Training]

    I --> I1[ResNet]
    I --> I2[MobileNet]
    I --> I3[EfficientNet]

    I1 --> J[MLflow Experiments]
    I2 --> J
    I3 --> J

    J --> K[Model Registry]
    K --> L[Calibration]

    L --> M[Production Model]

    M --> N[BentoML API]

    N --> O[ONNX Runtime]
    N --> P[OpenVINO]
    N --> Q[Triton / TensorRT]

    M --> R[Model Optimization]

    R --> R1[Structured Pruning]
    R --> R2[INT8 PTQ]
    R --> R3[QAT]
    R --> R4[Knowledge Distillation]
    R --> R5[TensorRT FP16]

    R1 --> S[Benchmarking]
    R2 --> S
    R3 --> S
    R4 --> S
    R5 --> S

    N --> T[Production Predictions]

    T --> U[Monitoring]

    U --> U1[Embedding Drift]
    U --> U2[Pixel Statistics Drift]
    U --> U3[Confidence Drift]

    U1 --> V[Prometheus]
    U2 --> V
    U3 --> V

    V --> W[Grafana]
    W --> X[Drift Alert]

    X --> Y[Airflow Retraining DAG]

    Y --> Z[Fine-tune on Drifted / Corrupted Distribution]

    Z --> AA[Evaluation + Quality Gate]

    AA -->|Pass| AB[Promote Model]
    AA -->|Fail| AC[Reject Model]

    AB --> K