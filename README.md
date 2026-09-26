# Pet Breed Classification MLOps

An end-to-end MLOps system for 37-class pet breed classification using the Oxford-IIIT Pet dataset.

The project demonstrates:

- reproducible data versioning with DVC
- experiment tracking with MLflow
- model serving
- calibration and confidence-based abstention
- ONNX / OpenVINO / TensorRT optimization
- BentoML and Triton serving
- load testing
- drift detection
- Prometheus/Grafana monitoring
- automated retraining with Airflow
- CI/CD and model quality gates

## Quality Checks

The project uses automated testing, linting, formatting, and static type checking
to establish a deterministic quality baseline before data and model implementation.

### Run tests

```bash
python -m pytest
```

### Run linting

```bash
python -m ruff check .
```

### Check formatting

```bash
python -m ruff format --check .
```

To automatically format the project:

```bash
python -m ruff format .
```

### Run type checking

```bash
python -m mypy src
```

### Run all quality checks

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```

Each command returns a non-zero exit code when its check fails, so the same
commands can be used directly as CI quality gates.
