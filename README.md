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

## CI/CD

GitHub Actions is defined in `.github/workflows/ci-cd.yml`.

For pull requests, CI runs:

1. Python installation and project installation
2. Ruff linting
3. Ruff formatting check
4. Pytest
5. the model-quality gate
6. Docker image build

A push to `main` runs the same quality/build path and then publishes the
Docker image to GitHub Container Registry (GHCR).

### Model quality gate

The quality gate compares a candidate evaluation report containing
`top_1_accuracy` against the committed baseline in
`reports/quality_baseline.json`.

The configured rule is in `configs/quality_gate.yaml`:

```yaml
metric: top_1_accuracy
baseline_report: reports/quality_baseline.json
minimum_delta: 0.0
```

Run it locally with:

```bash
python -m scripts.quality_gate \
  --baseline reports/quality_baseline.json \
  --candidate reports/quality_baseline.json
```

The current committed baseline is explicitly labeled as the Sprint 2 smoke-test
baseline. It is not a final model benchmark. A later training/evaluation stage
must replace the baseline with a real committed evaluation report before using
it as the production benchmark.

The gate fails when the candidate top-1 accuracy is below the configured
baseline.

### Container publishing

The workflow publishes to:

```
ghcr.io/<github-owner>/pet-breed-mlops:latest
```

No registry password is committed. GHCR publishing uses the workflow-provided
`GITHUB_TOKEN` with `packages: write` permission.

For a different registry, replace the login and tag configuration with the
appropriate registry secret/token stored in GitHub Actions Secrets.

## Evidence

A successful CI workflow run should be captured from the repository's
**Actions** tab as evidence for Issue #25.
