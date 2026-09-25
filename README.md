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

The project uses automated testing, linting, formatting, and static type
checking to establish a deterministic quality baseline.

### Run tests

```bash
python -m pytest