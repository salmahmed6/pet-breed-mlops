# Pet Breed Classification MLOps

End-to-end MLOps platform for Oxford-IIIT Pet breed classification.

## Inference optimization

Optimization paths and reproducible benchmark commands are documented in [docs/optimization.md](docs/optimization.md).

The project evaluates structured pruning, INT8 post-training quantization, QAT, knowledge distillation, and TensorRT FP16 where the local hardware supports them. Benchmark outputs belong under `reports/optimization/`.

Results must be generated from the actual execution environment; unsupported GPU/TensorRT measurements are recorded as skipped rather than fabricated.

## Quality Checks

```bash
ruff check .
ruff format --check .
python -m pytest -q
```
