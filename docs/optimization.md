# Inference Optimization and Benchmarking

This issue provides reproducible optimization paths without fabricating hardware-dependent results.

## Benchmark

Run the baseline first:

```bash
python -m scripts.benchmark_optimization --checkpoint artifacts/models/resnet18_best.pt --method baseline --output reports/optimization/resnet18_baseline.json
```

Then structured pruning:

```bash
python -m scripts.benchmark_optimization --checkpoint artifacts/models/resnet18_best.pt --method pruned --prune-amount 0.20 --output reports/optimization/resnet18_pruned.json
```

The benchmark records validation top-1, parameter count, pruning sparsity, and p50/p95 latency plus throughput for batch sizes 1, 8, 16, and 32. Measurements are generated on the machine where the command runs.

## INT8 PTQ

Export the model to ONNX, then run static ONNX Runtime quantization using validation images as calibration data:

```bash
python -m scripts.quantize_onnx_int8 --onnx artifacts/models/resnet18.onnx --output artifacts/models/resnet18_int8.onnx
```

Static quantization derives activation ranges from calibration data rather than inventing them. The test split is not used.

## QAT

A reproducible QAT experiment is provided for CPU execution:

```bash
python -m scripts.run_qat --checkpoint artifacts/models/resnet18_best.pt --output artifacts/models/resnet18_qat.pt
```

The script uses PyTorch's QAT flow and is intentionally a small experiment by default; use larger epoch/batch settings for a final benchmark.

## Knowledge distillation

Use ResNet-50 as teacher and ResNet-18 as student:

```bash
python -m scripts.distill --teacher-checkpoint artifacts/models/resnet50_best.pt --student-checkpoint artifacts/models/resnet18_best.pt --output artifacts/models/resnet18_distilled.pt
```

The loss combines hard labels with temperature-scaled teacher probabilities.

## TensorRT FP16

```bash
python -m scripts.tensorrt_fp16 --onnx artifacts/models/resnet18.onnx --output artifacts/models/resnet18_fp16.engine
```

If `trtexec` is unavailable, the script records `status: skipped` instead of fabricating a result.

## MLflow

The benchmark output is designed to be logged from the same machine where the measurement was produced. Hardware and runtime details are included in the report so results remain interpretable.

No final accuracy, latency, throughput, or TensorRT numbers are committed until they are actually measured.
