# Serving and Inference Contract

## Contract

The serving layer exposes a single image prediction operation and returns the top-3 predictions. Each prediction contains:

- `breed`
- `species`
- `class_index`
- `confidence` (raw softmax confidence)
- `calibrated_confidence`
- `abstained`
- `model_version`

The service uses the same RGB conversion, resize-to-224, and ImageNet normalization used by validation/test inference.

## BentoML

After installing the project dependencies and producing a checkpoint:

```bash
bentoml serve src/pet_breed_mlops/bento_service.py:PetBreedClassifier
```

The default checkpoint is `artifacts/models/resnet18_best.pt`. Override it with `PET_BREED_CHECKPOINT`.

Calibration settings can be supplied without changing code:

```bash
PET_BREED_TEMPERATURE=1.03
PET_BREED_ABSTENTION_THRESHOLD=0.90
PET_BREED_MODEL_VERSION=resnet18-v1
```

## ONNX Runtime

Export and verify:

```bash
python -m scripts.export_onnx --checkpoint artifacts/models/resnet18_best.pt --output artifacts/models/resnet18.onnx
python -m scripts.verify_onnx --checkpoint artifacts/models/resnet18_best.pt --onnx artifacts/models/resnet18.onnx
```

The verification command fails if the maximum absolute PyTorch/ONNX logit difference exceeds `1e-4`.

## Triton

Place the exported model at:

```text
model_repository/pet-breed-classifier/1/model.onnx
```

and copy `configs/triton/config.pbtxt` into that model directory as `config.pbtxt`.

The provided configuration is CPU-based and batchable, with preferred batch sizes 1, 8, 16, and 32. TensorRT FP16 is intentionally left for the optimization issue, where the deployment backend and hardware can be selected based on the available environment.

## Scope note

OpenVINO and TensorRT execution/benchmarking are part of the optimization/deployment work in Issue #33. This issue establishes the common inference contract, BentoML service, ONNX export/agreement check, and Triton model configuration without fabricating hardware-dependent results.
