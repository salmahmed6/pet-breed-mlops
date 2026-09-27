# Calibration and Abstention

Issue #24 implements validation-only probability calibration for the Pet Breed classifier.

## Method

1. Load a trained checkpoint.
2. Collect logits from the validation split only.
3. Fit one positive temperature with LBFGS by minimizing validation cross-entropy.
4. Compute raw and calibrated probabilities.
5. Compute Expected Calibration Error (ECE).
6. Generate raw and calibrated reliability diagrams.
7. Select an abstention threshold deterministically from calibrated validation confidences.
8. Report coverage and selective accuracy.
9. Log temperature and calibration metrics to MLflow in a dedicated calibration run.

The protected test split is not used by the calibration workflow.

## Abstention rule

The eventual prediction API should compute:

- calibrated_confidence = max(softmax(logits / temperature))
- abstained = calibrated_confidence < abstention_threshold

When abstained is true, the API can return the top candidate for observability while marking the decision as uncertain.

The response contract is represented by PetBreedPrediction in src/pet_breed_mlops/predict_contract.py.

## Local command

After a trained checkpoint exists:

    python -m scripts.calibrate --model resnet18

Use --no-mlflow only when MLflow tracking is intentionally unavailable.

Outputs are written to reports/calibration/:

- <model>_calibration.json
- <model>_reliability_raw.png
- <model>_reliability_calibrated.png
