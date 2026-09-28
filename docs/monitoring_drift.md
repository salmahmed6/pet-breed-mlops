# Drift Detection and Scorecard

Sprint 4 Issue #43 implements three deterministic monitoring signals:
- Embedding drift: RBF-kernel MMD over penultimate visual features.
- Pixel drift: RBF-kernel MMD over brightness, contrast, and edge/sharpness statistics.
- Confidence drift: normalized mean shift of maximum softmax confidence.

The clean reference is the protected Oxford-IIIT Pet test split. Corrupted evaluation uses the existing deterministic corruption suite and generated metadata; no test images are modified.

## Run locally

Ensure DVC data and a trained checkpoint are available:
    dvc pull
    python -m scripts.run_drift_scorecard

Reports are written to reports/monitoring/drift_scorecard.json and reports/monitoring/drift_scorecard.csv.

## Configuration

configs/monitoring.yaml controls image size, samples per corruption group, and detector thresholds. Thresholds are configurable monitoring policy values, not claims about universal statistical significance.

## Reproducibility

Records are sorted by image ID, corruption type, and severity before deterministic truncation. The same checkpoint, data artifacts, configuration, and environment produce the same scorecard inputs and detector calculations.