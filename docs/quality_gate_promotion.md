# Sprint 5 Issue #47 — Retraining Quality Gate and Model Promotion

## Purpose

Issue #47 adds the decision layer after candidate retraining:

candidate evaluation → production baseline → fixed quality gate → MLflow decision → promotion or rejection

The workflow does not fabricate a candidate result. A real candidate evaluation artifact is required before the promotion path can run.

## Quality rule

The Sprint 5 handbook threshold is enforced exactly:

`candidate top-1 accuracy >= production top-1 accuracy - 0.01`

Therefore:

- exactly -0.01 is accepted;
- anything below -0.01 is rejected;
- the threshold cannot be overridden through the CLI.

The current committed production/evaluation baseline is read from `reports/quality_baseline.json` when a usable MLflow `Production` alias is not available.

## MLflow behavior

The quality gate uses the MLflow configuration in `configs/mlflow.yaml`.

For every decision it records:

- production top-1 accuracy;
- candidate top-1 accuracy;
- required candidate accuracy;
- candidate delta;
- threshold;
- baseline source;
- candidate and baseline artifacts;
- final decision.

For an accepted candidate:

1. the candidate checkpoint is loaded;
2. the model is logged to MLflow;
3. the model is registered as a new version;
4. the version receives quality-gate and baseline tags;
5. the `Production` alias is moved to the new version.

For a rejected candidate:

- the candidate is not promoted;
- the existing Production alias is left unchanged;
- the rejection decision is persisted and logged.

## CLI

Validate only the quality rule without MLflow or promotion:

    python -m scripts.quality_gate --baseline reports/quality_baseline.json --candidate reports/retraining/candidate_evaluation.json --validate-only

Run the complete quality gate and promotion workflow:

    python -m scripts.quality_gate --baseline reports/quality_baseline.json --candidate reports/retraining/candidate_evaluation.json --decision reports/retraining/quality_gate_decision.json --mlflow-config configs/mlflow.yaml

## Airflow integration

The Issue #46 DAG now ends with:

`check_drift → retrain_candidate → evaluate_candidate → quality_gate`

The final task writes `reports/retraining/quality_gate_decision.json` and uses the existing MLflow configuration for promotion.

## Testing

Boundary coverage includes:

- candidate equal to Production;
- candidate exactly 0.01 below Production;
- candidate below the -0.01 boundary;
- larger quality drops;
- fixed-threshold enforcement;
- baseline artifact loading;
- invalid metric values.

A complete real promotion run should only be claimed when MLflow has actually registered the candidate and moved the Production alias. No such result is fabricated by this implementation.

## Rollback

A rejected candidate never changes the `Production` alias, so rejection is a no-op for the live model.

If a promoted version later needs to be rolled back, the MLflow `Production` alias can be moved back to the previously known-good model version. The quality-gate decision and model-version tags provide the audit trail needed to identify the promoted version and its production baseline.

No rollback result is claimed until the alias change has actually been performed in MLflow.
