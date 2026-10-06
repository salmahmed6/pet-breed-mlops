from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import torch
import yaml
from mlflow import MlflowClient

from pet_breed_mlops.models.factory import create_model

QUALITY_THRESHOLD = -0.01
PRODUCTION_ALIAS = "Production"


def load_top_1(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"Quality gate report not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("top_1_accuracy")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{path} must contain numeric 'top_1_accuracy'")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{path} top_1_accuracy must be between 0 and 1")
    return value


def load_production_top_1(
    baseline_path: Path,
    *,
    tracking_uri: str | None = None,
    registered_model_name: str | None = None,
    alias: str = PRODUCTION_ALIAS,
) -> tuple[float, str]:
    """Read Production accuracy from MLflow when available, otherwise the evaluation artifact."""
    if tracking_uri and registered_model_name:
        try:
            client = MlflowClient(tracking_uri=tracking_uri)
            version = client.get_model_version_by_alias(registered_model_name, alias)
            value = version.tags.get("top_1_accuracy")
            if value is not None:
                accuracy = float(value)
                if 0.0 <= accuracy <= 1.0:
                    return accuracy, f"mlflow:{registered_model_name}@{alias}"
        except Exception:
            pass

    return load_top_1(baseline_path), f"artifact:{baseline_path}"


def compare_quality(
    production_top_1: float,
    candidate_top_1: float,
    minimum_delta: float = QUALITY_THRESHOLD,
) -> dict[str, Any]:
    """Apply the fixed -0.01 quality threshold exactly."""
    if minimum_delta != QUALITY_THRESHOLD:
        raise ValueError(
            f"The Sprint 5 quality gate requires minimum_delta={QUALITY_THRESHOLD:.2f}."
        )

    required_top_1 = production_top_1 + QUALITY_THRESHOLD
    accepted = candidate_top_1 >= required_top_1
    return {
        "accepted": accepted,
        "production_top_1_accuracy": production_top_1,
        "candidate_top_1_accuracy": candidate_top_1,
        "required_candidate_top_1_accuracy": required_top_1,
        "minimum_delta": QUALITY_THRESHOLD,
        "delta": candidate_top_1 - production_top_1,
    }


def _load_candidate_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = create_model(checkpoint["model_name"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _find_registered_version(client: MlflowClient, model_name: str, run_id: str):
    versions = client.search_model_versions(f"name='{model_name}'")
    matches = [version for version in versions if version.run_id == run_id]
    if not matches:
        raise RuntimeError(
            f"MLflow did not return a registered version for run {run_id} and model {model_name}."
        )
    return max(matches, key=lambda version: int(version.version))


def promote_registered_version(
    client: MlflowClient,
    *,
    registered_model_name: str,
    version: str,
    production_top_1: float,
    candidate_top_1: float,
) -> None:
    """Tag an accepted version and move the Production alias to it."""
    client.set_model_version_tag(
        registered_model_name,
        version,
        "top_1_accuracy",
        f"{candidate_top_1:.8f}",
    )
    client.set_model_version_tag(
        registered_model_name,
        version,
        "quality_gate_decision",
        "accepted",
    )
    client.set_model_version_tag(
        registered_model_name,
        version,
        "production_baseline_top_1_accuracy",
        f"{production_top_1:.8f}",
    )
    client.set_registered_model_alias(
        registered_model_name,
        PRODUCTION_ALIAS,
        version,
    )


def run_quality_gate(
    *,
    baseline_path: Path,
    candidate_path: Path,
    decision_path: Path,
    mlflow_config_path: Path,
) -> dict[str, Any]:
    """Compare, record the decision in MLflow, and promote an accepted candidate."""
    config = yaml.safe_load(mlflow_config_path.read_text(encoding="utf-8"))
    tracking_uri = str(config["tracking_uri"])
    experiment_name = str(config["experiment_name"])
    registered_model_name = str(config["registered_model_name"])

    production_top_1, baseline_source = load_production_top_1(
        baseline_path,
        tracking_uri=tracking_uri,
        registered_model_name=registered_model_name,
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_top_1 = candidate.get("top_1_accuracy")
    if not isinstance(candidate_top_1, (int, float)):
        raise ValueError(f"{candidate_path} must contain numeric 'top_1_accuracy'")
    candidate_top_1 = float(candidate_top_1)
    if not 0.0 <= candidate_top_1 <= 1.0:
        raise ValueError(f"{candidate_path} top_1_accuracy must be between 0 and 1")

    decision = compare_quality(production_top_1, candidate_top_1)
    decision.update(
        {
            "model": candidate.get("model"),
            "candidate_report": str(candidate_path),
            "checkpoint_path": candidate.get("checkpoint_path"),
            "baseline_source": baseline_source,
            "registered_model_name": registered_model_name,
            "production_alias": PRODUCTION_ALIAS,
        }
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    client = MlflowClient(tracking_uri=tracking_uri)
    with mlflow.start_run(run_name=f"quality-gate-{candidate.get('model', 'candidate')}") as run:
        mlflow.log_params(
            {
                "registered_model_name": registered_model_name,
                "production_alias": PRODUCTION_ALIAS,
                "quality_threshold": QUALITY_THRESHOLD,
                "baseline_source": baseline_source,
                "candidate_report": str(candidate_path),
            }
        )
        mlflow.log_metrics(
            {
                "production_top_1_accuracy": production_top_1,
                "candidate_top_1_accuracy": candidate_top_1,
                "required_candidate_top_1_accuracy": decision["required_candidate_top_1_accuracy"],
                "candidate_delta": decision["delta"],
            }
        )
        mlflow.log_artifact(str(baseline_path), artifact_path="quality-gate")
        mlflow.log_artifact(str(candidate_path), artifact_path="quality-gate")

        if decision["accepted"]:
            checkpoint_value = decision.get("checkpoint_path")
            if not checkpoint_value:
                raise ValueError("Accepted candidate report must contain 'checkpoint_path'.")
            checkpoint_path = Path(checkpoint_value)
            model, checkpoint = _load_candidate_model(checkpoint_path)
            mlflow.pytorch.log_model(
                model,
                name="model",
                input_example=torch.zeros(
                    1,
                    3,
                    int(checkpoint.get("image_size", 224)),
                    int(checkpoint.get("image_size", 224)),
                ),
                registered_model_name=registered_model_name,
                serialization_format="pickle",
            )
            version = _find_registered_version(client, registered_model_name, run.info.run_id)
            promote_registered_version(
                client,
                registered_model_name=registered_model_name,
                version=version.version,
                production_top_1=production_top_1,
                candidate_top_1=candidate_top_1,
            )
            decision["promoted_version"] = int(version.version)
            decision["mlflow_run_id"] = run.info.run_id
        else:
            decision["rejected_version"] = None
            decision["mlflow_run_id"] = run.info.run_id

        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(decision_path), artifact_path="quality-gate")

    print(json.dumps(decision, indent=2))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the retraining quality gate and promotion workflow."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--mlflow-config", type=Path, default=Path("configs/mlflow.yaml"))
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the fixed quality rule without starting an MLflow run or promoting a model.",
    )
    args = parser.parse_args()

    if args.validate_only:
        production_top_1 = load_top_1(args.baseline)
        candidate_top_1 = load_top_1(args.candidate)
        decision = compare_quality(production_top_1, candidate_top_1)
        print(json.dumps(decision, indent=2))
        if not decision["accepted"]:
            raise SystemExit(
                "Model quality gate rejected candidate: "
                f"{candidate_top_1:.6f} < "
                f"{decision['required_candidate_top_1_accuracy']:.6f}."
            )
        return

    if args.decision is None:
        parser.error("--decision is required unless --validate-only is used.")

    decision = run_quality_gate(
        baseline_path=args.baseline,
        candidate_path=args.candidate,
        decision_path=args.decision,
        mlflow_config_path=args.mlflow_config,
    )
    if not decision["accepted"]:
        raise SystemExit(
            "Model quality gate rejected candidate: "
            f"{decision['candidate_top_1_accuracy']:.6f} < "
            f"{decision['required_candidate_top_1_accuracy']:.6f}."
        )


if __name__ == "__main__":
    main()
