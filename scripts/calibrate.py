from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from pet_breed_mlops.calibration import (
    calibrate,
    result_to_json,
    save_reliability_diagram,
)
from pet_breed_mlops.data.loaders import create_dataloaders
from pet_breed_mlops.models.factory import create_model
from pet_breed_mlops.tracking.mlflow_tracker import MLflowTracker


def collect_validation_logits(
    checkpoint_path: Path,
    manifest_path: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model_name = str(checkpoint["model_name"])
    model = create_model(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, val_loader, _ = create_dataloaders(
        manifest_path=manifest_path,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in val_loader:
            logits_list.append(model(images))
            labels_list.append(labels)

    if not logits_list:
        raise ValueError("Validation loader produced no batches.")

    return torch.cat(logits_list), torch.cat(labels_list), model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit validation-only temperature scaling and abstention."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["resnet18", "resnet50", "mobilenet_v3_small"],
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
    )
    parser.add_argument(
        "--mlflow-config",
        type=Path,
        default=Path("configs/mlflow.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/calibration"),
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Skip creating a calibration MLflow run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    checkpoint_path = Path(config["output"]["directory"]) / f"{args.model}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train the selected model first."
        )

    logits, labels, model_name = collect_validation_logits(
        checkpoint_path,
        str(config["data"]["manifest"]),
        int(config["data"]["image_size"]),
        int(config["training"]["batch_size"]),
        int(config["data"]["num_workers"]),
    )

    result = calibrate(logits, labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_probabilities = torch.softmax(logits.float(), dim=1)
    calibrated_probabilities = torch.softmax(
        logits.float() / result.temperature,
        dim=1,
    )

    raw_plot = args.output_dir / f"{model_name}_reliability_raw.png"
    calibrated_plot = args.output_dir / f"{model_name}_reliability_calibrated.png"
    save_reliability_diagram(
        raw_probabilities,
        labels,
        raw_plot,
        title=f"{model_name} — raw reliability",
    )
    save_reliability_diagram(
        calibrated_probabilities,
        labels,
        calibrated_plot,
        title=f"{model_name} — calibrated reliability",
    )

    report = {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "calibration_split": "val",
        "test_split_used": False,
        **result_to_json(result),
    }
    report_path = args.output_dir / f"{model_name}_calibration.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not args.no_mlflow:
        tracker = MLflowTracker.from_config(args.mlflow_config)
        tracker.start(run_name=f"{model_name}-calibration")
        tracker.log_params(
            {
                "backbone": model_name,
                "calibration_method": "temperature_scaling",
                "calibration_split": "val",
                "abstention_method": "max_calibrated_confidence_threshold",
                "abstention_threshold": result.abstention_threshold,
            }
        )
        tracker.log_metrics(
            {
                "temperature": result.temperature,
                "raw_ece": result.raw_ece,
                "calibrated_ece": result.calibrated_ece,
                "coverage": result.coverage,
                "selective_accuracy": result.selective_accuracy,
            }
        )
        tracker.log_artifact(report_path, artifact_path="calibration")
        tracker.log_artifact(raw_plot, artifact_path="calibration")
        tracker.log_artifact(calibrated_plot, artifact_path="calibration")
        run_id = tracker.run_id
        tracker.finish()
        print(f"MLflow calibration run: {run_id}")

    print(f"Model: {model_name}")
    print(f"Temperature: {result.temperature:.6f}")
    print(f"Raw ECE: {result.raw_ece:.6f}")
    print(f"Calibrated ECE: {result.calibrated_ece:.6f}")
    print(f"Abstention threshold: {result.abstention_threshold:.6f}")
    print(f"Coverage: {result.coverage:.6f}")
    print(f"Selective accuracy: {result.selective_accuracy:.6f}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
