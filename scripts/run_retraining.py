"""Run candidate retraining and persist an evaluation artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from pet_breed_mlops.training.trainer import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a drift-triggered candidate training job.")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/retraining/candidate"))
    parser.add_argument("--report", type=Path, default=Path("reports/retraining/candidate_evaluation.json"))
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    result = train_model(
        model_name=args.model,
        manifest_path=config["data"]["manifest"],
        image_size=int(config["data"]["image_size"]),
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        epochs=int(config["training"]["epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        seed=int(config["seed"]),
        output_dir=args.output_dir,
        max_batches=args.max_batches,
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": result["model_name"],
        "split": "val",
        "top_1_accuracy": result["best_val_top_1_accuracy"],
        "macro_f1": result["best_val_macro_f1"],
        "checkpoint_path": result["checkpoint_path"],
        "history_path": result["history_path"],
        "device": result["device"],
        "data_policy": {
            "fit_split": "train",
            "selection_split": "val",
            "protected_split": "test",
        },
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
