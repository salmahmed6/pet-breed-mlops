"""Evaluate a retrained candidate checkpoint and persist its report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from pet_breed_mlops.training.evaluate import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a retrained candidate checkpoint.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["resnet18", "mobilenet_v3_small", "resnet50"],
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint
    if checkpoint.name != f"{args.model}_best.pt":
        raise ValueError(
            f"Candidate checkpoint {checkpoint} does not match model {args.model}."
        )

    metrics = evaluate_checkpoint(
        checkpoint_path=checkpoint,
        manifest_path=config["data"]["manifest"],
        image_size=int(config["data"]["image_size"]),
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        split=args.split,
    )
    report = {
        "model": args.model,
        "split": args.split,
        "checkpoint_path": str(checkpoint),
        "top_1_accuracy": metrics["top_1_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "data_policy": {
            "fit_split": "train",
            "selection_split": "val",
            "protected_split": "test",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
