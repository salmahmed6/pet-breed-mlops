from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from pet_breed_mlops.data.loaders import create_dataloaders
from pet_breed_mlops.models.factory import create_model
from pet_breed_mlops.training.metrics import calculate_metrics, confusion_matrix


def evaluate_checkpoint(checkpoint_path, manifest_path, image_size, batch_size, num_workers, split="val"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = create_model(checkpoint["model_name"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_loader, val_loader, test_loader = create_dataloaders(
        manifest_path=str(manifest_path), image_size=image_size,
        batch_size=batch_size, num_workers=num_workers,
    )
    loader = val_loader if split == "val" else test_loader
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=1).tolist())

    metrics = calculate_metrics(y_true, y_pred)
    return {**metrics, "split": split,
            "confusion_matrix": confusion_matrix(y_true, y_pred, 37).tolist()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained pet breed classifier.")
    parser.add_argument("--model", required=True,
                        choices=["resnet18", "resnet50", "mobilenet_v3_small"])
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    checkpoint = Path(config["output"]["directory"]) / f"{args.model}_best.pt"
    result = evaluate_checkpoint(
        checkpoint, config["data"]["manifest"], int(config["data"]["image_size"]),
        int(config["training"]["batch_size"]), int(config["data"]["num_workers"]), args.split,
    )
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / f"{args.model}_{args.split}_evaluation.json"
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")
    print(f"Top-1 accuracy: {result['top_1_accuracy']:.4f}")
    print(f"Macro-F1: {result['macro_f1']:.4f}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
