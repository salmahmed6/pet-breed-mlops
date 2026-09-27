from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from pet_breed_mlops.data.loaders import create_dataloaders
from pet_breed_mlops.models.factory import create_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("reports/serving/batch_inference.json"))
    a = p.parse_args()
    if a.batch_size < 1 or a.max_batches < 0:
        raise ValueError("invalid batch settings")
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(a.checkpoint, map_location=device, weights_only=False)
    model = create_model(payload["model_name"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    _, loader, _ = create_dataloaders(
        cfg["data"]["manifest"],
        int(cfg["data"]["image_size"]),
        a.batch_size,
        int(cfg["data"]["num_workers"]),
    )
    processed = correct = batches = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for images, labels in loader:
            pred = model(images.to(device)).argmax(1).cpu()
            correct += int((pred == labels).sum())
            processed += labels.numel()
            batches += 1
            if a.max_batches and batches >= a.max_batches:
                break
    elapsed = time.perf_counter() - start
    result = {
        "checkpoint": str(a.checkpoint),
        "device": str(device),
        "batch_size": a.batch_size,
        "batches": batches,
        "images": processed,
        "elapsed_seconds": elapsed,
        "images_per_second": processed / elapsed if elapsed else 0.0,
        "top_1_accuracy": correct / processed if processed else 0.0,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
