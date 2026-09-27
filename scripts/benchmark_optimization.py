from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import yaml

from pet_breed_mlops.data.loaders import create_dataloaders
from pet_breed_mlops.models.factory import create_model
from pet_breed_mlops.optimization import batch_sizes_from_string, count_parameters, structured_prune_model


def load_model(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = create_model(payload["model_name"])
    model.load_state_dict(payload["model_state_dict"])
    return model.to(device).eval(), payload["model_name"]


def benchmark(model, batch_size: int, image_size: int, warmup: int, iterations: int, device):
    sample = torch.randn(batch_size, 3, image_size, image_size, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    p50 = statistics.median(timings)
    p95 = sorted(timings)[max(0, int(len(timings) * 0.95) - 1)]
    return {
        "batch_size": batch_size,
        "latency_ms_p50": p50,
        "latency_ms_p95": p95,
        "throughput_images_per_second": batch_size / (p50 / 1000),
    }


def validation_accuracy(model, loader, device):
    correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            correct += int((logits.argmax(1).cpu() == labels).sum())
            total += labels.numel()
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--method", choices=["baseline", "pruned"], default="baseline")
    parser.add_argument("--prune-amount", type=float, default=0.2)
    parser.add_argument("--batch-sizes", default="1,8,16,32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("reports/optimization/benchmark.json"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name = load_model(args.checkpoint, device)
    sparsity = 0.0
    if args.method == "pruned":
        model, sparsity = structured_prune_model(model, args.prune_amount)
        model = model.to(device).eval()

    _, val_loader, _ = create_dataloaders(
        manifest_path=config["data"]["manifest"],
        image_size=int(config["data"]["image_size"]),
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
    )
    result = {
        "model": model_name,
        "method": args.method,
        "device": str(device),
        "parameter_count": count_parameters(model),
        "structured_pruning_sparsity": sparsity,
        "val_top_1_accuracy": validation_accuracy(model, val_loader, device),
        "batch_size_matrix": [
            benchmark(model, size, int(config["data"]["image_size"]), args.warmup, args.iterations, device)
            for size in batch_sizes_from_string(args.batch_sizes)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
