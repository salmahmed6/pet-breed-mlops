from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch import nn, optim

from pet_breed_mlops.data.loaders import create_dataloaders
from pet_breed_mlops.models.factory import create_model


def main():
    parser = argparse.ArgumentParser(description="Run a reproducible CPU QAT experiment.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=10)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = create_model(payload["model_name"])
    model.load_state_dict(payload["model_state_dict"])
    model.train()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")
    torch.ao.quantization.prepare_qat(model, inplace=True)

    train_loader, _, _ = create_dataloaders(
        config["data"]["manifest"],
        int(config["data"]["image_size"]),
        int(config["training"]["batch_size"]),
        int(config["data"]["num_workers"]),
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
    )
    criterion = nn.CrossEntropyLoss()

    for _ in range(args.epochs):
        for batch_index, (images, labels) in enumerate(train_loader):
            if batch_index >= args.max_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    model.eval()
    quantized = torch.ao.quantization.convert(model, inplace=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": payload["model_name"],
            "model_state_dict": quantized.state_dict(),
        },
        args.output,
    )
    print(f"QAT checkpoint: {args.output}")


if __name__ == "__main__":
    main()
