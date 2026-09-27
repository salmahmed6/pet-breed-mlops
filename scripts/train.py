from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pet_breed_mlops.training.trainer import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pet breed classifier.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["resnet18", "resnet50", "mobilenet_v3_small"],
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    epochs = args.epochs if args.epochs is not None else int(config["training"]["epochs"])

    print(f"Training model: {args.model}", flush=True)
    print(f"Epochs: {epochs}", flush=True)
    if args.max_batches is not None:
        print(f"Max batches per epoch: {args.max_batches}", flush=True)

    result = train_model(
        model_name=args.model,
        manifest_path=config["data"]["manifest"],
        image_size=int(config["data"]["image_size"]),
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
        epochs=epochs,
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        seed=int(config["seed"]),
        output_dir=Path(config["output"]["directory"]),
        max_batches=args.max_batches,
    )

    print(
        f"Best validation top-1: {result['best_val_top_1_accuracy']:.4f}",
        flush=True,
    )
    print(
        f"Best validation macro-F1: {result['best_val_macro_f1']:.4f}",
        flush=True,
    )
    print(f"Checkpoint: {result['checkpoint_path']}", flush=True)


if __name__ == "__main__":
    main()
