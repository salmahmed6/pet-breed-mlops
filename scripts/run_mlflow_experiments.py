from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pet_breed_mlops.training.trainer import train_model

CONFIG = Path("configs/training.yaml")
MLFLOW_CONFIG = Path("configs/mlflow.yaml")

RUNS = [
    ("resnet18", 0.001),
    ("resnet50", 0.001),
    ("mobilenet_v3_small", 0.001),
    ("resnet18", 0.0005),
    ("resnet50", 0.0005),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=5)
    args = parser.parse_args()

    with CONFIG.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    for index, (model_name, learning_rate) in enumerate(RUNS, start=1):
        print(f"\n=== MLflow run {index}/{len(RUNS)}: {model_name} lr={learning_rate} ===")
        result = train_model(
            model_name=model_name,
            manifest_path=config["data"]["manifest"],
            image_size=int(config["data"]["image_size"]),
            batch_size=int(config["training"]["batch_size"]),
            num_workers=int(config["data"]["num_workers"]),
            epochs=args.epochs,
            learning_rate=learning_rate,
            weight_decay=float(config["training"]["weight_decay"]),
            seed=int(config["seed"]) + index,
            output_dir=Path(config["output"]["directory"]),
            max_batches=args.max_batches,
            mlflow_config_path=MLFLOW_CONFIG,
        )
        print(
            f"run={result['mlflow_run_id']} "
            f"val_top1={result['best_val_top_1_accuracy']:.4f} "
            f"val_f1={result['best_val_macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
