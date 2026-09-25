"""Create deterministic train/validation/test splits for Oxford-IIIT Pet."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

from scripts.download_dataset import read_split


def load_config(path: Path) -> dict:
    """Load split configuration from YAML."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def create_splits(
    dataset_root: Path,
    output_path: Path,
    seed: int,
    validation_size: float,
) -> None:
    """Create deterministic train/validation/test image splits."""
    annotations = dataset_root / "annotations"

    trainval_names, _ = read_split(annotations / "trainval.txt")
    test_names, _ = read_split(annotations / "test.txt")

    if trainval_names & test_names:
        raise RuntimeError("Trainval and test splits overlap.")

    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    trainval = sorted(trainval_names)
    test = sorted(test_names)

    rng = random.Random(seed)
    rng.shuffle(trainval)

    validation_count = round(len(trainval) * validation_size)

    validation = sorted(trainval[:validation_count])
    train = sorted(trainval[validation_count:])

    if set(train) & set(validation):
        raise RuntimeError("Train and validation splits overlap.")

    if set(train) & set(test):
        raise RuntimeError("Train and test splits overlap.")

    if set(validation) & set(test):
        raise RuntimeError("Validation and test splits overlap.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "seed": seed,
        "validation_size": validation_size,
        "train": train,
        "validation": validation,
        "test": test,
    }

    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(output, file, sort_keys=False)

    print(f"Train: {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test: {len(test)}")
    print(f"Total: {len(train) + len(validation) + len(test)}")
    print(f"Seed: {seed}")
    print(f"Output: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create deterministic Oxford-IIIT Pet splits.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/oxford-iiit-pet"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/splits.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/splits.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    """Create the configured dataset splits."""
    args = parse_args()
    config = load_config(args.config)

    create_splits(
        dataset_root=args.dataset_root,
        output_path=args.output,
        seed=int(config["seed"]),
        validation_size=float(config["validation_size"]),
    )


if __name__ == "__main__":
    main()
