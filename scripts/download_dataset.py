"""Download and validate the Oxford-IIIT Pet dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import OxfordIIITPet

EXPECTED_IMAGES = 7_349
EXPECTED_CLASSES = 37

SPLITS = ("trainval", "test")


def download_dataset(root: Path) -> None:
    """Download the Oxford-IIIT Pet dataset if it is not already present."""
    root.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        print(f"Preparing Oxford-IIIT Pet split: {split}")

        dataset = OxfordIIITPet(
            root=str(root),
            split=split,
            target_types="category",
            download=True,
        )

        print(f"  {split}: {len(dataset)} samples")


def validate_dataset(root: Path) -> None:
    """Validate the expected dataset size and class count."""
    datasets = [
        OxfordIIITPet(
            root=str(root),
            split=split,
            target_types="category",
            download=False,
        )
        for split in SPLITS
    ]

    total_images = sum(len(dataset) for dataset in datasets)

    if total_images != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} images, found {total_images}."
        )

    labels = set()

    for dataset in datasets:
        labels.update(dataset._labels)

    if len(labels) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_CLASSES} classes, found {len(labels)}."
        )

    print(f"Validation passed: {total_images} images, {len(labels)} classes.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and validate Oxford-IIIT Pet."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw"),
        help="Dataset root directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Run dataset download and validation."""
    args = parse_args()

    download_dataset(args.root)
    validate_dataset(args.root)


if __name__ == "__main__":
    main()