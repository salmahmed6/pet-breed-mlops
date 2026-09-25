"""Download and validate the Oxford-IIIT Pet dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import OxfordIIITPet

EXPECTED_IMAGES = 7_349
EXPECTED_CLASSES = 37
EXPECTED_TRAINVAL = 3_680
EXPECTED_TEST = 3_669

SPLITS = ("trainval", "test")
DATASET_NAME = "Oxford-IIIT Pet"
DATASET_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/"


def read_split(annotation_file: Path) -> tuple[set[str], set[int]]:
    """Read image IDs and 1-based class labels from an Oxford-IIIT Pet split."""
    image_names: set[str] = set()
    labels: set[int] = set()

    for line in annotation_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        image_id, label, *_ = line.split()
        image_names.add(f"{image_id}.jpg")
        labels.add(int(label))

    return image_names, labels


def download_dataset(root: Path) -> None:
    """Download the official dataset using torchvision's pinned resources."""
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
    """Validate the official Oxford-IIIT Pet image and annotation lists."""
    annotations = root / "annotations"
    images = root / "images"

    if not annotations.is_dir():
        raise RuntimeError(f"Missing annotations directory: {annotations}")

    if not images.is_dir():
        raise RuntimeError(f"Missing images directory: {images}")

    trainval, trainval_labels = read_split(annotations / "trainval.txt")
    test, test_labels = read_split(annotations / "test.txt")

    if len(trainval) != EXPECTED_TRAINVAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAINVAL} trainval images, found {len(trainval)}."
        )

    if len(test) != EXPECTED_TEST:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST} test images, found {len(test)}."
        )

    if trainval & test:
        raise RuntimeError("Trainval and test splits overlap.")

    labels = trainval_labels | test_labels
    if len(labels) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_CLASSES} classes, found {len(labels)}."
        )

    expected = trainval | test
    actual = {path.name for path in images.glob("*.jpg")}

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    print(f"Dataset: {DATASET_NAME}")
    print(f"Source: {DATASET_URL}")
    print(f"Trainval images: {len(trainval)}")
    print(f"Test images: {len(test)}")
    print(f"Expected images: {len(expected)}")
    print(f"Actual images: {len(actual)}")
    print(f"Classes: {len(labels)}")
    print(f"Missing images: {len(missing)}")
    print(f"Extra images: {len(extra)}")

    if len(expected) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} images from official splits, "
            f"found {len(expected)}."
        )

    if missing:
        print("Missing images:")
        for name in missing:
            print(f"  - {name}")

    if extra:
        print("Extra images:")
        for name in extra:
            print(f"  - {name}")

    if missing or extra:
        raise RuntimeError(
            "The image directory does not exactly match the official "
            "trainval/test image lists."
        )

    print(
        f"Validation passed: {EXPECTED_IMAGES} images, "
        f"{EXPECTED_CLASSES} classes."
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and validate Oxford-IIIT Pet."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw"),
        help="Parent directory used by torchvision for the dataset.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the official dataset before validation.",
    )
    return parser.parse_args()


def main() -> None:
    """Run dataset download when requested, then validate it."""
    args = parse_args()
    dataset_root = args.root / "oxford-iiit-pet"

    if args.download:
        download_dataset(args.root)

    validate_dataset(dataset_root)


if __name__ == "__main__":
    main()
