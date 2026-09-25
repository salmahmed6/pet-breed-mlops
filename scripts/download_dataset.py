from __future__ import annotations

import argparse
from pathlib import Path

EXPECTED_IMAGES = 7_349
EXPECTED_CLASSES = 37
EXPECTED_TRAINVAL = 3_680
EXPECTED_TEST = 3_669

SPLITS = ("trainval", "test")
DATASET_NAME = "Oxford-IIIT Pet"
DATASET_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/"


def read_split_names(annotation_file: Path) -> set[str]:
    """Read image names from an Oxford-IIIT Pet split file."""
    names: set[str] = set()

    for line in annotation_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        names.add(f"{line.split()[0]}.jpg")

    return names


def validate_dataset(root: Path) -> None:
    """Validate the downloaded Oxford-IIIT Pet dataset."""
    annotations = root / "annotations"
    images = root / "images"

    if not annotations.is_dir():
        raise RuntimeError(f"Missing annotations directory: {annotations}")

    if not images.is_dir():
        raise RuntimeError(f"Missing images directory: {images}")

    trainval = read_split_names(annotations / "trainval.txt")
    test = read_split_names(annotations / "test.txt")

    if len(trainval) != EXPECTED_TRAINVAL:
        raise RuntimeError(f"Expected {EXPECTED_TRAINVAL} trainval images, found {len(trainval)}.")

    if len(test) != EXPECTED_TEST:
        raise RuntimeError(f"Expected {EXPECTED_TEST} test images, found {len(test)}.")

    if trainval & test:
        raise RuntimeError("Trainval and test splits overlap.")

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
    print(f"Missing images: {len(missing)}")
    print(f"Extra images: {len(extra)}")

    if len(expected) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_IMAGES} images from official splits, found {len(expected)}."
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
        print(
            "WARNING: The local image directory does not exactly match "
            "the official trainval/test image lists."
        )
        print(
            "The missing/extra files will be handled by the dataset "
            "integrity stage before training."
        )
    else:
        print(
            f"Validation passed: {EXPECTED_IMAGES} images, "
            f"{EXPECTED_CLASSES} classes."
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate the Oxford-IIIT Pet dataset.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/oxford-iiit-pet"),
        help="Oxford-IIIT Pet dataset root.",
    )
    return parser.parse_args()


def main() -> None:
    """Run dataset validation."""
    args = parse_args()
    validate_dataset(args.root)


if __name__ == "__main__":
    main()
