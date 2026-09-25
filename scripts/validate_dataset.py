"""Validate the Oxford-IIIT Pet dataset before model training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from pet_breed_mlops.labels import load_label_map

MIN_TRAIN_IMAGES_PER_CLASS = 50
MIN_IMAGE_SIZE = 32
EXPECTED_CLASSES = 37


def load_manifest(path: Path) -> list[dict]:
    """Load the dataset manifest."""
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("Manifest must contain a JSON list.")

    return data


def validate_manifest_paths(records: list[dict]) -> list[str]:
    """Validate that every manifest image exists and can be opened."""
    errors: list[str] = []

    for record in records:
        path = Path(record["path"])

        if not path.is_file():
            errors.append(f"Missing image: {path}")
            continue

        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            errors.append(f"Cannot open image {path}: {exc}")

    return errors


def validate_split_overlap(records: list[dict]) -> list[str]:
    """Ensure no image appears in more than one split."""
    errors: list[str] = []
    image_splits: dict[str, set[str]] = {}

    for record in records:
        image_id = record["image_id"]
        split = record["split"]

        image_splits.setdefault(image_id, set()).add(split)

    for image_id, splits in image_splits.items():
        if len(splits) > 1:
            errors.append(f"Image {image_id} appears in multiple splits: {sorted(splits)}")

    return errors


def validate_label_map(label_map_path: Path) -> list[str]:
    """Validate the committed 37-class label map."""
    errors: list[str] = []

    try:
        label_map = load_label_map(label_map_path)
    except Exception as exc:
        return [f"Invalid label map: {exc}"]

    if len(label_map) != EXPECTED_CLASSES:
        errors.append(f"Expected {EXPECTED_CLASSES} classes, found {len(label_map)}.")

    return errors


def validate_training_distribution(records: list[dict]) -> list[str]:
    """Ensure every class has at least 50 training images."""
    errors: list[str] = []

    counts = Counter(record["class_index"] for record in records if record["split"] == "train")

    classes = sorted({record["class_index"] for record in records})

    for class_index in classes:
        count = counts.get(class_index, 0)

        if count < MIN_TRAIN_IMAGES_PER_CLASS:
            errors.append(
                f"Class {class_index} has only {count} training images; "
                f"minimum is {MIN_TRAIN_IMAGES_PER_CLASS}."
            )

    return errors


def validate_image_properties(records: list[dict]) -> list[str]:
    """Validate dimensions and RGB conversion."""
    errors: list[str] = []

    for record in records:
        path = Path(record["path"])

        if not path.is_file():
            continue

        try:
            with Image.open(path) as image:
                width, height = image.size

                if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                    errors.append(
                        f"Image {record['image_id']} is too small: "
                        f"{width}x{height}; minimum is "
                        f"{MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}."
                    )

                try:
                    image.convert("RGB")
                except Exception as exc:
                    errors.append(f"Image {record['image_id']} cannot be converted to RGB: {exc}")

        except Exception:
            # Opening errors are already reported by validate_manifest_paths.
            continue

    return errors


def validate_dataset(
    manifest_path: Path,
    label_map_path: Path,
) -> list[str]:
    """Run all dataset integrity validations."""
    records = load_manifest(manifest_path)

    errors: list[str] = []

    errors.extend(validate_manifest_paths(records))
    errors.extend(validate_split_overlap(records))
    errors.extend(validate_label_map(label_map_path))
    errors.extend(validate_training_distribution(records))
    errors.extend(validate_image_properties(records))

    return errors


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate the Oxford-IIIT Pet dataset.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifest.json"),
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=Path("configs/label_map.json"),
    )

    return parser.parse_args()


def main() -> None:
    """Run dataset validation."""
    args = parse_args()

    errors = validate_dataset(
        manifest_path=args.manifest,
        label_map_path=args.label_map,
    )

    if errors:
        print("Dataset validation FAILED.")
        print()

        for error in errors:
            print(f"- {error}")

        raise SystemExit(1)

    print("Dataset validation PASSED.")
    print("All manifest paths are valid.")
    print("No split overlap detected.")
    print("37-class label map validated.")
    print("All classes meet the minimum training-image requirement.")
    print("All images meet the minimum size requirement.")
    print("All images are RGB-convertible.")


if __name__ == "__main__":
    main()
