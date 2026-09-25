"""Build a reproducible Oxford-IIIT Pet dataset manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image

from pet_breed_mlops.labels import load_label_map

REQUIRED_FIELDS = (
    "image_id",
    "path",
    "breed",
    "species",
    "class_index",
    "split",
    "corruption",
    "severity",
    "width",
    "height",
)


def load_splits(path: Path) -> dict[str, list[str]]:
    """Load frozen dataset splits."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    return {
        "train": list(data["train"]),
        "validation": list(data["validation"]),
        "test": list(data["test"]),
    }


def species_from_breed(breed: str) -> str:
    """Infer species from the Oxford-IIIT Pet breed name."""
    cat_breeds = {
        "Abyssinian",
        "Bengal",
        "Birman",
        "Bombay",
        "British_Shorthair",
        "Egyptian_Mau",
        "Maine_Coon",
        "Persian",
        "Ragdoll",
        "Russian_Blue",
        "Siamese",
        "Sphynx",
    }

    return "cat" if breed in cat_breeds else "dog"


def breed_from_image_id(image_id: str) -> str:
    """Extract breed from an Oxford-IIIT Pet image ID."""
    return image_id.rsplit("_", 1)[0]


def build_manifest(
    dataset_root: Path,
    split_path: Path,
    label_map_path: Path,
) -> list[dict]:
    """Build one manifest record for every frozen dataset image."""
    images_root = dataset_root / "images"

    splits = load_splits(split_path)
    label_map = load_label_map(label_map_path)

    split_by_image: dict[str, str] = {}

    for split_name, image_names in splits.items():
        for image_name in image_names:
            if image_name in split_by_image:
                raise ValueError(f"Image appears in multiple splits: {image_name}")

            split_by_image[image_name] = split_name

    records: list[dict] = []

    for image_name in sorted(split_by_image):
        image_id = Path(image_name).stem
        breed = breed_from_image_id(image_id)

        if breed not in label_map:
            raise ValueError(f"Breed missing from label map: {breed}")

        image_path = images_root / image_name

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image: {image_path}")

        with Image.open(image_path) as image:
            width, height = image.size

        records.append(
            {
                "image_id": image_id,
                "path": str(image_path.as_posix()),
                "breed": breed,
                "species": species_from_breed(breed),
                "class_index": label_map[breed],
                "split": split_by_image[image_name],
                "corruption": "clean",
                "severity": 0,
                "width": width,
                "height": height,
            }
        )

    return records


def write_manifest(records: list[dict], output_path: Path) -> None:
    """Write the manifest as deterministic JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build the Oxford-IIIT Pet dataset manifest.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/oxford-iiit-pet"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/processed/splits.yaml"),
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=Path("configs/label_map.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/manifest.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Build and write the dataset manifest."""
    args = parse_args()

    records = build_manifest(
        dataset_root=args.dataset_root,
        split_path=args.splits,
        label_map_path=args.label_map,
    )

    write_manifest(records, args.output)

    print(f"Manifest records: {len(records)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
