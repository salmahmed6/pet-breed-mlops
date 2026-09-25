"""Create a deterministic Oxford-IIIT Pet breed label map."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts.download_dataset import read_split

EXPECTED_CLASSES = 37
IMAGE_ID_PATTERN = re.compile(r"^(?P<breed>.+)_(?P<number>\d+)$")


def breed_from_image_name(image_name: str) -> str:
    """Extract the breed name from an Oxford-IIIT Pet image filename."""
    image_id = Path(image_name).stem
    match = IMAGE_ID_PATTERN.match(image_id)

    if match is None:
        raise ValueError(f"Invalid Oxford-IIIT Pet image ID: {image_id}")

    return match.group("breed")


def build_label_map(dataset_root: Path) -> dict[str, int]:
    """Build a deterministic zero-based breed-to-class mapping."""
    annotations = dataset_root / "annotations"

    trainval_names, _ = read_split(annotations / "trainval.txt")
    test_names, _ = read_split(annotations / "test.txt")

    image_names = trainval_names | test_names

    breeds = {breed_from_image_name(image_name) for image_name in image_names}

    if len(breeds) != EXPECTED_CLASSES:
        raise RuntimeError(f"Expected {EXPECTED_CLASSES} breeds, found {len(breeds)}.")

    return {breed: class_index for class_index, breed in enumerate(sorted(breeds))}


def write_label_map(label_map: dict[str, int], output_path: Path) -> None:
    """Write the label map as a stable JSON artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(label_map, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Create deterministic Oxford-IIIT Pet label map.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/oxford-iiit-pet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/label_map.json"),
    )
    return parser.parse_args()


def main() -> None:
    """Generate the deterministic label map."""
    args = parse_args()

    label_map = build_label_map(args.dataset_root)
    write_label_map(label_map, args.output)

    print(f"Classes: {len(label_map)}")
    print(f"Output: {args.output}")

    for breed, class_index in label_map.items():
        print(f"{class_index:02d}: {breed}")


if __name__ == "__main__":
    main()
