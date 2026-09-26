import json
from pathlib import Path

from PIL import Image

from scripts.validate_dataset import (
    validate_image_properties,
    validate_label_map,
    validate_manifest_paths,
    validate_split_overlap,
    validate_training_distribution,
)


def create_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    Image.new("RGB", size).save(path)


def test_manifest_paths_are_valid(tmp_path: Path) -> None:
    image_path = tmp_path / "pet.jpg"
    create_image(image_path)

    records = [
        {
            "image_id": "pet_1",
            "path": str(image_path),
            "split": "train",
        }
    ]

    assert validate_manifest_paths(records) == []


def test_missing_manifest_path_is_reported(tmp_path: Path) -> None:
    records = [
        {
            "image_id": "missing_1",
            "path": str(tmp_path / "missing.jpg"),
            "split": "train",
        }
    ]

    errors = validate_manifest_paths(records)

    assert len(errors) == 1
    assert "Missing image" in errors[0]


def test_split_overlap_is_detected() -> None:
    records = [
        {
            "image_id": "pet_1",
            "split": "train",
        },
        {
            "image_id": "pet_1",
            "split": "test",
        },
    ]

    errors = validate_split_overlap(records)

    assert len(errors) == 1
    assert "multiple splits" in errors[0]


def test_training_distribution_requires_50_images() -> None:
    records = [
        {
            "class_index": 0,
            "split": "train",
        }
    ]

    errors = validate_training_distribution(records)

    assert len(errors) == 1
    assert "minimum is 50" in errors[0]


def test_small_image_is_rejected(tmp_path: Path) -> None:
    image_path = tmp_path / "small.jpg"
    create_image(image_path, (16, 16))

    records = [
        {
            "image_id": "small_1",
            "path": str(image_path),
        }
    ]

    errors = validate_image_properties(records)

    assert len(errors) == 1
    assert "too small" in errors[0]


def test_rgb_conversion_is_validated(tmp_path: Path) -> None:
    image_path = tmp_path / "pet.png"
    Image.new("RGBA", (64, 64)).save(image_path)

    records = [
        {
            "image_id": "pet_1",
            "path": str(image_path),
        }
    ]

    assert validate_image_properties(records) == []


def test_label_map_requires_37_classes(tmp_path: Path) -> None:
    label_map_path = tmp_path / "label_map.json"

    label_map = {f"class_{i}": i for i in range(3)}

    label_map_path.write_text(
        json.dumps(label_map),
        encoding="utf-8",
    )

    errors = validate_label_map(label_map_path)

    assert len(errors) == 1
    assert "Expected 37 classes" in errors[0]
