import json
from pathlib import Path

from PIL import Image

from scripts.create_manifest import (
    REQUIRED_FIELDS,
    build_manifest,
    write_manifest,
)


def create_test_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset_root = tmp_path / "dataset"
    images = dataset_root / "images"
    images.mkdir(parents=True)

    Image.new("RGB", (64, 48)).save(images / "Abyssinian_1.jpg")
    Image.new("RGB", (32, 32)).save(images / "boxer_1.jpg")
    Image.new("RGB", (80, 60)).save(images / "pug_1.jpg")

    splits = tmp_path / "splits.yaml"
    splits.write_text(
        """
train:
  - Abyssinian_1.jpg
validation:
  - boxer_1.jpg
test:
  - pug_1.jpg
""".strip()
        + "\n",
        encoding="utf-8",
    )

    breeds = [
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
        "american_bulldog",
        "american_pit_bull_terrier",
        "basset_hound",
        "beagle",
        "boxer",
        "chihuahua",
        "english_cocker_spaniel",
        "english_setter",
        "german_shorthaired",
        "great_pyrenees",
        "havanese",
        "japanese_chin",
        "keeshond",
        "leonberger",
        "miniature_pinscher",
        "newfoundland",
        "pomeranian",
        "pug",
        "saint_bernard",
        "samoyed",
        "scottish_terrier",
        "shiba_inu",
        "staffordshire_bull_terrier",
        "wheaten_terrier",
        "yorkshire_terrier",
    ]

    label_map = tmp_path / "label_map.json"
    label_map.write_text(
        json.dumps({breed: index for index, breed in enumerate(breeds)}),
        encoding="utf-8",
    )

    return dataset_root, splits, label_map


def test_manifest_contains_required_fields(tmp_path: Path) -> None:
    dataset_root, splits, label_map = create_test_dataset(tmp_path)

    records = build_manifest(dataset_root, splits, label_map)

    assert len(records) == 3

    for record in records:
        assert set(record) == set(REQUIRED_FIELDS)


def test_manifest_uses_frozen_split_metadata(tmp_path: Path) -> None:
    dataset_root, splits, label_map = create_test_dataset(tmp_path)

    records = build_manifest(dataset_root, splits, label_map)

    split_by_image = {record["image_id"]: record["split"] for record in records}

    assert split_by_image == {
        "Abyssinian_1": "train",
        "boxer_1": "validation",
        "pug_1": "test",
    }


def test_manifest_captures_actual_dimensions(tmp_path: Path) -> None:
    dataset_root, splits, label_map = create_test_dataset(tmp_path)

    records = build_manifest(dataset_root, splits, label_map)

    dimensions = {record["image_id"]: (record["width"], record["height"]) for record in records}

    assert dimensions == {
        "Abyssinian_1": (64, 48),
        "boxer_1": (32, 32),
        "pug_1": (80, 60),
    }


def test_manifest_is_reproducible(tmp_path: Path) -> None:
    dataset_root, splits, label_map = create_test_dataset(tmp_path)

    records_one = build_manifest(dataset_root, splits, label_map)
    records_two = build_manifest(dataset_root, splits, label_map)

    output_one = tmp_path / "manifest_one.json"
    output_two = tmp_path / "manifest_two.json"

    write_manifest(records_one, output_one)
    write_manifest(records_two, output_two)

    assert output_one.read_text(encoding="utf-8") == (output_two.read_text(encoding="utf-8"))
