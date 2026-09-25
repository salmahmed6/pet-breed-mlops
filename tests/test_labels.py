import json
from pathlib import Path

from pet_breed_mlops.labels import EXPECTED_CLASSES, load_label_map
from scripts.create_label_map import build_label_map

DATASET_ROOT = Path("data/raw/oxford-iiit-pet")
REFERENCE_PATH = Path("configs/label_map.json")


def test_committed_label_map_has_37_classes() -> None:
    label_map = load_label_map(REFERENCE_PATH)

    assert len(label_map) == EXPECTED_CLASSES
    assert set(label_map.values()) == set(range(EXPECTED_CLASSES))


def test_label_map_is_sorted_deterministically() -> None:
    label_map = load_label_map(REFERENCE_PATH)

    breeds = list(label_map)

    assert breeds == sorted(breeds)


def test_generated_mapping_matches_committed_reference() -> None:
    generated = build_label_map(DATASET_ROOT)

    committed = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))

    committed = {str(breed): int(class_index) for breed, class_index in committed.items()}

    assert generated == committed
