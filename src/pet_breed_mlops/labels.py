"""Shared Oxford-IIIT Pet label-map loading utilities."""

from __future__ import annotations

import json
from pathlib import Path

EXPECTED_CLASSES = 37

DEFAULT_LABEL_MAP_PATH = Path(__file__).resolve().parents[2] / "configs" / "label_map.json"


def load_label_map(
    path: Path = DEFAULT_LABEL_MAP_PATH,
) -> dict[str, int]:
    """Load and validate the committed breed-to-class mapping."""
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("Label map must be a JSON object.")

    label_map = {str(breed): int(class_index) for breed, class_index in data.items()}

    if len(label_map) != EXPECTED_CLASSES:
        raise ValueError(f"Expected {EXPECTED_CLASSES} classes, found {len(label_map)}.")

    expected_indices = set(range(EXPECTED_CLASSES))

    if set(label_map.values()) != expected_indices:
        raise ValueError("Label map indices must contain every value from 0 to 36.")

    if list(label_map.values()) != sorted(label_map.values()):
        raise ValueError("Label map indices must be ordered deterministically.")

    return label_map
