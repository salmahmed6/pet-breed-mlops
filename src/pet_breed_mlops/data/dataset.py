from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from .transforms import build_transforms


class PetBreedDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        image_size: int = 224,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split

        with self.manifest_path.open("r", encoding="utf-8") as file:
            records: list[dict[str, Any]] = json.load(file)

        self.records = [
            record
            for record in records
            if record["split"] == split
        ]

        if not self.records:
            raise ValueError(f"No records found for split: {split}")

        self.transform = build_transforms(
            split=split,
            image_size=image_size,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]

        image_path = Path(record["path"])

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = int(record["class_index"])

        return image, label