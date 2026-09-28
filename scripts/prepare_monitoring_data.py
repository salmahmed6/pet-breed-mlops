"""Prepare actual reference/current datasets for Evidently from the monitoring inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from pet_breed_mlops.drift import load_json_records
from pet_breed_mlops.serving import build_inference_transform, load_checkpoint_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Evidently monitoring datasets.")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/models/resnet18_best.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.json"))
    parser.add_argument(
        "--corrupted-metadata", type=Path, default=Path("data/corrupted/metadata.json")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/monitoring.yaml"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/monitoring")
    )
    return parser.parse_args()


def image_row(
    model: torch.nn.Module,
    record: dict[str, object],
    transform,
    source: str,
) -> dict[str, object]:
    with Image.open(str(record["path"])) as image:
        rgb = image.convert("RGB")
        tensor = transform(rgb).unsqueeze(0)
        array = torch.from_numpy(np.asarray(rgb)).float() / 255.0
        brightness = float(array.mean())
        contrast = float(array.std())
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=-1)[0]
    predicted = int(probabilities.argmax().item())
    confidence = float(probabilities.max().item())
    return {
        "source": source,
        "image_id": str(record["image_id"]),
        "class_index": int(record["class_index"]),
        "prediction": predicted,
        "correct": int(predicted == int(record["class_index"])),
        "confidence": confidence,
        "brightness": brightness,
        "contrast": contrast,
    }


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    limit = int(config["max_samples_per_group"])
    model, _ = load_checkpoint_model(args.checkpoint)
    transform = build_inference_transform(int(config["image_size"]))

    manifest = load_json_records(args.manifest)
    corrupted = load_json_records(args.corrupted_metadata)
    clean = sorted(
        (record for record in manifest if record["split"] == "test"),
        key=lambda record: str(record["image_id"]),
    )[:limit]

    current_groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for record in corrupted:
        key = (str(record["corruption"]), int(record["severity"]))
        current_groups.setdefault(key, []).append(record)

    reference_rows = [image_row(model, record, transform, "reference") for record in clean]
    current_rows: list[dict[str, object]] = []
    for key in sorted(current_groups):
        records = sorted(
            current_groups[key], key=lambda record: str(record["image_id"])
        )[:limit]
        current_rows.extend(image_row(model, record, transform, f"{key[0]}_severity_{key[1]}") for record in records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(reference_rows).to_csv(
        args.output_dir / "reference_monitoring.csv", index=False
    )
    pd.DataFrame(current_rows).to_csv(
        args.output_dir / "current_monitoring.csv", index=False
    )
    print(f"Reference rows: {len(reference_rows)}")
    print(f"Current rows: {len(current_rows)}")


if __name__ == "__main__":
    main()
