"""Generate deterministic corrupted copies of the Oxford-IIIT Pet test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from pet_breed_mlops.corruptions import CORRUPTION_TYPES, SEVERITIES, apply_corruption


def generate_corruptions(
    manifest_path: Path,
    output_root: Path,
) -> list[dict]:
    """Generate corrupted copies without modifying source images."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    test_records = [record for record in manifest if record["split"] == "test"]

    generated: list[dict] = []

    for record in test_records:
        source_path = Path(record["path"])

        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source image: {source_path}")

        with Image.open(source_path) as source:
            image = source.convert("RGB")

            for corruption in CORRUPTION_TYPES:
                for severity in SEVERITIES:
                    corrupted = apply_corruption(
                        image=image,
                        corruption=corruption,
                        severity=severity,
                    )

                    output_dir = output_root / corruption / f"severity_{severity}"
                    output_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    output_path = output_dir / f"{record['image_id']}.jpg"

                    output_quality = 30 if corruption == "jpeg_quality" else 95

                    corrupted.save(
                        output_path,
                        format="JPEG",
                        quality=output_quality,
                    )

                    generated.append(
                        {
                            "image_id": record["image_id"],
                            "source_path": record["path"],
                            "path": output_path.as_posix(),
                            "breed": record["breed"],
                            "species": record["species"],
                            "class_index": record["class_index"],
                            "split": "test",
                            "corruption": corruption,
                            "severity": severity,
                            "width": corrupted.width,
                            "height": corrupted.height,
                        }
                    )

    return generated


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate Track C image corruptions.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/corrupted"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/corrupted/metadata.json"),
    )

    return parser.parse_args()


def main() -> None:
    """Generate and record corrupted test images."""
    args = parse_args()

    records = generate_corruptions(
        manifest_path=args.manifest,
        output_root=args.output,
    )

    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    args.metadata.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Source test images: {len(records) // 18}")
    print(f"Corrupted images: {len(records)}")
    print(f"Corruption types: {len(CORRUPTION_TYPES)}")
    print(f"Severity levels: {len(SEVERITIES)}")
    print(f"Output: {args.output}")
    print(f"Metadata: {args.metadata}")


if __name__ == "__main__":
    main()
