"""Run the Sprint 4 drift and corruption scorecard."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from pet_breed_mlops.drift import (
    build_scorecard,
    load_checkpoint_model,
    load_json_records,
    write_scorecard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Track C drift scorecard.")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/models/resnet18_best.pt")
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.json"))
    parser.add_argument(
        "--corrupted-metadata", type=Path, default=Path("data/corrupted/metadata.json")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/monitoring.yaml"))
    parser.add_argument(
        "--output-json", type=Path, default=Path("reports/monitoring/drift_scorecard.json")
    )
    parser.add_argument(
        "--output-csv", type=Path, default=Path("reports/monitoring/drift_scorecard.csv")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model, _ = load_checkpoint_model(args.checkpoint)
    manifest = load_json_records(args.manifest)
    corrupted = load_json_records(args.corrupted_metadata)
    clean_records = [record for record in manifest if record["split"] == "test"]
    scorecard = build_scorecard(
        model,
        clean_records,
        corrupted,
        image_size=int(config["image_size"]),
        max_samples_per_group=int(config["max_samples_per_group"]),
        mmd_threshold=float(config["thresholds"]["embedding_mmd"]),
        pixel_threshold=float(config["thresholds"]["pixel_mmd"]),
        confidence_threshold=float(config["thresholds"]["confidence_drift"]),
    )
    write_scorecard(scorecard, args.output_json, args.output_csv)
    print(f"JSON report: {args.output_json}")
    print(f"CSV report: {args.output_csv}")


if __name__ == "__main__":
    main()
