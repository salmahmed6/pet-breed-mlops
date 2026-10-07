"""Generate an Evidently drift report from actual monitoring CSV datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an Evidently drift report.")
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("reports/monitoring/reference_monitoring.csv"),
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=Path("reports/monitoring/current_monitoring.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/monitoring/evidently_drift.html"),
    )
    return parser.parse_args()


def main() -> None:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:
        raise RuntimeError(
            "Evidently is required. Install project dependencies before running this script."
        ) from exc

    args = parse_args()
    reference = pd.read_csv(args.reference)
    current = pd.read_csv(args.current)
    if reference.empty or current.empty:
        raise ValueError("Reference and current monitoring datasets must not be empty.")

    report = Report([DataDriftPreset()])
    snapshot = report.run(current_data=current, reference_data=reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(args.output))
    print(f"Evidently report: {args.output}")


if __name__ == "__main__":
    main()
