"""Evaluate the drift scorecard and return whether retraining is required."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.run_drift_scorecard import main as run_scorecard


def _contains_trigger(payload: Any) -> bool:
    """Find an explicit drift flag in scorecard output.

    The scorecard schema is intentionally treated defensively because monitoring
    reports may evolve. Explicit boolean drift flags are preferred over numeric
    metrics; configured thresholds remain the source of truth inside the scorecard.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = str(key).lower()
            if normalized in {"drift_detected", "retraining_required", "trigger_retraining"}:
                if isinstance(value, bool):
                    return value
            if normalized in {"drift", "drift_flag", "flags"} and isinstance(value, bool):
                return value
        return any(_contains_trigger(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_contains_trigger(value) for value in payload)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run drift scorecard and decide retraining.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/models/resnet18_best.pt"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifest.json"),
    )
    parser.add_argument(
        "--corrupted-metadata",
        type=Path,
        default=Path("data/corrupted/metadata.json"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/monitoring.yaml"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/monitoring/drift_scorecard.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/monitoring/drift_scorecard.csv"),
    )
    parser.add_argument(
        "--decision-file",
        type=Path,
        default=Path("reports/retraining/drift_decision.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.decision_file.parent.mkdir(parents=True, exist_ok=True)

    # Reuse the existing Sprint 4 scorecard implementation.
    run_scorecard_args = [
        "--checkpoint", str(args.checkpoint),
        "--manifest", str(args.manifest),
        "--corrupted-metadata", str(args.corrupted_metadata),
        "--config", str(args.config),
        "--output-json", str(args.output_json),
        "--output-csv", str(args.output_csv),
    ]

    import sys

    previous = sys.argv
    try:
        sys.argv = ["run_drift_scorecard"] + run_scorecard_args
        run_scorecard()
    finally:
        sys.argv = previous

    payload = json.loads(args.output_json.read_text(encoding="utf-8"))
    groups = payload.get("groups", [])
    drift_detected = any(
        bool(group.get("embedding_drift"))
        or bool(group.get("pixel_drift"))
        or bool(group.get("confidence_drift_flag"))
        for group in groups
        if isinstance(group, dict)
    ) or _contains_trigger(payload)

    decision = {
        "drift_detected": drift_detected,
        "retraining_required": drift_detected,
        "trigger_reasons": [
            key
            for key in ("embedding_drift", "pixel_drift", "confidence_drift_flag")
            if any(bool(group.get(key)) for group in groups if isinstance(group, dict))
        ],
        "scorecard": str(args.output_json),
    }
    args.decision_file.write_text(json.dumps(decision, indent=2), encoding="utf-8")

    print(json.dumps(decision, indent=2))
    if not drift_detected:
        raise SystemExit(0)


if __name__ == "__main__":
    main()
