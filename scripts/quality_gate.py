from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_top_1(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("top_1_accuracy")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{path} must contain numeric 'top_1_accuracy'")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{path} top_1_accuracy must be between 0 and 1")
    return float(value)


def check_quality(
    baseline_path: Path,
    candidate_path: Path,
    minimum_delta: float = 0.0,
) -> tuple[float, float]:
    baseline = load_top_1(baseline_path)
    candidate = load_top_1(candidate_path)
    minimum = baseline + minimum_delta

    if candidate < minimum:
        raise SystemExit(
            f"Model quality gate failed: candidate top-1={candidate:.6f} "
            f"is below required baseline={minimum:.6f}."
        )

    print(
        f"Model quality gate passed: candidate top-1={candidate:.6f}, "
        f"required baseline={minimum:.6f}."
    )
    return baseline, candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Check model top-1 against a committed baseline.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--minimum-delta", type=float, default=0.0)
    args = parser.parse_args()

    check_quality(args.baseline, args.candidate, args.minimum_delta)


if __name__ == "__main__":
    main()
