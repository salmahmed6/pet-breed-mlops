from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn


@dataclass(frozen=True)
class CalibrationResult:
    temperature: float
    raw_ece: float
    calibrated_ece: float
    abstention_threshold: float
    coverage: float
    selective_accuracy: float
    calibrated_confidence: list[float]
    predictions: list[int]
    labels: list[int]


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    max_iter: int = 50,
) -> float:
    """Fit a single positive temperature using validation logits only."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [N, C]")
    if labels.ndim != 1 or labels.numel() != logits.shape[0]:
        raise ValueError("labels must have shape [N] matching logits")

    logits = logits.detach().float().cpu()
    labels = labels.detach().long().cpu()

    temperature = nn.Parameter(torch.ones(1, dtype=torch.float32))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(
        [temperature],
        lr=0.1,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        safe_temperature = temperature.clamp_min(1e-3)
        loss = criterion(logits / safe_temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp_min(1e-3).item())


def expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_bins: int = 10,
) -> float:
    """Compute ECE using confidence bins over the supplied evaluation split."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [N, C]")
    if labels.ndim != 1 or labels.numel() != probabilities.shape[0]:
        raise ValueError("labels must have shape [N] matching probabilities")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels)
    ece = torch.zeros(1, dtype=torch.float64)

    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if bin_index == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if mask.any():
            fraction = mask.float().mean().double()
            accuracy = correct[mask].float().mean().double()
            avg_confidence = confidence[mask].double().mean()
            ece += torch.abs(avg_confidence - accuracy) * fraction

    return float(ece.item())


def reliability_data(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels)
    rows: list[dict[str, float | int]] = []

    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if bin_index == n_bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        count = int(mask.sum().item())
        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "confidence": float(confidence[mask].mean().item()) if count else 0.0,
                "accuracy": float(correct[mask].float().mean().item()) if count else 0.0,
            }
        )

    return rows


def selective_metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    threshold: float,
) -> tuple[float, float]:
    confidence, predictions = probabilities.max(dim=1)
    selected = confidence >= threshold
    coverage = float(selected.float().mean().item())
    if not selected.any():
        return coverage, 0.0
    accuracy = float(predictions[selected].eq(labels[selected]).float().mean().item())
    return coverage, accuracy


def choose_abstention_threshold(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float, float]:
    """Choose a deterministic threshold from validation calibrated confidences."""
    confidence = probabilities.max(dim=1).values
    candidates = torch.unique(confidence).sort().values

    best_threshold = 1.0
    best_coverage = 0.0
    best_accuracy = 0.0

    for threshold in candidates:
        coverage, accuracy = selective_metrics(
            probabilities,
            labels,
            float(threshold.item()),
        )
        if (accuracy > best_accuracy) or (accuracy == best_accuracy and coverage > best_coverage):
            best_threshold = float(threshold.item())
            best_coverage = coverage
            best_accuracy = accuracy

    return best_threshold, best_coverage, best_accuracy


def calibrate(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_bins: int = 10,
) -> CalibrationResult:
    temperature = fit_temperature(logits, labels)
    raw_probabilities = torch.softmax(logits.detach().float().cpu(), dim=1)
    calibrated_probabilities = torch.softmax(
        logits.detach().float().cpu() / temperature,
        dim=1,
    )

    threshold, coverage, selective_accuracy = choose_abstention_threshold(
        calibrated_probabilities,
        labels.detach().long().cpu(),
    )

    return CalibrationResult(
        temperature=temperature,
        raw_ece=expected_calibration_error(
            raw_probabilities,
            labels.detach().long().cpu(),
            n_bins=n_bins,
        ),
        calibrated_ece=expected_calibration_error(
            calibrated_probabilities,
            labels.detach().long().cpu(),
            n_bins=n_bins,
        ),
        abstention_threshold=threshold,
        coverage=coverage,
        selective_accuracy=selective_accuracy,
        calibrated_confidence=calibrated_probabilities.max(dim=1).values.tolist(),
        predictions=calibrated_probabilities.argmax(dim=1).tolist(),
        labels=labels.detach().long().cpu().tolist(),
    )


def save_reliability_diagram(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    output_path: str | Path,
    *,
    title: str,
    n_bins: int = 10,
) -> None:
    rows = reliability_data(probabilities, labels, n_bins=n_bins)
    centers = [(row["bin_lower"] + row["bin_upper"]) / 2 for row in rows]
    accuracies = [row["accuracy"] for row in rows]

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    axis.plot(centers, accuracies, marker="o", label="Model")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Accuracy")
    axis.set_title(title)
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def result_to_json(result: CalibrationResult) -> dict[str, object]:
    return {
        "temperature": result.temperature,
        "raw_ece": result.raw_ece,
        "calibrated_ece": result.calibrated_ece,
        "abstention_threshold": result.abstention_threshold,
        "coverage": result.coverage,
        "selective_accuracy": result.selective_accuracy,
        "calibrated_confidence": result.calibrated_confidence,
        "predictions": result.predictions,
        "labels": result.labels,
    }
