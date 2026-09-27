import torch

from pet_breed_mlops.calibration import (
    calibrate,
    choose_abstention_threshold,
    expected_calibration_error,
    fit_temperature,
    selective_metrics,
)


def _logits_and_labels() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.tensor(
        [
            [3.0, 0.5, -1.0],
            [0.2, 2.5, -0.4],
            [2.0, 1.0, 0.0],
            [-0.2, 0.4, 2.8],
            [1.2, 2.0, 0.1],
            [2.5, 0.2, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0, 2, 1, 0], dtype=torch.long)
    return logits, labels


def test_temperature_scaling_is_deterministic() -> None:
    logits, labels = _logits_and_labels()
    first = fit_temperature(logits, labels)
    second = fit_temperature(logits, labels)
    assert first == second
    assert first > 0.0


def test_ece_is_deterministic_and_bounded() -> None:
    logits, labels = _logits_and_labels()
    probabilities = torch.softmax(logits, dim=1)
    first = expected_calibration_error(probabilities, labels)
    second = expected_calibration_error(probabilities, labels)
    assert first == second
    assert 0.0 <= first <= 1.0


def test_abstention_threshold_is_deterministic() -> None:
    logits, labels = _logits_and_labels()
    probabilities = torch.softmax(logits, dim=1)
    first = choose_abstention_threshold(probabilities, labels)
    second = choose_abstention_threshold(probabilities, labels)
    assert first == second
    threshold, coverage, selective_accuracy = first
    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= coverage <= 1.0
    assert 0.0 <= selective_accuracy <= 1.0


def test_selective_accuracy_uses_only_confident_predictions() -> None:
    logits, labels = _logits_and_labels()
    probabilities = torch.softmax(logits, dim=1)
    threshold = 0.9
    coverage, accuracy = selective_metrics(probabilities, labels, threshold)
    selected = probabilities.max(dim=1).values >= threshold
    expected_accuracy = float(
        probabilities.argmax(dim=1)[selected].eq(labels[selected]).float().mean().item()
    )
    assert coverage == float(selected.float().mean().item())
    assert accuracy == expected_accuracy


def test_calibration_result_contains_abstention_fields() -> None:
    logits, labels = _logits_and_labels()
    result = calibrate(logits, labels)
    assert result.temperature > 0.0
    assert 0.0 <= result.raw_ece <= 1.0
    assert 0.0 <= result.calibrated_ece <= 1.0
    assert 0.0 <= result.coverage <= 1.0
    assert 0.0 <= result.selective_accuracy <= 1.0
