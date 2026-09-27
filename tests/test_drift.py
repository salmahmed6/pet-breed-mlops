from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from pet_breed_mlops.drift import build_scorecard, image_statistics, rbf_mmd, scalar_drift


def test_mmd_is_zero_for_identical_samples() -> None:
    values = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    assert rbf_mmd(values, values) == 0.0


def test_mmd_detects_shift() -> None:
    reference = np.zeros((8, 2))
    current = np.ones((8, 2))
    assert rbf_mmd(reference, current) > 0.1


def test_scalar_drift_detects_confidence_shift() -> None:
    reference = [0.5, 0.55, 0.6, 0.65]
    current = [0.8, 0.85, 0.9, 0.95]
    assert scalar_drift(reference, current) > 1.0


def test_image_statistics_are_deterministic() -> None:
    image = Image.new("RGB", (32, 32), (100, 120, 140))
    assert np.array_equal(image_statistics(image), image_statistics(image))


def test_scorecard_contains_all_detector_flags(tmp_path) -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.avgpool(x).flatten(1)
            return self.classifier(features)

    model = TinyModel().eval()
    clean_path = tmp_path / "clean.jpg"
    corrupted_path = tmp_path / "corrupt.jpg"
    Image.new("RGB", (32, 32), (80, 80, 80)).save(clean_path)
    Image.new("RGB", (32, 32), (200, 200, 200)).save(corrupted_path)
    clean = [{"image_id": "clean_1", "path": str(clean_path), "class_index": 0, "split": "test"}]
    corrupted = [
        {
            "image_id": "clean_1",
            "path": str(corrupted_path),
            "class_index": 0,
            "corruption": "brightness_up",
            "severity": 1,
        }
    ]
    scorecard = build_scorecard(model, clean, corrupted, max_samples_per_group=1)
    assert scorecard["clean_reference"]["samples"] == 1
    assert len(scorecard["groups"]) == 1
    assert {"embedding_mmd", "pixel_mmd", "confidence_drift"}.issubset(scorecard["groups"][0])
