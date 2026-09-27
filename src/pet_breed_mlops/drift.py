"""Reproducible drift detectors and clean-vs-corrupted scorecards."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter

from pet_breed_mlops.serving import build_inference_transform

DEFAULT_MMD_THRESHOLD = 0.05
DEFAULT_PIXEL_THRESHOLD = 0.10
DEFAULT_CONFIDENCE_THRESHOLD = 0.05


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Drift detector inputs must not be empty.")
    return array


def rbf_mmd(reference: np.ndarray, current: np.ndarray, sigma: float | None = None) -> float:
    """Compute an RBF MMD estimate."""
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(current, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("MMD inputs must be 2-D arrays with matching feature dimensions.")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("MMD inputs must not be empty.")

    if sigma is None:
        combined = np.vstack([x, y])
        distances = np.linalg.norm(
            combined[:, None, :] - combined[None, :, :],
            axis=-1,
        )
        non_zero = distances[distances > 0]
        sigma = float(np.median(non_zero)) if non_zero.size else 1.0
    sigma = max(float(sigma), 1e-8)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        distances = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-distances / (2.0 * sigma**2))

    return float(kernel(x, x).mean() + kernel(y, y).mean() - 2.0 * kernel(x, y).mean())


def scalar_drift(reference: Iterable[float], current: Iterable[float]) -> float:
    """Return normalized absolute mean shift."""
    ref = _as_float_array(reference)
    cur = _as_float_array(current)
    scale = max(float(np.std(ref)), 1e-8)
    return float(abs(np.mean(cur) - np.mean(ref)) / scale)


def image_statistics(image: Image.Image) -> np.ndarray:
    """Extract deterministic brightness, contrast, and edge statistics."""
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    grayscale = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
    edges = (
        np.asarray(
            rgb.convert("L").filter(ImageFilter.FIND_EDGES),
            dtype=np.float32,
        )
        / 255.0
    )
    return np.array(
        [
            float(array.mean()),
            float(array.std()),
            float(grayscale.mean()),
            float(grayscale.std()),
            float(edges.mean()),
            float(edges.std()),
        ],
        dtype=np.float64,
    )


def pixel_drift(reference_images: list[Image.Image], current_images: list[Image.Image]) -> float:
    """Measure drift between image-statistics distributions."""
    reference = np.vstack([image_statistics(image) for image in reference_images])
    current = np.vstack([image_statistics(image) for image in current_images])
    return rbf_mmd(reference, current)


def _feature_module(model: torch.nn.Module) -> torch.nn.Module:
    """Return a stable visual feature module for supported backbones."""
    if hasattr(model, "avgpool"):
        return model.avgpool
    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, torch.nn.Sequential) and len(classifier) >= 2:
        return classifier[0]
    raise ValueError("Unsupported model architecture for embedding extraction.")


def extract_embeddings(
    model: torch.nn.Module,
    images: list[Image.Image],
    image_size: int = 224,
) -> np.ndarray:
    """Extract visual embeddings before the classification head."""
    if not images:
        raise ValueError("Embedding input must not be empty.")

    transform = build_inference_transform(image_size)
    batch = torch.stack([transform(image.convert("RGB")) for image in images])
    feature_module = _feature_module(model)
    captured: list[torch.Tensor] = []

    def hook(
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        captured.append(output.detach().cpu())

    handle = feature_module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(batch)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("The embedding feature hook did not capture model features.")

    return captured[0].flatten(start_dim=1).numpy()


def embedding_drift(
    model: torch.nn.Module,
    reference_images: list[Image.Image],
    current_images: list[Image.Image],
    image_size: int = 224,
) -> float:
    """Measure visual embedding drift using RBF MMD."""
    reference = extract_embeddings(model, reference_images, image_size)
    current = extract_embeddings(model, current_images, image_size)
    return rbf_mmd(reference, current)


def _load_images(records: list[dict[str, Any]], max_samples: int) -> list[Image.Image]:
    selected = sorted(records, key=lambda record: str(record["image_id"]))[:max_samples]
    images: list[Image.Image] = []
    for record in selected:
        with Image.open(record["path"]) as image:
            images.append(image.convert("RGB").copy())
    return images


def _predict_records(
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    image_size: int,
) -> tuple[int, int, list[float]]:
    transform = build_inference_transform(image_size)
    correct = 0
    confidences: list[float] = []
    with torch.inference_mode():
        for record in records:
            with Image.open(record["path"]) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0)
            logits = model(tensor)
            probabilities = torch.softmax(logits, dim=-1)[0]
            predicted = int(probabilities.argmax().item())
            correct += int(predicted == int(record["class_index"]))
            confidences.append(float(probabilities.max().item()))
    return correct, len(records), confidences


def build_scorecard(
    model: torch.nn.Module,
    clean_records: list[dict[str, Any]],
    corrupted_records: list[dict[str, Any]],
    *,
    image_size: int = 224,
    max_samples_per_group: int = 32,
    mmd_threshold: float = DEFAULT_MMD_THRESHOLD,
    pixel_threshold: float = DEFAULT_PIXEL_THRESHOLD,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Build a deterministic clean-vs-corrupted monitoring scorecard."""
    clean_records = sorted(clean_records, key=lambda record: str(record["image_id"]))
    corrupted_records = sorted(
        corrupted_records,
        key=lambda record: (
            str(record["corruption"]),
            int(record["severity"]),
            str(record["image_id"]),
        ),
    )
    clean_subset = clean_records[:max_samples_per_group]
    clean_images = _load_images(clean_subset, max_samples_per_group)
    clean_correct, clean_total, clean_conf = _predict_records(
        model,
        clean_subset,
        image_size,
    )
    reference_embeddings = extract_embeddings(model, clean_images, image_size)
    reference_pixels = np.vstack([image_statistics(image) for image in clean_images])

    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in corrupted_records:
        groups.setdefault(
            (str(record["corruption"]), int(record["severity"])),
            [],
        ).append(record)

    rows: list[dict[str, Any]] = []
    for (corruption, severity), records in sorted(groups.items()):
        subset = records[:max_samples_per_group]
        images = _load_images(subset, max_samples_per_group)
        embeddings = extract_embeddings(model, images, image_size)
        pixels = np.vstack([image_statistics(image) for image in images])
        correct, total, confidences = _predict_records(model, subset, image_size)
        embedding_score = rbf_mmd(reference_embeddings, embeddings)
        pixel_score = rbf_mmd(reference_pixels, pixels)
        confidence_score = scalar_drift(clean_conf, confidences)

        rows.append(
            {
                "corruption": corruption,
                "severity": severity,
                "samples": total,
                "accuracy": correct / total if total else 0.0,
                "embedding_mmd": embedding_score,
                "embedding_drift": embedding_score >= mmd_threshold,
                "pixel_mmd": pixel_score,
                "pixel_drift": pixel_score >= pixel_threshold,
                "confidence_drift": confidence_score,
                "confidence_drift_flag": confidence_score >= confidence_threshold,
            }
        )

    return {
        "clean_reference": {
            "samples": clean_total,
            "accuracy": clean_correct / clean_total if clean_total else 0.0,
        },
        "thresholds": {
            "embedding_mmd": mmd_threshold,
            "pixel_mmd": pixel_threshold,
            "confidence_drift": confidence_threshold,
        },
        "groups": rows,
    }


def load_json_records(path: Path) -> list[dict[str, Any]]:
    """Load a JSON list of monitoring records."""
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {path}.")
    return records


def write_scorecard(scorecard: dict[str, Any], output_json: Path, output_csv: Path) -> None:
    """Write deterministic JSON and CSV scorecards."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(scorecard, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "corruption",
        "severity",
        "samples",
        "accuracy",
        "embedding_mmd",
        "embedding_drift",
        "pixel_mmd",
        "pixel_drift",
        "confidence_drift",
        "confidence_drift_flag",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scorecard["groups"])
