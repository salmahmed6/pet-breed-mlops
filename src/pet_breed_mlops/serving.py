from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from pet_breed_mlops.labels import load_label_map
from pet_breed_mlops.models.factory import create_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CAT_CLASS_MAX = 11
DEFAULT_IMAGE_SIZE = 224


def build_inference_transform(image_size: int = DEFAULT_IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[torch.nn.Module, str]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model_name = str(checkpoint["model_name"])
    model = create_model(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, model_name


def species_for_class(class_index: int) -> str:
    return "cat" if class_index <= CAT_CLASS_MAX else "dog"


def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    return torch.softmax(logits / temperature, dim=-1)


def predict_top_k(
    logits: torch.Tensor,
    label_map: dict[str, int],
    temperature: float = 1.0,
    abstention_threshold: float | None = None,
    model_version: str = "unknown",
    top_k: int = 3,
) -> dict[str, Any]:
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("Expected logits with shape [1, num_classes].")
    if top_k < 1:
        raise ValueError("top_k must be positive.")

    index_to_breed = {index: breed for breed, index in label_map.items()}
    raw_probabilities = torch.softmax(logits, dim=-1)[0]
    probabilities = temperature_scale(logits, temperature)[0]
    values, indices = torch.topk(probabilities, k=min(top_k, probabilities.numel()))

    predictions = []
    for calibrated_confidence, class_index in zip(values.tolist(), indices.tolist()):
        raw_confidence = float(raw_probabilities[int(class_index)])
        predictions.append(
            {
                "breed": index_to_breed[int(class_index)],
                "species": species_for_class(int(class_index)),
                "class_index": int(class_index),
                "confidence": calibrated_confidence,
                "calibrated_confidence": calibrated_confidence,
                "abstained": (
                    abstention_threshold is not None
                    and calibrated_confidence < abstention_threshold
                ),
                "model_version": model_version,
            }
        )

    return {
        "predictions": predictions,
        "abstained": predictions[0]["abstained"],
        "model_version": model_version,
    }


def predict_image(
    model: torch.nn.Module,
    image_bytes: bytes,
    label_map: dict[str, int],
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    temperature: float = 1.0,
    abstention_threshold: float | None = None,
    model_version: str = "unknown",
    top_k: int = 3,
) -> dict[str, Any]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = build_inference_transform(image_size)(image).unsqueeze(0)

    with torch.inference_mode():
        logits = model(tensor)

    return predict_top_k(
        logits,
        label_map,
        temperature=temperature,
        abstention_threshold=abstention_threshold,
        model_version=model_version,
        top_k=top_k,
    )


def load_serving_settings() -> dict[str, Any]:
    checkpoint = Path(os.getenv("PET_BREED_CHECKPOINT", "artifacts/models/resnet18_best.pt"))
    temperature = float(os.getenv("PET_BREED_TEMPERATURE", "1.0"))
    threshold_raw = os.getenv("PET_BREED_ABSTENTION_THRESHOLD")
    threshold = float(threshold_raw) if threshold_raw else None
    version = os.getenv("PET_BREED_MODEL_VERSION", checkpoint.stem)
    return {
        "checkpoint": checkpoint,
        "temperature": temperature,
        "abstention_threshold": threshold,
        "model_version": version,
    }


def build_bentoml_service():
    """Build the BentoML service lazily so core modules stay importable without BentoML."""
    try:
        import bentoml
    except ImportError as exc:
        raise RuntimeError(
            "BentoML is required to run the serving service. Install project dependencies."
        ) from exc

    settings = load_serving_settings()
    label_map = load_label_map()
    model, _ = load_checkpoint_model(settings["checkpoint"])

    @bentoml.service(
        name="pet-breed-classifier",
        resources={"cpu": "1"},
        traffic={"timeout": 30},
    )
    class PetBreedClassifier:
        def __init__(self) -> None:
            self.model = model

        @bentoml.api
        def predict(self, image: bytes) -> dict[str, Any]:
            return predict_image(
                self.model,
                image,
                label_map,
                temperature=settings["temperature"],
                abstention_threshold=settings["abstention_threshold"],
                model_version=settings["model_version"],
            )

    return PetBreedClassifier
