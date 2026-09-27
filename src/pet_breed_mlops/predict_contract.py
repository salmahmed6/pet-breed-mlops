from __future__ import annotations

from typing import TypedDict


class PetBreedPrediction(TypedDict):
    breed: str
    species: str
    class_index: int
    confidence: float
    calibrated_confidence: float
    abstained: bool
    model_version: str
