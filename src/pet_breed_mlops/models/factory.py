from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
    resnet50,
)


NUM_CLASSES = 37


def create_model(name: str, num_classes: int = NUM_CLASSES):
    name = name.lower()

    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "mobilenet_v3_small":
        model = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT
        )
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes,
        )
        return model

    raise ValueError(f"Unsupported model: {name}")