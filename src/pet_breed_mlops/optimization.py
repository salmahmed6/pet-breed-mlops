from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn.utils import prune


def structured_prune_model(
    model: nn.Module,
    amount: float = 0.2,
) -> tuple[nn.Module, float]:
    """Apply permanent channel pruning to Conv2d outputs using L-norm."""
    if not 0 < amount < 1:
        raise ValueError("amount must be between 0 and 1")
    optimized = copy.deepcopy(model)
    conv_layers = [module for module in optimized.modules() if isinstance(module, nn.Conv2d)]
    for layer in conv_layers:
        prune.ln_structured(layer, name="weight", amount=amount, n=2, dim=0)
        prune.remove(layer, "weight")
    total = sum(layer.weight.numel() for layer in conv_layers)
    zeros = sum(int(torch.count_nonzero(layer.weight == 0).item()) for layer in conv_layers)
    sparsity = zeros / total if total else 0.0
    return optimized, float(sparsity)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Blend hard-label CE with temperature-scaled teacher KL divergence."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    hard = nn.functional.cross_entropy(student_logits, labels)
    soft = nn.functional.kl_div(
        nn.functional.log_softmax(student_logits / temperature, dim=1),
        nn.functional.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature**2)
    return alpha * hard + (1 - alpha) * soft


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def batch_sizes_from_string(value: str) -> list[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("batch sizes must be positive integers")
    return sizes
