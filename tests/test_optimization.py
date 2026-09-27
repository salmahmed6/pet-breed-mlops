import torch
from torch import nn

from pet_breed_mlops.optimization import batch_sizes_from_string, distillation_loss, structured_prune_model


def test_batch_sizes_parser():
    assert batch_sizes_from_string("1, 8,16") == [1, 8, 16]


def test_structured_pruning_creates_expected_sparsity():
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU())
    pruned, sparsity = structured_prune_model(model, 0.25)
    assert 0.0 < sparsity < 1.0
    assert pruned[0].weight.shape == model[0].weight.shape


def test_distillation_loss_is_finite():
    student = torch.randn(4, 37, requires_grad=True)
    teacher = torch.randn(4, 37)
    labels = torch.tensor([0, 1, 2, 3])
    loss = distillation_loss(student, teacher, labels)
    assert torch.isfinite(loss)
