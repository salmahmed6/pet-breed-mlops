from pathlib import Path

import pytest
import torch

from pet_breed_mlops.data.dataset import PetBreedDataset
from pet_breed_mlops.data.loaders import create_dataloaders
from pet_breed_mlops.models.factory import create_model
from pet_breed_mlops.training.metrics import (
    calculate_metrics,
    confusion_matrix,
)

MANIFEST_PATH = Path("data/processed/manifest.json")

EXPECTED_TRAIN_SIZE = 2944
EXPECTED_VAL_SIZE = 736
EXPECTED_TEST_SIZE = 3669
EXPECTED_NUM_CLASSES = 37


def test_manifest_exists():
    assert MANIFEST_PATH.exists()


@pytest.mark.parametrize(
    ("split", "expected_size"),
    [
        ("train", EXPECTED_TRAIN_SIZE),
        ("val", EXPECTED_VAL_SIZE),
        ("test", EXPECTED_TEST_SIZE),
    ],
)
def test_dataset_split_sizes(split, expected_size):
    dataset = PetBreedDataset(
        manifest_path=MANIFEST_PATH,
        split=split,
        image_size=224,
    )

    assert len(dataset) == expected_size


def test_dataset_returns_image_and_label():
    dataset = PetBreedDataset(
        manifest_path=MANIFEST_PATH,
        split="train",
        image_size=224,
    )

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert isinstance(label, int)
    assert 0 <= label < EXPECTED_NUM_CLASSES


def test_all_splits_are_non_empty():
    train_dataset = PetBreedDataset(
        manifest_path=MANIFEST_PATH,
        split="train",
        image_size=224,
    )

    val_dataset = PetBreedDataset(
        manifest_path=MANIFEST_PATH,
        split="val",
        image_size=224,
    )

    test_dataset = PetBreedDataset(
        manifest_path=MANIFEST_PATH,
        split="test",
        image_size=224,
    )

    assert len(train_dataset) > 0
    assert len(val_dataset) > 0
    assert len(test_dataset) > 0


def test_dataloaders():
    train_loader, val_loader, test_loader = create_dataloaders(
        manifest_path=str(MANIFEST_PATH),
        image_size=224,
        batch_size=4,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    assert images.shape == (4, 3, 224, 224)
    assert labels.shape == (4,)

    assert len(val_loader.dataset) == EXPECTED_VAL_SIZE
    assert len(test_loader.dataset) == EXPECTED_TEST_SIZE


@pytest.mark.parametrize(
    "model_name",
    [
        "resnet18",
        "resnet50",
        "mobilenet_v3_small",
    ],
)
def test_model_factory_output_shape(model_name):
    model = create_model(model_name)
    model.eval()

    inputs = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (2, EXPECTED_NUM_CLASSES)


def test_model_factory_rejects_unknown_model():
    with pytest.raises(ValueError):
        create_model("unknown_model")


def test_metrics():
    y_true = [0, 1, 2, 0]
    y_pred = [0, 1, 0, 0]

    metrics = calculate_metrics(y_true, y_pred)

    assert "top_1_accuracy" in metrics
    assert "macro_f1" in metrics

    assert metrics["top_1_accuracy"] == pytest.approx(0.75)


def test_confusion_matrix():
    y_true = [0, 1, 2, 0]
    y_pred = [0, 1, 0, 0]

    matrix = confusion_matrix(
        y_true,
        y_pred,
        num_classes=3,
    )

    assert matrix.shape == (3, 3)

    assert matrix[0, 0] == 2
    assert matrix[1, 1] == 1
    assert matrix[2, 0] == 1


def test_labels_are_within_37_classes():
    dataset = PetBreedDataset(
        manifest_path=MANIFEST_PATH,
        split="train",
        image_size=224,
    )

    labels = {
        dataset[index][1]
        for index in range(len(dataset))
    }

    assert len(labels) == EXPECTED_NUM_CLASSES
    assert labels == set(range(EXPECTED_NUM_CLASSES))