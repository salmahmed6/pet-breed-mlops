from torch.utils.data import DataLoader

from .dataset import PetBreedDataset


def create_dataloaders(
    manifest_path: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
):
    train_dataset = PetBreedDataset(
        manifest_path,
        split="train",
        image_size=image_size,
    )

    val_dataset = PetBreedDataset(
        manifest_path,
        split="val",
        image_size=image_size,
    )

    test_dataset = PetBreedDataset(
        manifest_path,
        split="test",
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
