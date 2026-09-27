from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn, optim

from pet_breed_mlops.data.loaders import create_dataloaders
from pet_breed_mlops.models.factory import create_model
from pet_breed_mlops.training.metrics import calculate_metrics
from pet_breed_mlops.training.seed import set_seed


def _run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    y_true, y_pred = [], []

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    if not y_true:
        raise ValueError("No batches were processed.")
    return total_loss / len(y_true), calculate_metrics(y_true, y_pred)


def train_model(*, model_name: str, manifest_path: str | Path, image_size: int,
                batch_size: int, num_workers: int, epochs: int,
                learning_rate: float, weight_decay: float, seed: int,
                output_dir: Path, max_batches: int | None = None) -> dict[str, Any]:
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be >= 1")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train_loader, val_loader, _ = create_dataloaders(
        manifest_path=str(manifest_path), image_size=image_size,
        batch_size=batch_size, num_workers=num_workers,
    )
    model = create_model(model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{model_name}_best.pt"
    history_path = output_dir / f"{model_name}_history.json"
    best_accuracy, best_macro_f1 = -1.0, -1.0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = _run_epoch(
            model, train_loader, criterion, device, optimizer, max_batches
        )
        val_loss, val_metrics = _run_epoch(
            model, val_loader, criterion, device, None, max_batches
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top_1_accuracy": train_metrics["top_1_accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_top_1_accuracy": val_metrics["top_1_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(record)
        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} "
            f"train_top1={train_metrics['top_1_accuracy']:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} | "
            f"val_loss={val_loss:.4f} val_top1={val_metrics['top_1_accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )

        if val_metrics["top_1_accuracy"] > best_accuracy:
            best_accuracy = val_metrics["top_1_accuracy"]
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save({
                "model_name": model_name, "num_classes": 37,
                "image_size": image_size, "seed": seed, "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_top_1_accuracy": best_accuracy,
                "val_macro_f1": best_macro_f1,
            }, checkpoint_path)

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {
        "model_name": model_name,
        "best_val_top_1_accuracy": best_accuracy,
        "best_val_macro_f1": best_macro_f1,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "device": str(device),
    }
