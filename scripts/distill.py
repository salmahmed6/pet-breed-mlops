from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch import optim

from pet_breed_mlops.data.loaders import create_dataloaders
from pet_breed_mlops.models.factory import create_model
from pet_breed_mlops.optimization import distillation_loss


def main():
    parser = argparse.ArgumentParser(description="Distill a teacher into a smaller student.")
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--student-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    teacher_payload = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    student_payload = torch.load(args.student_checkpoint, map_location="cpu", weights_only=False)
    teacher = create_model(teacher_payload["model_name"])
    student = create_model(student_payload["model_name"])
    teacher.load_state_dict(teacher_payload["model_state_dict"])
    student.load_state_dict(student_payload["model_state_dict"])
    teacher.eval()
    student.train()

    train_loader, _, _ = create_dataloaders(
        config["data"]["manifest"],
        int(config["data"]["image_size"]),
        int(config["training"]["batch_size"]),
        int(config["data"]["num_workers"]),
    )
    optimizer = optim.AdamW(
        student.parameters(),
        lr=float(config["training"]["learning_rate"]),
    )
    for _ in range(args.epochs):
        for batch_index, (images, labels) in enumerate(train_loader):
            if batch_index >= args.max_batches:
                break
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)
            loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=args.temperature,
                alpha=args.alpha,
            )
            loss.backward()
            optimizer.step()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": student_payload["model_name"],
            "model_state_dict": student.state_dict(),
        },
        args.output,
    )
    print(f"Distilled checkpoint: {args.output}")


if __name__ == "__main__":
    main()
