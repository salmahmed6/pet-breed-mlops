from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pet_breed_mlops.models.factory import create_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Pet Breed checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = create_model(checkpoint["model_name"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        example,
        args.output,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"ONNX model: {args.output}")


if __name__ == "__main__":
    main()
