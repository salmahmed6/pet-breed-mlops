from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from pet_breed_mlops.models.factory import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = create_model(checkpoint["model_name"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        torch_logits = model(sample).numpy()

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(None, {"images": sample.numpy().astype(np.float32)})[0]
    max_error = float(np.max(np.abs(torch_logits - onnx_logits)))
    print(f"Max absolute logit error: {max_error:.8f}")
    if max_error > args.tolerance:
        raise SystemExit(
            f"ONNX agreement failed: {max_error:.8f} > {args.tolerance:.8f}"
        )


if __name__ == "__main__":
    main()
