from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from pet_breed_mlops.data.loaders import create_dataloaders


class PetCalibrationReader(CalibrationDataReader):
    def __init__(self, manifest: str, image_size: int, batch_size: int, num_workers: int, max_batches: int):
        _, self.loader, _ = create_dataloaders(manifest, image_size, batch_size, num_workers)
        self.max_batches = max_batches
        self.iterator = iter(self.loader)

    def get_next(self):
        if self.max_batches <= 0:
            return None
        try:
            images, _ = next(self.iterator)
        except StopIteration:
            return None
        self.max_batches -= 1
        return {"images": images.numpy().astype(np.float32)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--calibration-batches", type=int, default=32)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reader = PetCalibrationReader(
        config["data"]["manifest"],
        int(config["data"]["image_size"]),
        int(config["training"]["batch_size"]),
        int(config["data"]["num_workers"]),
        args.calibration_batches,
    )
    onnx.load(str(args.onnx))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(args.onnx),
        str(args.output),
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    print(f"INT8 PTQ model: {args.output}")


if __name__ == "__main__":
    main()
