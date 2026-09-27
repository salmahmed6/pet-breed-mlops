from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Build a TensorRT FP16 engine when trtexec is available."
    )
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    trtexec = shutil.which("trtexec")
    result = {
        "supported": trtexec is not None,
        "onnx": str(args.onnx),
        "engine": str(args.output),
    }
    if trtexec is None:
        result["status"] = "skipped"
        result["reason"] = "trtexec is not available in this environment"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                trtexec,
                f"--onnx={args.onnx}",
                f"--saveEngine={args.output}",
                "--fp16",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        result["status"] = "success" if completed.returncode == 0 else "failed"
        result["returncode"] = completed.returncode
        result["stdout_tail"] = completed.stdout[-4000:]
        result["stderr_tail"] = completed.stderr[-4000:]
    report = Path("reports/optimization/tensorrt_fp16.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
