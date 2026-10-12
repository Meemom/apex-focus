"""Report the runtime and verify the selected device without model downloads."""

import argparse
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys


OPENCV_PACKAGES = (
    "opencv-python", "opencv-python-headless",
    "opencv-contrib-python", "opencv-contrib-python-headless",
)
PACKAGES = (
    "apexfocus", "torch", "torchvision", "ultralytics", "gymnasium",
    "numpy", "pyyaml", "wandb", "pytest", *OPENCV_PACKAGES,
)


def installed_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def driver_report():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return {"devices": result.stdout.strip().splitlines(), "error": None}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"devices": [], "error": str(exc)}


def collect_report(device="cpu"):
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": installed_versions(),
        "requested_device": device,
        "nvidia_driver": driver_report(),
        "checks": {},
        "errors": [],
    }
    providers = [name for name in OPENCV_PACKAGES if report["packages"][name]]
    report["checks"]["single_opencv_provider"] = providers == ["opencv-python"]
    if not report["checks"]["single_opencv_provider"]:
        report["errors"].append(f"Expected only opencv-python; found {providers}")
    try:
        import cv2
        import numpy as np
        import torch
        from torchvision.ops import nms

        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda_runtime"] = torch.version.cuda
        report["cudnn_version"] = torch.backends.cudnn.version()
        report["gpus"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if device == "cuda" and not report["cuda_available"]:
            raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback")
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        torch.testing.assert_close(
            values @ values,
            torch.tensor([[7.0, 10.0], [15.0, 22.0]], device=device),
        )
        report["checks"]["tensor_operation"] = True
        boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]], device=device)
        kept = nms(boxes, torch.tensor([0.9, 0.8], device=device), 0.5)
        if kept.tolist() != [0]:
            raise RuntimeError("Torchvision NMS returned an unexpected result")
        report["checks"]["torchvision_nms"] = True
        resized = cv2.resize(np.zeros((4, 4, 3), dtype=np.uint8), (2, 2))
        if resized.shape != (2, 2, 3):
            raise RuntimeError("OpenCV resize returned an unexpected shape")
        report["checks"]["opencv_operation"] = True
        if device == "cuda":
            torch.cuda.synchronize()
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["ok"] = not report["errors"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    report = collect_report(args.device)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
