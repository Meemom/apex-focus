import json
import subprocess
import sys

import torch

from apexfocus import environment


def test_cpu_cli_runs_without_model_downloads():
    result = subprocess.run(
        [sys.executable, "-m", "apexfocus.environment", "--device", "cpu"],
        capture_output=True, text=True, check=True,
    )
    report = json.loads(result.stdout)
    assert report["ok"]
    assert report["checks"] == {
        "single_opencv_provider": True,
        "tensor_operation": True,
        "torchvision_nms": True,
        "opencv_operation": True,
    }


def test_duplicate_opencv_is_rejected(monkeypatch):
    versions = environment.installed_versions()
    versions["opencv-python-headless"] = "test-conflict"
    monkeypatch.setattr(environment, "installed_versions", lambda: versions)
    report = environment.collect_report()
    assert not report["ok"]
    assert not report["checks"]["single_opencv_provider"]


def test_cuda_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    report = environment.collect_report("cuda")
    assert not report["ok"]
    assert any("no CPU fallback" in error for error in report["errors"])


def test_cli_returns_failure_status(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["apexfocus-env"])
    monkeypatch.setattr(environment, "collect_report", lambda device: {"ok": False})
    assert environment.main() == 1
    assert json.loads(capsys.readouterr().out) == {"ok": False}
