import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from apexfocus.data.prepare import CLASSES, prepare
from apexfocus.detection import detections
from apexfocus.experiment_data import file_hash, load_development, verify_checkpoint
from apexfocus import feasibility
from test_data import dataset  # Reuse the audited six-group fixture.


def checkpoint(tmp_path, fingerprint):
    weights = tmp_path / "fixture.pt"
    weights.write_bytes(b"not a real model: used only with injected fixture detector")
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"status": "trained", "initialization": "random_yolo11n",
                                     "dataset_fingerprint": fingerprint, "classes": list(CLASSES),
                                     "weight_sha256": file_hash(weights)}))
    return weights, provenance


def test_provenance_binds_weights_and_data(dataset, tmp_path):
    prepare(dataset)
    records, threshold, fingerprint = load_development(Path(dataset["output"]))
    assert len(records["train"]) == 10 and threshold > 0
    weights, provenance = checkpoint(tmp_path, fingerprint)
    verify_checkpoint(weights, provenance, fingerprint)
    with pytest.raises(ValueError, match="provenance"):
        verify_checkpoint(weights, provenance, "changed dataset")
    weights.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="provenance"):
        verify_checkpoint(weights, provenance, fingerprint)


def test_training_rejects_extra_files_and_changed_labels(dataset):
    prepare(dataset)
    root = Path(dataset["output"])
    extra = root / "images/train/unmanifested.png"
    extra.write_bytes(b"unexpected data")
    with pytest.raises(ValueError, match="Unmanifested"):
        load_development(root)
    extra.unlink()
    label = next((root / "labels/train").glob("*.txt"))
    label.write_text("0 0.5 0.5 0.5 0.5\n")
    with pytest.raises(ValueError, match="YOLO labels"):
        load_development(root)


def test_full_runner_fixture_produces_table_all_pairs_and_pending_gate(dataset, tmp_path, monkeypatch):
    prepare(dataset)
    _, _, fingerprint = load_development(Path(dataset["output"]))
    weights, provenance = checkpoint(tmp_path, fingerprint)
    config = yaml.safe_load(Path("configs/experiment.yaml").read_text())
    config.update(dataset=dataset["output"], weights=str(weights), provenance=str(provenance),
                  output=str(tmp_path / "evaluation"), device="cpu", warmup=1, overlays=1)
    class FixtureDetector:
        def __init__(self, *args): pass
        def synchronize(self): pass
        def predict(self, image, size): return detections([])
    monkeypatch.setattr(feasibility, "Detector", FixtureDetector)
    monkeypatch.setattr(feasibility, "collect_report", lambda _: {"ok": True, "fixture": True})
    result = feasibility.run(config)
    output = Path(result["output"])
    assert result["gate"]["status"] == "pending"
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["table"]) == set(feasibility.METHODS)
    assert summary["table"]["full_640"]["images"] == 2
    assert summary["table"]["full_640"]["gt"] == 1
    details = [json.loads(line) for line in (output / "per_image.jsonl").read_text().splitlines()]
    assert all(len(row["pair_utilities"]) == 15 for row in details)
    assert all(row["pairs"]["oracle_2"] == [0, 1] for row in details)
    assert (output / "table.md").is_file()
    assert len(list((output / "overlays").glob("*.jpg"))) == 1
    assert json.loads((output / "protocol.json").read_text())["status"] == "complete"
    with pytest.raises(FileExistsError):
        feasibility.run(config)


def test_overlay_records_both_recovered_and_lost_true_positives(tmp_path):
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    truth = [{"xyxy": [10, 10, 20, 20], "class_id": 0}, {"xyxy": [50, 10, 60, 20], "class_id": 0}]
    settings = {"confidence": .25, "match_iou": .5, "alpha": 1}
    changes = feasibility.overlay(tmp_path / "overlay.jpg", image, feasibility.regions(120, 100), (0, 1),
                                  [[10, 10, 20, 20, .8, 0]], [[50, 10, 60, 20, .8, 0]], truth, settings)
    assert changes == {"recovered_gt_indices": [1], "lost_gt_indices": [0]}
