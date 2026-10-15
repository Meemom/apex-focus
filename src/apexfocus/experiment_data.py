"""Verify prepared development data and bind detector provenance to its contents."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from apexfocus.data.prepare import CLASSES, digest, source_path, yolo_lines
from apexfocus.detection import detections


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_image(root, record):
    name = record["id"] + Path(record["image"]).suffix.lower()
    path = source_path(root, f"images/{record['split']}/{name}")
    data = path.read_bytes()
    if digest(data) != record["image_sha256"]:
        raise ValueError(f"Prepared image changed: {path}")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None or image.shape[:2] != (record["height"], record["width"]):
        raise ValueError(f"Image dimensions differ from manifest: {path}")
    return image


def load_development(root):
    """Read only train/val records. Never load test images or labels."""
    root = Path(root)
    records = [json.loads(line) for line in (root / "canonical.jsonl").read_text().splitlines() if line]
    audit = json.loads((root / "audit.json").read_text())
    threshold = audit["small_area_threshold"]
    if audit["threshold_fit_split"] != "train" or not 0 < threshold <= 1:
        raise ValueError("Small-object threshold must be fitted on training data")
    split_records = {}
    for split in ("train", "val"):
        manifest = [json.loads(line) for line in (root / f"manifests/{split}.jsonl").read_text().splitlines() if line]
        subset = sorted([r for r in records if r["split"] == split], key=lambda r: r["id"])
        by_id = {r["id"]: r for r in subset}
        if not subset or len(by_id) != len(subset) or len(manifest) != len(subset):
            raise ValueError(f"Empty, duplicate or inconsistent {split} records")
        for entry in manifest:
            record = by_id.get(entry["id"])
            if record is None or entry != {k: v for k, v in record.items() if k not in ("boxes", "source_annotation")}:
                raise ValueError("Canonical data disagrees with split manifest")
        if len({m["id"] for m in manifest}) != len(manifest):
            raise ValueError("Duplicate manifest entries")
        expected_images = {r["id"] + Path(r["image"]).suffix.lower() for r in subset}
        expected_labels = {r["id"] + ".txt" for r in subset}
        if ({p.name for p in (root / "images" / split).iterdir()} != expected_images
                or {p.name for p in (root / "labels" / split).iterdir()} != expected_labels):
            raise ValueError("Unmanifested or missing images/labels in development split")
        for record in subset:
            read_image(root, record)
            boxes = detections([[*b["xyxy"], 1., b["class_id"]] for b in record["boxes"]])
            if np.any(boxes[:, 2] > record["width"]) or np.any(boxes[:, 3] > record["height"]):
                raise ValueError("Ground truth outside image")
            for box in record["boxes"]:
                x1, y1, x2, y2 = box["xyxy"]
                area = (x2 - x1) * (y2 - y1) / (record["width"] * record["height"])
                if not np.isclose(area, box["area_fraction"], rtol=1e-12, atol=0):
                    raise ValueError("Ground-truth area disagrees with coordinates")
            label = source_path(root, f"labels/{split}/{record['id']}.txt")
            if label.read_text() != yolo_lines(record):
                raise ValueError(f"YOLO labels differ from canonical annotations: {label}")
        split_records[split] = subset
    training_areas = [b["area_fraction"] for r in split_records["train"] for b in r["boxes"]]
    fitted = float(np.quantile(training_areas, audit["config"]["small_area_quantile"]))
    if not np.isclose(fitted, threshold, rtol=1e-12, atol=0):
        raise ValueError("Small-object threshold differs from training annotations")
    for field in ("id", "group", "image_sha256", "pixel_sha256"):
        if {r[field] for r in split_records["train"]} & {r[field] for r in split_records["val"]}:
            raise ValueError(f"Train/validation overlap in {field}")
    # Include annotations as well as image identities, excluding path-specific audit config.
    fingerprint = digest(json.dumps(split_records, sort_keys=True, allow_nan=False).encode())
    return split_records, threshold, fingerprint


def verify_checkpoint(weights, provenance_path, dataset_fingerprint):
    weights = Path(weights)
    provenance = json.loads(Path(provenance_path).read_text())
    if (provenance.get("status") != "trained"
            or provenance.get("initialization") != "random_yolo11n"
            or provenance.get("dataset_fingerprint") != dataset_fingerprint
            or provenance.get("classes") != list(CLASSES)
            or provenance.get("weight_sha256") != file_hash(weights)):
        raise ValueError("Checkpoint hash, class mapping, or training provenance is incompatible")
    return provenance
