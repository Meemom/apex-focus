"""Validate an FSOCO download and write canonical records, YOLO labels, and audit."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np
import yaml

from apexfocus.data.splits import assign_splits

CLASSES = ("blue_cone", "yellow_cone", "orange_cone", "large_orange_cone", "unknown_cone")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_path(root, relative):
    relative = Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Source path escapes dataset root: {relative}")
    return path


def validate_annotation(annotation, width, height):
    if not isinstance(annotation, dict):
        raise ValueError("Annotation must be a JSON object")
    if annotation.get("size") != {"width": width, "height": height}:
        raise ValueError("Annotation dimensions do not match original image")
    objects = annotation.get("objects")
    if not isinstance(objects, list):
        raise ValueError("Annotation must contain an objects list (empty is allowed)")
    boxes = []
    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError("Each annotation object must be a JSON object")
        if obj.get("classTitle") not in CLASSES or obj.get("geometryType") != "rectangle":
            raise ValueError(f"Unsupported class or geometry: {obj.get('classTitle')}")
        points = obj.get("points", {}).get("exterior")
        if (not isinstance(points, list) or len(points) != 2
                or any(not isinstance(p, list) or len(p) != 2 for p in points)):
            raise ValueError("Rectangle must have two xy corners")
        coordinates = [value for point in points for value in point]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in coordinates):
            raise ValueError("Box coordinates must be finite numbers")
        x1, y1, x2, y2 = coordinates
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Invalid or out-of-bounds box: {coordinates}")
        if obj.get("points", {}).get("interior"):
            raise ValueError("Rectangle contains unexpected interior geometry")
        boxes.append({"class_id": CLASSES.index(obj["classTitle"]), "xyxy": coordinates,
                      "area_fraction": (x2 - x1) * (y2 - y1) / (width * height),
                      "source_object": obj})
    return boxes


def load_records(root):
    provenance = json.loads((root / "download.json").read_text())
    files = provenance["files"]
    if not files or len(files) != provenance["count"]:
        raise ValueError("Download manifest is incomplete")
    records, seen = [], set()
    for entry in sorted(files, key=lambda item: item["image"]):
        image_path = source_path(root, entry["image"])
        annotation_path = source_path(root, entry["annotation"])
        parts = Path(entry["image"]).parts
        if len(parts) != 3 or parts[1] != "img" or entry["annotation"] != f"{parts[0]}/ann/{parts[2]}.json":
            raise ValueError("Expected matching team/img/name and team/ann/name.json")
        image_bytes, annotation_bytes = image_path.read_bytes(), annotation_path.read_bytes()
        if digest(image_bytes) != entry["image_sha256"] or digest(annotation_bytes) != entry["annotation_sha256"]:
            raise ValueError(f"Source changed after download: {entry['image']}")
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise ValueError(f"Cannot decode {entry['image']}")
        height, width = image.shape[:2]
        annotation = json.loads(annotation_bytes)
        try:
            boxes = validate_annotation(annotation, width, height)
        except ValueError as exc:
            raise ValueError(f"{entry['annotation']}: {exc}") from exc
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        spectrum = cv2.dct(cv2.resize(gray, (32, 32)).astype(np.float32))[:8, :8].flatten()
        bits = spectrum > np.median(spectrum[1:])
        bits[0] = False  # Ignore brightness/DC when comparing image structure.
        phash = sum(int(bit) << i for i, bit in enumerate(bits))
        image_id = digest(entry["image"].encode())[:20]
        if image_id in seen:
            raise ValueError(f"Duplicate source identifier: {entry['image']}")
        seen.add(image_id)
        records.append({"id": image_id, "group": parts[0], "image": entry["image"],
                        "annotation": entry["annotation"], "width": width, "height": height,
                        "image_sha256": entry["image_sha256"],
                        "annotation_sha256": entry["annotation_sha256"],
                        "pixel_sha256": digest(str(image.shape).encode() + image.tobytes()),
                        "phash": f"{phash:016x}", "boxes": boxes,
                        "source_annotation": {k: v for k, v in annotation.items() if k != "objects"}})
    return records, provenance


def yolo_lines(record):
    width, height = record["width"], record["height"]
    lines = []
    for box in record["boxes"]:
        x1, y1, x2, y2 = box["xyxy"]
        coordinates = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
                       (x2 - x1) / width, (y2 - y1) / height)
        lines.append(str(box["class_id"]) + " " + " ".join(f"{v:.10f}" for v in coordinates))
    return "\n".join(lines) + ("\n" if lines else "")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(v, sort_keys=True, allow_nan=False) + "\n" for v in values))


def prepare(config):
    root, output = Path(config["source_root"]), Path(config["output"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; use a new versioned output path")
    mode = config["mode"]
    if mode not in ("pilot", "study"):
        raise ValueError("mode must be pilot or study")
    excluded = set()
    pilot_records = []
    pilot = config.get("pilot_manifest")
    if mode == "study":
        if not pilot:
            raise ValueError("A study requires the prior pilot all.jsonl manifest")
        pilot_records = [json.loads(line) for line in Path(pilot).read_text().splitlines() if line]
        excluded = {r["group"] for r in pilot_records}
        if not excluded:
            raise ValueError("Pilot exclusion manifest is empty")
    test_fraction = config["test_fraction"]
    if mode == "pilot" and test_fraction != 0:
        raise ValueError("Pilot images are development data; pilot test_fraction must be zero")
    if mode == "study" and test_fraction <= 0:
        raise ValueError("A study must reserve non-pilot test groups")
    quantile = config["small_area_quantile"]
    if not isinstance(quantile, (int, float)) or not 0 <= quantile <= 1:
        raise ValueError("small_area_quantile must be between zero and one")
    records, provenance = load_records(root)
    pilot_matches = []
    for record in records:
        for previous in pilot_records:
            distance = (int(record["phash"], 16) ^ int(previous["phash"], 16)).bit_count()
            if (record["image_sha256"] == previous["image_sha256"]
                    or record["pixel_sha256"] == previous["pixel_sha256"]
                    or distance <= config["near_duplicate_hamming"]):
                excluded.add(record["group"])
                pilot_matches.append({"current": record["id"], "pilot": previous["id"], "distance": distance})
    assignments, duplicates = assign_splits(
        records, config["seed"], config["validation_fraction"], test_fraction,
        excluded, config["near_duplicate_hamming"],
    )
    for record in records:
        record["split"] = assignments[record["group"]]
    areas = [b["area_fraction"] for r in records if r["split"] == "train" for b in r["boxes"]]
    if not areas:
        raise ValueError("Training split contains no boxes for small-object threshold estimation")
    threshold = float(np.quantile(areas, quantile))
    summary = {}
    for split in ("train", "val", "test"):
        subset = [r for r in records if r["split"] == split]
        boxes = [b for r in subset for b in r["boxes"]]
        summary[split] = {
            "images": len(subset), "boxes": len(boxes),
            "empty_images": sum(not r["boxes"] for r in subset),
            "groups": sorted({r["group"] for r in subset}),
            "classes": dict(Counter(CLASSES[b["class_id"]] for b in boxes)),
            "resolutions": dict(Counter(f"{r['width']}x{r['height']}" for r in subset)),
            "small_boxes": sum(b["area_fraction"] <= threshold for b in boxes),
            "box_pixel_area_quantiles": [float(v) for v in np.quantile(
                [(b["xyxy"][2] - b["xyxy"][0]) * (b["xyxy"][3] - b["xyxy"][1]) for b in boxes],
                [0, .25, .5, .75, 1])] if boxes else [],
            "tags": dict(Counter(t["name"] for b in boxes for t in b["source_object"].get("tags", []))),
        }
    by_id = {r["id"]: r for r in records}
    cross = [p for p in duplicates if by_id[p["left"]]["split"] != by_id[p["right"]]["split"]]
    if cross:
        raise ValueError("Detected duplicate leakage across splits")
    audit = {
        "schema_version": 1, "config": config, "source": {k: v for k, v in provenance.items() if k != "files"},
        "splits": summary, "small_area_threshold": threshold, "threshold_fit_split": "train",
        "duplicate_pairs": duplicates, "cross_split_duplicate_pairs": len(cross),
        "pilot_matches": pilot_matches,
        "duplicate_method": "file SHA256, decoded-pixel SHA256, 63-bit pHash Hamming candidates",
        "duplicate_limitations": "Perceptual hashing flags candidates; it cannot prove absence of all near duplicates.",
        "occlusion_status": "truncated also includes image-edge truncation; not an occlusion-only label",
        "checkpoint_provenance": config["checkpoint_provenance"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".prepare-", dir=output.parent))
    try:
        (stage / "manifests").mkdir()
        (stage / "overlays").mkdir()
        write_json(stage / "audit.json", audit)
        write_jsonl(stage / "canonical.jsonl", records)
        manifests = [{k: v for k, v in r.items() if k not in ("boxes", "source_annotation")} for r in records]
        write_jsonl(stage / "manifests/all.jsonl", manifests)
        for split in ("train", "val", "test"):
            (stage / "images" / split).mkdir(parents=True)
            (stage / "labels" / split).mkdir(parents=True)
            write_jsonl(stage / "manifests" / f"{split}.jsonl", [r for r in manifests if r["split"] == split])
        overlaid = set()
        for r in records:
            name = r["id"] + Path(r["image"]).suffix.lower()
            shutil.copyfile(source_path(root, r["image"]), stage / "images" / r["split"] / name)
            (stage / "labels" / r["split"] / f"{r['id']}.txt").write_text(yolo_lines(r))
            # Never expose held-out test images in development overlays.
            if r["split"] != "test" and r["group"] not in overlaid:
                overlaid.add(r["group"])
                image = cv2.imread(str(source_path(root, r["image"])), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                for b in r["boxes"]:
                    x1, y1, x2, y2 = [round(v) for v in b["xyxy"]]
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, str(b["class_id"]), (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
                if not cv2.imwrite(str(stage / "overlays" / f"{r['group']}-{r['id']}.jpg"), image):
                    raise OSError("Could not write annotation overlay")
        dataset = {"path": str(output.resolve()), "train": "images/train", "val": "images/val",
                   "names": dict(enumerate(CLASSES))}
        if summary["test"]["images"]:
            dataset["test"] = "images/test"
        (stage / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False))
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    args = parser.parse_args()
    try:
        config = yaml.safe_load(args.config.read_text())
        result = prepare(config)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")
    print(json.dumps({"output": config["output"], "splits": result["splits"]}, indent=2))


if __name__ == "__main__":
    main()
