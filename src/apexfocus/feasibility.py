"""Validation-only crop feasibility comparison and synchronized uncached timing."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
import yaml

from apexfocus.data.prepare import digest, write_json, write_jsonl
from apexfocus.detection import Detector, extract, heuristic_action, merge, regions, to_image
from apexfocus.environment import collect_report
from apexfocus.evaluation import exact_oracle, operating, summarize
from apexfocus.experiment_data import load_development, read_image, verify_checkpoint

METHODS = ("full_640", "full_1280", "random_2", "heuristic_2", "oracle_2")


def validate_config(config):
    settings = config["detector"]
    for key in ("score_floor", "confidence", "nms_iou", "match_iou"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
            raise ValueError(f"{key} must be in (0, 1)")
    if settings["score_floor"] >= settings["confidence"]:
        raise ValueError("AP score floor must be below operating confidence")
    if not math.isfinite(settings["alpha"]) or settings["alpha"] < 0:
        raise ValueError("False-positive penalty must be finite and nonnegative")
    for key, minimum in (("warmup", 1), ("overlays", 0), ("seed", 0)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if type(settings["max_det"]) is not int or settings["max_det"] < 1:
        raise ValueError("max_det must be a positive integer")
    if config["limit"] is not None and (type(config["limit"]) is not int or config["limit"] < 1):
        raise ValueError("limit must be null or a positive integer")
    for key, value in config["gate"].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Gate criterion {key} must be finite and nonnegative")


def selected_run(detector, image, settings, choose):
    """Run coarse + two sequential crops without caching or privileged inputs."""
    grid = regions(image.shape[1], image.shape[0])
    sources = [detector.predict(image, 640)]
    current, selected = merge(sources, settings["nms_iou"]), []
    for _ in range(2):
        action = choose(current, grid, tuple(selected))
        if type(action) is not int or action not in range(6) or action in selected:
            raise ValueError("Selector must choose two distinct region IDs")
        sources.append(to_image(detector.predict(extract(image, grid[action]), 640), grid[action]))
        selected.append(action)
        current = merge(sources, settings["nms_iou"])
    return current, selected


def timed(detector, operation):
    detector.synchronize()
    start = time.perf_counter()
    result = operation()
    detector.synchronize()
    return result, (time.perf_counter() - start) * 1000


def gate_decision(table, config):
    gate = config["gate"]
    coarse, high, oracle = (table[m] for m in ("full_640", "full_1280", "oracle_2"))
    if config["device"] != "cuda:0" or config["limit"] is not None:
        return {"status": "pending", "reason": "Requires the full validation split and CUDA timing"}
    if oracle["small_recall"] is None:
        return {"status": "pending", "reason": "No small ground-truth cones in validation"}
    checks = {
        "small_recall_gain": oracle["small_recall"] - coarse["small_recall"] >= gate["minimum_small_recall_gain"],
        "false_positive_change": oracle["fp_per_image"] - coarse["fp_per_image"] <= gate["maximum_fp_per_image_increase"],
        "positive_utility_headroom": oracle["utility_per_image"] > coarse["utility_per_image"],
        "crop_latency_ceiling": oracle["p95_ms"] <= gate["maximum_crop_p95_ms"],
        "latency_against_1280": oracle["p95_ms"] <= high["p95_ms"] * gate["maximum_crop_to_1280_p95_ratio"],
        "recall_against_1280": oracle["small_recall"] - high["small_recall"] >= gate["minimum_small_recall_gain_over_1280"],
    }
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks,
            "scope": "Pilot headroom only. Oracle replay excludes privileged search and future policy/encoder cost."}


def overlay(path, image, grid, pair, coarse, final, truth, settings):
    _, coarse_rows, coarse_match = operating(coarse, truth, settings["confidence"], settings["match_iou"], settings["alpha"])
    _, final_rows, final_match = operating(final, truth, settings["confidence"], settings["match_iou"], settings["alpha"])
    before, after = set(coarse_match[coarse_match >= 0]), set(final_match[final_match >= 0])
    panels = [image.copy() for _ in range(3)]
    scale = min(1., 720 / image.shape[1])
    thickness = max(2, round(2 / scale))

    def box(panel, coordinates, color):
        x1, y1, x2, y2 = (round(float(v)) for v in coordinates)
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, thickness)

    for rows, matches, panel in ((coarse_rows, coarse_match, panels[0]), (final_rows, final_match, panels[1])):
        for row, matched in zip(rows, matches):
            box(panel, row[:4], (0, 200, 255) if matched >= 0 else (0, 0, 255))
    for index in pair:
        box(panels[1], grid[index], (255, 200, 0))
    for index in sorted(after - before):
        box(panels[2], truth[index]["xyxy"], (0, 255, 0))
    for index in sorted(before - after):
        box(panels[2], truth[index]["xyxy"], (255, 0, 255))
    captions = ["Coarse: TP yellow / FP red", f"Oracle crops {pair}: cyan / FP red",
                "Recovered GT green / lost GT magenta"]
    previews = []
    for panel, caption in zip(panels, captions):
        resized = cv2.resize(panel, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
        resized = cv2.copyMakeBorder(resized, 34, 0, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(resized, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        previews.append(resized)
    if not cv2.imwrite(str(path), np.concatenate(previews, axis=1)):
        raise OSError(f"Cannot write overlay: {path}")
    return {"recovered_gt_indices": sorted(int(i) for i in after - before),
            "lost_gt_indices": sorted(int(i) for i in before - after)}


def markdown_table(table):
    lines = ["| Method | mAP50–95* | Recall | Small recall | Precision | FP/image | Utility/image | Oracle gap | P50 ms | P95 ms |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    keys = ("map50_95", "recall", "small_recall", "precision", "fp_per_image", "utility_per_image", "oracle_gap", "p50_ms", "p95_ms")
    for method in METHODS:
        cells = ["N/A" if table[method][key] is None else f"{table[method][key]:.4f}" for key in keys]
        lines.append("| " + " | ".join([method, *cells]) + " |")
    lines.append("\n*101-point AP over GT-present classes; not COCO area/maxDets AP. Oracle timing is selected-pair replay, excluding GT search.\n")
    return "\n".join(lines)


def run(config):
    validate_config(config)
    output, root = Path(config["output"]), Path(config["dataset"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {output}")
    records, small_threshold, fingerprint = load_development(root)
    provenance = verify_checkpoint(config["weights"], config["provenance"], fingerprint)
    environment = collect_report("cuda" if config["device"] == "cuda:0" else "cpu")
    if not environment["ok"]:
        raise RuntimeError(f"Environment checks failed: {environment['errors']}")
    settings = config["detector"]
    detector = Detector(config["weights"], config["device"], settings)
    validation = records["val"][:config["limit"]]
    output.mkdir(parents=True)
    (output / "overlays").mkdir()
    # Save criteria before any inference; interrupted runs remain visibly incomplete.
    protocol = {"status": "started", "config": config, "dataset_fingerprint": fingerprint,
                "checkpoint": provenance, "environment": environment,
                "small_area_threshold": small_threshold,
                "validation_ids": [r["id"] for r in validation],
                "timing": "Synchronized warm batch-one FP32; includes preprocessing, inference, coordinate mapping, selection and union NMS. Excludes image I/O, evaluation, overlays. Oracle replay excludes GT search. No policy/encoder yet."}
    write_json(output / "protocol.json", protocol)
    warm_image = read_image(root, validation[0])
    for _ in range(config["warmup"]):
        detector.predict(warm_image, 640)
        detector.predict(warm_image, 1280)
        for region in regions(warm_image.shape[1], warm_image.shape[0]):
            detector.predict(extract(warm_image, region), 640)
    samples, timings, details = defaultdict(list), defaultdict(list), []
    for number, record in enumerate(validation):
        image, truth = read_image(root, record), record["boxes"]
        grid = regions(record["width"], record["height"])
        outputs, pairs, elapsed = {}, {}, {}
        for size in (640, 1280):
            outputs[f"full_{size}"], elapsed[f"full_{size}"] = timed(
                detector, lambda: merge([detector.predict(image, size)], settings["nms_iou"]))
        rng = random.Random(int(digest(f"{config['seed']}:{record['id']}".encode()), 16))
        # Sampling and heuristic choice execute inside their measured sequential runs.
        choosers = {
            "random_2": lambda current, grid, selected: rng.choice([i for i in range(6) if i not in selected]),
            "heuristic_2": lambda current, grid, selected: heuristic_action(current, grid, selected, settings["confidence"]),
        }
        for method, choose in choosers.items():
            (outputs[method], pairs[method]), elapsed[method] = timed(
                detector, lambda: selected_run(detector, image, settings, choose))
        candidates = [to_image(detector.predict(extract(image, region), 640), region) for region in grid]
        pair, oracle_output, scores = exact_oracle(outputs["full_640"], candidates, truth, settings)
        (outputs["oracle_2"], pairs["oracle_2"]), elapsed["oracle_2"] = timed(
            detector, lambda: selected_run(detector, image, settings, lambda current, grid, selected: pair[len(selected)]))
        if not np.allclose(outputs["oracle_2"], oracle_output, rtol=1e-5, atol=1e-4):
            raise RuntimeError("Oracle replay disagrees with exhaustive outputs")
        detail = {"id": record["id"], "group": record["group"], "pairs": pairs, "latency_ms": elapsed,
                  "pair_utilities": {f"{a},{b}": value for (a, b), value in scores.items()},
                  "predictions": {method: rows.tolist() for method, rows in outputs.items()}}
        if number < config["overlays"]:
            detail["overlay"] = overlay(output / "overlays" / f"{record['id']}.jpg", image, grid, pair,
                                        outputs["full_640"], outputs["oracle_2"], truth, settings)
        details.append(detail)
        for method in METHODS:
            samples[method].append((record["id"], outputs[method], truth))
            timings[method].append(elapsed[method])
        print(f"Validated {number + 1}/{len(validation)}: {record['id']}", flush=True)
    table = {method: summarize(samples[method], settings, small_threshold) for method in METHODS}
    for method in METHODS:
        table[method].update(p50_ms=float(np.percentile(timings[method], 50)),
                             p95_ms=float(np.percentile(timings[method], 95)),
                             oracle_gap=table["oracle_2"]["utility_per_image"] - table[method]["utility_per_image"])
    decision = gate_decision(table, config)
    write_jsonl(output / "per_image.jsonl", details)
    write_json(output / "summary.json", {"table": table, "gate": decision})
    (output / "table.md").write_text(markdown_table(table) + "\nGate A: " + json.dumps(decision) + "\n")
    protocol["status"] = "complete"
    write_json(output / "protocol.json", protocol)
    return {"output": str(output), "gate": decision}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(yaml.safe_load(args.config.read_text())), indent=2))


if __name__ == "__main__":
    main()
