"""Class-aware confidence-greedy matching, AP, and a privileged exact pair oracle."""

from itertools import combinations

import numpy as np

from apexfocus.detection import detections, iou, merge, ordered


def match(predictions, truth, threshold):
    """Return sorted predictions and matched GT indices (-1 is a false positive)."""
    predictions = ordered(predictions)
    targets = detections([[*b["xyxy"], 1., b["class_id"]] for b in truth])
    matches = np.full(len(predictions), -1, dtype=int)
    available = np.ones(len(targets), dtype=bool)
    for index, box in enumerate(predictions):
        overlaps = iou(box, targets[:, :4])
        candidates = np.flatnonzero(available & (targets[:, 5] == box[5]) & (overlaps >= threshold))
        if len(candidates):
            # Stable ground-truth order breaks equal-IoU ties.
            target = candidates[np.argmax(overlaps[candidates])]
            matches[index] = target
            available[target] = False
    return predictions, matches


def operating(predictions, truth, confidence, match_iou, alpha):
    rows = detections(predictions)
    rows, matched = match(rows[rows[:, 4] >= confidence], truth, match_iou)
    tp = int((matched >= 0).sum())
    fp = len(rows) - tp
    return {"tp": tp, "fp": fp, "utility": tp - alpha * fp}, rows, matched


def exact_oracle(coarse, candidates, truth, settings):
    if len(candidates) != 6:
        raise ValueError("Exact two-crop oracle requires six candidate outputs")
    scores, outputs = {}, {}
    for pair in combinations(range(6), 2):
        output = merge([coarse, *(candidates[i] for i in pair)], settings["nms_iou"])
        score, _, _ = operating(output, truth, settings["confidence"], settings["match_iou"], settings["alpha"])
        scores[pair], outputs[pair] = score["utility"], output
    # combinations is lexicographic: equal-utility pairs choose the smallest IDs.
    best = max(scores, key=scores.get)
    return best, outputs[best], scores


def summarize(samples, settings, small_area_threshold):
    """101-recall-point precision envelope; macro AP over GT-present classes.

    This is our explicitly defined metric, not pycocotools area/maxDets AP.
    Samples are (stable image ID, predictions, canonical ground-truth boxes).
    """
    if not samples:
        raise ValueError("Cannot evaluate an empty split")
    tp = fp = total = small_total = small_tp = 0
    classes = sorted({b["class_id"] for _, _, truth in samples for b in truth})
    for _, predictions, truth in samples:
        counts, _, matched = operating(predictions, truth, settings["confidence"], settings["match_iou"], settings["alpha"])
        tp += counts["tp"]
        fp += counts["fp"]
        total += len(truth)
        small_total += sum(b["area_fraction"] <= small_area_threshold for b in truth)
        small_tp += sum(truth[i]["area_fraction"] <= small_area_threshold for i in matched if i >= 0)
    average_precisions = []
    for threshold in np.linspace(.5, .95, 10):
        for class_id in classes:
            entries, target_count = [], 0
            for image_id, predictions, truth in sorted(samples, key=lambda s: s[0]):
                rows = detections(predictions)
                rows = rows[rows[:, 5] == class_id]
                targets = [b for b in truth if b["class_id"] == class_id]
                rows, matched = match(rows, targets, threshold)
                target_count += len(targets)
                entries.extend((float(row[4]), image_id, index, int(matched[index] >= 0))
                               for index, row in enumerate(rows))
            entries.sort(key=lambda e: (-e[0], e[1], e[2]))
            hits = np.cumsum([entry[3] for entry in entries])
            recall = hits / target_count
            precision = hits / np.arange(1, len(hits) + 1)
            average_precisions.append(float(np.mean([
                precision[recall >= level].max(initial=0.) for level in np.linspace(0., 1., 101)
            ])))
    return {"images": len(samples), "gt": total, "tp": tp, "fp": fp,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / total if total else None,
            "small_gt": small_total, "small_tp": int(small_tp),
            "small_recall": small_tp / small_total if small_total else None,
            "fp_per_image": fp / len(samples),
            "utility_per_image": (tp - settings["alpha"] * fp) / len(samples),
            "map50_95": float(np.mean(average_precisions)) if classes else None,
            "ap_classes": classes}
