"""Original-pixel crop geometry and deterministic merging of detector outputs.

Detections are float64 arrays of [x1, y1, x2, y2, confidence, class_id].
Regions use half-open integer bounds, in row-major order, with no overlap.
"""

import numpy as np

from apexfocus.data.prepare import CLASSES


def detections(rows):
    boxes = np.asarray(rows, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 6), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 6 or not np.isfinite(boxes).all():
        raise ValueError("Detections must be finite Nx6 rows")
    if (np.any(boxes[:, :2] < 0) or np.any(boxes[:, 2:4] <= boxes[:, :2])
            or np.any((boxes[:, 4] < 0) | (boxes[:, 4] > 1))
            or np.any(boxes[:, 5] != boxes[:, 5].astype(int))
            or np.any((boxes[:, 5] < 0) | (boxes[:, 5] >= len(CLASSES)))):
        raise ValueError("Invalid detection coordinates, score, or class")
    return boxes.copy()


def regions(width, height):
    if not isinstance(width, int) or not isinstance(height, int) or width < 3 or height < 2:
        raise ValueError("A 2x3 grid requires integer width >=3 and height >=2")
    xs, ys = [i * width // 3 for i in range(4)], [i * height // 2 for i in range(3)]
    return [(xs[x], ys[y], xs[x + 1], ys[y + 1]) for y in range(2) for x in range(3)]


def extract(image, region):
    x1, y1, x2, y2 = region
    if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
        raise ValueError("Crop bounds exceed image")
    return np.ascontiguousarray(image[y1:y2, x1:x2])


def to_image(rows, region):
    boxes = detections(rows)
    x1, y1, x2, y2 = region
    if np.any(boxes[:, 2] > x2 - x1) or np.any(boxes[:, 3] > y2 - y1):
        raise ValueError("Detector boxes exceed supplied crop dimensions")
    # Ultralytics Results.xyxy already undoes letterboxing; only translate here.
    boxes[:, :4] += [x1, y1, x1, y1]
    return boxes


def iou(box, others):
    others = np.asarray(others).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(box[2:4], others[:, 2:4])
                              - np.maximum(box[:2], others[:, :2])).prod(axis=1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return intersection / (area + areas - intersection)


def ordered(rows):
    boxes = detections(rows)
    # Canonical ties are independent of crop execution order and input row order.
    order = sorted(range(len(boxes)), key=lambda i: (-boxes[i, 4], boxes[i, 5], *boxes[i, :4]))
    return boxes[order]


def merge(sources, nms_iou):
    """Recompute NMS from ALL original pass outputs, never from a previous merge."""
    if not 0 <= nms_iou <= 1:
        raise ValueError("NMS IoU must be in [0, 1]")
    boxes = ordered(np.concatenate([detections(s) for s in sources], axis=0) if sources else [])
    kept = []
    while len(boxes):
        head, rest = boxes[0], boxes[1:]
        kept.append(head)
        suppress = (rest[:, 5] == head[5]) & (iou(head, rest[:, :4]) > nms_iou)
        boxes = rest[~suppress]
    return detections(kept)


def heuristic_action(current, grid, selected, confidence):
    """Zoom where observable low-confidence detections concentrate; tie by grid ID."""
    boxes = detections(current)
    boxes = boxes[boxes[:, 4] >= confidence]
    centers = (boxes[:, :2] + boxes[:, 2:4]) / 2
    scores = []
    for index, (x1, y1, x2, y2) in enumerate(grid):
        inside = ((centers[:, 0] >= x1) & (centers[:, 0] < x2)
                  & (centers[:, 1] >= y1) & (centers[:, 1] < y2))
        scores.append(float((1 - boxes[inside, 4]).sum()) if index not in selected else -np.inf)
    if len(set(selected)) != len(selected) or len(selected) >= len(grid):
        raise ValueError("No valid distinct action remains")
    return int(np.argmax(scores))


def require_device(device):
    import torch

    if device not in ("cpu", "cuda:0"):
        raise ValueError("Use cpu or cuda:0 (one GPU, no implicit fallback)")
    if device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback")


class Detector:
    """Frozen YOLO adapter. Local, provenance-checked weights are supplied by caller."""

    def __init__(self, weights, device, settings):
        from ultralytics import YOLO

        require_device(device)
        self.device, self.settings = device, settings
        self.model = YOLO(str(weights), task="detect")
        if tuple(self.model.names[i] for i in range(len(self.model.names))) != CLASSES:
            raise ValueError("Checkpoint class order differs from audited FSOCO classes")
        self.model.model.eval()
        self.model.model.requires_grad_(False)

    def predict(self, image, size):
        result = self.model.predict(
            source=image, imgsz=size, device=self.device, conf=self.settings["score_floor"],
            iou=self.settings["nms_iou"], max_det=self.settings["max_det"],
            rect=False, half=False, augment=False, agnostic_nms=False,
            verbose=False, save=False,
        )[0]
        boxes = detections(result.boxes.data.detach().cpu().numpy())
        if len(boxes) >= self.settings["max_det"]:
            raise RuntimeError("max_det reached; increase it before comparing methods")
        return to_image(boxes, (0, 0, image.shape[1], image.shape[0]))

    def synchronize(self):
        if self.device == "cuda:0":
            import torch
            torch.cuda.synchronize()
