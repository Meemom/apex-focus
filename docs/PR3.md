# PR 3: crop inference and validation feasibility

## Outcome and status

Branch: `crop-inference`, based on PR 2 commit `baa2561`. Review against
`pilot-dataset` until that dependency is merged into `main`.

Implements original-pixel crop extraction, frozen detector inference, deterministic
merging, auditable detector training, five-method validation evaluation, overlays,
uncached timing, and a configured feasibility gate. No policy training is included.

**Gate A: pending.** A trained baseline and the complete CUDA validation experiment
must establish useful headroom before PR 4. Local tests do not establish that zoom
improves detection. EC2 access was not supplied during implementation.

## Fixed protocol

- Six half-open regions in row-major order. Boundaries are `floor(i*width/3)` and
  `floor(j*height/2)`, including the final width/height edge. No overlap, padding,
  or discarded remainder pixels; exactly two distinct crops, no STOP.
- Extract original-resolution BGR pixels. Ultralytics letterboxes each supplied
  image/crop to square 640 (1280 for the high-resolution baseline), FP32,
  `rect=False`, no test-time augmentation. `Results.boxes.xyxy` already refers to
  supplied pixels: crop mapping adds only its top-left offset.
- Each detector pass uses class-aware NMS at IoU 0.5 and score floor 0.001.
  Preserve every post-NMS source-pass prediction. Recompute final class-aware NMS
  over the union of original coarse and selected-crop predictions after each action.
  Suppress IoU strictly greater than 0.5. Ties sort by descending confidence, then
  class and coordinates. Final merging is independent of action order; this does
  not claim bitwise-identical detector outputs across hardware versions.
- Utility is `TP - FP`, using confidence >=0.25 and class-aware matching IoU >=0.5.
  Greedily match confidence-sorted predictions to the highest-IoU unmatched GT of
  the same class; equal overlaps break by annotation order. Duplicate predictions
  count as FP. Lost true positives therefore reduce utility.
- The exact oracle evaluates all 15 pairs and maximizes final utility. Equal scores
  choose the lexicographically first pair. It is privileged and only an upper bound
  for this candidate set, two-crop budget, detector, and utility. It is not an upper
  bound on recall, AP, full-image 1280, or even the coarse-only score if every pair hurts.
- Random chooses uniformly without replacement, using a seed derived from experiment
  seed + image ID. The heuristic sums `1-confidence` for currently observable
  detections above the operating threshold whose centres fall inside each unselected
  region; chooses the greatest sum, ties by region ID. It recomputes after the first
  crop and never sees GT or unselected candidate predictions. Empty regions tie at zero.
- AP uses confidence floor 0.001, ten IoUs 0.50–0.95, and the 101-point precision
  envelope. Macro-average only classes present in validation ground truth. This is
  an explicitly defined metric, not pycocotools area-range/maxDets AP. Recall and
  precision use the separate 0.25 operating point. Small recall counts matched small
  GT using the PR 2 training-fitted normalized area threshold. Also report FP/image,
  utility/image, and gap to the two-crop oracle. Empty denominators are `null` where
  recall/AP is undefined; precision with no predictions is zero.
- Timing is warm, uncached, batch-one, synchronized at both ends. It includes image
  preprocessing, detector passes, coordinate mapping, selection, and union NMS;
  excludes disk I/O, metrics, and overlays. Both crop passes execute sequentially.
  Oracle latency replays its already selected pair; ground-truth search is excluded.
  This measures an optimistic crop-execution cost, not deployable oracle latency.
  Future encoder and policy overhead must be added in PR 9.

Gate criteria are frozen in `configs/experiment.yaml`: at least 0.10 absolute
small-recall gain over 640; FP increase <=0.5/image; positive utility headroom;
oracle replay P95 <=100ms and <=1.25 times full-image 1280 P95; small recall at least
as good as 1280. CPU runs or runs with an explicit image limit always report pending.
These are exploratory pilot criteria, not a statistical significance claim.

## Setup and local verification

Use one existing environment and the committed lockfile, not a new environment per
PR. The documented standard name is `.venv-apexfocus`; if already using `.venv-pr2`,
set `UV_PROJECT_ENVIRONMENT=.venv-pr2` instead. A normal installation avoids the
editable `.pth` issue previously encountered on this Mac.

```bash
export UV_PROJECT_ENVIRONMENT=.venv-apexfocus
uv sync --locked --no-editable --reinstall-package apexfocus
uv run --locked --no-editable pytest -q tests/test_detection.py tests/test_feasibility.py
uv run --locked --no-editable pytest -q
```

Tests cover odd image sizes, border coverage, crop offsets, invalid detections,
class-aware matching/NMS, tied scores, action order, intermediate suppression,
negative utility changes, all 15 pairs, AP, sequential execution, provenance,
changed labels, extra training images, output preservation, overlays, and gate rules.
The fixture integration test produces all five method rows without downloading weights.

## EC2 training and experiment

Use the locked Python 3.14.5 / uv 0.13.0 environment on a Linux x86_64 NVIDIA instance.
Transfer the prepared dataset directory (including canonical records, manifests,
images, labels, and audit) or rerun PR 2 there. Training regenerates absolute dataset
paths for the current machine, so the Mac-generated `dataset.yaml` need not be edited.
Check available GPU memory before choosing the batch size in `configs/detector.yaml`;
the initial setting is 8 and inference is always batch one. No SSH credentials or
infrastructure are embedded in the repository.

```bash
uv run --locked --no-editable apexfocus-env --device cuda
uv run --locked --no-editable python -m apexfocus.train_detector --config configs/detector.yaml
uv run --locked --no-editable python -m apexfocus.feasibility --config configs/experiment.yaml
```

Training starts from the installed `yolo11n.yaml` architecture with `pretrained=False`,
100 epochs, seed 42, AdamW, no AMP, deterministic operations, no hue shift or flips.
Other augmentation/training defaults come from the pinned Ultralytics release; all
resolved arguments are saved in provenance. No third-party cone checkpoint is accepted
by this PR. That deliberately avoids unknown training overlap. A tiny pilot trained
from scratch may be inadequate: inspect the baseline before interpreting a failed gate.

Every train/validation image hash and YOLO label is verified against canonical data;
unmanifested images/labels are rejected. Provenance binds the image and annotation
fingerprint, class order, package versions, configuration, and final weight SHA256.
The dataset fingerprint must match during evaluation. Validation selects the best
checkpoint; its use here is declared development tuning. The runner exposes no test
split option. Do not turn pilot images into future test data.

Training output: `runs/detector-pilot-v1/fit/weights/best.pt`, Ultralytics loss/metric
logs, and `runs/detector-pilot-v1/provenance.json`. The provenance changes from
`started` to `trained` only after checkpoint hashing. Evaluation verifies it before
loading weights. Model parameters are frozen for all five inference methods.

Evaluation output: `results/feasibility-pilot-v1/`:

| File | Purpose |
| --- | --- |
| `protocol.json` | Full configuration, thresholds, dataset/weight identity, hardware/software; `started` until successful completion |
| `summary.json` | Metrics for all five methods and explicit Gate A checks |
| `table.md` | Compact validation comparison |
| `per_image.jsonl` | All 15 pair utilities, selected pairs, predictions, and per-method timings |
| `overlays/*.jpg` | Coarse/final detections, selected regions, FP, recovered and lost matched GT |

The runner refuses existing output directories. After a failure, inspect its logs and
use a new versioned output path for retry; it does not silently resume or overwrite.
Copy the completed compact table and gate decision into the tracked PR report after
EC2 acceptance, retaining large artifacts outside Git.

For a small EC2 smoke run, copy the experiment config, set `limit: 2` and a fresh
output directory, then run with `--config PATH`. Expected: five rows, 15 pairs per
image, overlays, finite timings, and **Gate A pending**. Then run the unchanged full
configuration for experiment acceptance.

## Evidence and limitations

Local verification results are recorded in `reports/pr3-validation.json`.
Real detector training/inference smoke evidence is separate from experimental claims.
The installed package passed **44 tests**. A one-epoch CPU training smoke used all
125 training images, then all five comparisons ran on all 25 validation images
(375 pair evaluations). One generated overlay was visually checked; nonempty
recovery/loss overlays and coordinate transformations are covered by fixtures.

This deliberately undertrained checkpoint produced no detections. Every method's
AP, recall, small recall, precision, FP/image, and utility was zero. These are genuine
smoke results, **not evidence about whether zoom is useful**. The timing table below
is Apple M5 CPU timing with empty detection outputs, not EC2 deployment performance:

| Method | P50 ms | P95 ms |
| --- | ---: | ---: |
| Full-image 640 | 24.73 | 29.87 |
| Full-image 1280 | 94.63 | 104.96 |
| Random two crops | 74.39 | 79.17 |
| Heuristic two crops | 74.49 | 81.95 |
| Oracle selected-pair replay | 74.87 | 87.75 |

The JSON report contains the full five-method metrics table, exact smoke configurations,
checkpoint hash, package versions, and dataset fingerprint. To repeat the smoke,
copy `configs/detector.yaml`, set `device: cpu`, `epochs: 1`, `batch: 4`, and a new
output directory; run the training CLI with that config. Copy `configs/experiment.yaml`,
set `device: cpu`, point weights/provenance to that run, and choose another new output
directory. Run the feasibility CLI. Expect zero or very poor detection quality and
Gate A pending; numerical timings and training results can vary between hardware.

The pilot has only one validation team, 25 validation images, and 18 small cones;
`unknown_cone` is absent in validation. No uncertainty or generalization claim is
justified by this sample. CUDA training, timing, and Gate A remain required.

## Source checks

Coordinate handling was checked against the installed Ultralytics 8.4.175
`models/yolo/detect/predict.py::construct_result` and the official
[prediction documentation](https://docs.ultralytics.com/modes/predict/).
Random initialization uses the documented
[training option `pretrained=False`](https://docs.ultralytics.com/modes/train/).
