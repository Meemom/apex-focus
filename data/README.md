# Pilot dataset: FSOCO

PR 2 prepares original-resolution cone images from the official public FSOCO
bounding-box training archive. It does not train a detector or establish a zoom
benefit. The pilot is development data; it is not the final held-out benchmark.

## Source and access

- [Official download page](https://fsoco.github.io/fsoco-dataset/download): public
  access without a contribution requirement, Supervisely format, approximately
  24 GB for the bounding-box ZIP.
- Archive: `http://fsoco.cs.uni-freiburg.de/datasets/fsoco_bounding_boxes_train.zip`.
  On 2026-10-12, HTTP and byte ranges worked; HTTPS on that host did not.
- [Class/tag definitions and annotation policy](https://fsoco.github.io/fsoco-dataset/labels/).
- [Official schema](https://github.com/fsoco/fsoco-devkit/blob/main/tools/label_converters/meta.json).
- Cite Vödisch, Dodel, and Schötz, *FSOCO: The Formula Student Objects in Context
  Dataset*, as requested by the [dataset authors](https://fsoco.github.io/fsoco-dataset/).

The public access and citation instructions were checked. A separate explicit
dataset redistribution license was not identified on those pages; the devkit's
software license is not evidence of an image license. This repository contains
code, manifests, and audit statistics, not redistributed source images or
annotations. Obtain originals through the official endpoint and retain their
provenance. This records the available terms, not a claim of commercial-use rights.

## Reproduce the pilot

Run from the repository root in the PR 1 Python 3.14.5 environment. If needed,
install uv 0.13.0 using the setup instructions from PR 1. All processing is CPU
only and uses existing dependencies; Requests is now declared directly because
the downloader uses its persistent HTTP connections.

```bash
export UV_PROJECT_ENVIRONMENT=.venv-apexfocus
uv sync --locked --no-editable
uv run --locked --no-editable python -m apexfocus.data.download --output data/raw/fsoco-pilot --count 150
uv run --locked --no-editable python -m apexfocus.data.prepare --config configs/dataset.yaml
uv run --locked --no-editable pytest -q
```

The same commands run locally and on EC2; CUDA is not needed for PR 2. Before
creating the environment, use the pinned Python from `.python-version`.
The review commands use a regular (`--no-editable`) package installation to avoid
the local macOS hidden-`.pth` issue observed with the old PR 1 environment. After
editing source, rebuild it with `uv sync --locked --no-editable --reinstall-package apexfocus`
before testing the installed package.

The downloader reads the ZIP directory and only selected members using byte
ranges. It checks range responses, pins the archive ETag during the transfer,
verifies ZIP member CRCs, and records file SHA256 hashes. Since the upstream
endpoint uses HTTP, these checks detect corruption and subsequent local changes;
they do not authenticate the server. The full archive is never required locally.

The fixed pilot uses 150 images, seeded at 42, with width >=1280 and height >=720,
round-robin across **amz, ecurieaix, frt, ka, mms, tuwr**. Teams were selected to
provide several original image shapes while leaving other source teams unused.
Within each team, candidates are ordered by a seed-dependent SHA256, not by
adjacent frame number. The recorded source paths and hashes are the reproducible
identity of the sample; a changed upstream archive requires a new dataset version.

Both commands refuse to overwrite an existing output. An interrupted download
has no completed `download.json` and must not be prepared. Preserve or remove
that incomplete directory explicitly, then retry with a fresh path. Preparation
validates every source hash before writing and publishes output only after the
entire conversion succeeds. For an intentional new version, change `source_root`
and/or `output` in a copied configuration rather than overwriting frozen results.

## Generated files

| Path | Meaning |
| --- | --- |
| `data/raw/fsoco-pilot/<team>/img/` | Original image bytes |
| `data/raw/fsoco-pilot/<team>/ann/` | Original Supervisely JSON |
| `data/raw/fsoco-pilot/download.json` | URL, ETag, selection settings, and source hashes |
| `data/processed/fsoco-pilot-v1/canonical.jsonl` | Pixel-coordinate boxes, original tags/metadata, identities, hashes, and splits |
| `data/processed/fsoco-pilot-v1/images/<split>/` | Byte-identical image copies with collision-resistant names |
| `data/processed/fsoco-pilot-v1/labels/<split>/` | Normalized YOLO labels; negative images have empty label files |
| `data/processed/fsoco-pilot-v1/dataset.yaml` | Local YOLO dataset configuration |
| `data/processed/fsoco-pilot-v1/manifests/` | Annotation-free all/train/val/test JSONL records |
| `data/processed/fsoco-pilot-v1/audit.json` | Counts, sizes, small-object statistics, duplicate pairs, and provenance limitations |
| `data/processed/fsoco-pilot-v1/overlays/` | One labeled development image per source team |

Only compact manifests and the audit summary belong in `data/manifests/` for Git.
Raw data and generated images, labels, caches, and canonical annotations stay
ignored. The generated YOLO YAML uses an absolute local path; rerun preparation
on EC2 instead of copying that YAML from the Mac unchanged.

Class IDs are fixed:

| ID | Source class |
| --- | --- |
| 0 | blue_cone |
| 1 | yellow_cone |
| 2 | orange_cone |
| 3 | large_orange_cone |
| 4 | unknown_cone |

Unknown classes, non-rectangle geometry, missing images, dimension mismatches,
non-finite coordinates, zero-area boxes, and out-of-bounds coordinates fail the
run. Boxes are not silently clipped or discarded. Image decoding ignores EXIF
rotation so the source annotation coordinates retain their original frame.

## Split and leakage rules

The default pilot reserves one of its six independent teams for validation and
uses the rest for training, unless duplicate clustering connects teams. Images
from the same team never cross splits. Original source `train` tags identify the
public upstream release; they do not override this project's grouped partition.
Session IDs are not invented when only team provenance is available.

The duplicate audit compares original-file hashes, decoded-pixel hashes, and
perceptual hashes. Pairs within the configured Hamming threshold are treated
conservatively: their entire source groups are connected before splitting.
The audit records candidate pairs for review. Zero detected cross-split pairs
does **not** prove all near duplicates have been found. Inspect overlays and
candidate pairs before interpreting evaluation results. Pairwise comparison is
limited to 3,000 images; larger studies need an indexed audit.

The pilot has **no test images**. For a later study with additional source groups:

1. Set `mode: study`, a positive `test_fraction`, and a new output directory.
2. Set `pilot_manifest` to the frozen pilot `all.jsonl` in `data/manifests/`.
3. Supply a completed source download containing new groups, using the downloader's
   `--groups` option and a larger `--count` as appropriate.

Preparation then excludes all pilot teams and any duplicate-linked teams from
test, failing if insufficient unseen groups remain. The pilot exclusion manifest
also screens historical image/pixel/perceptual hashes, including images that
reappear under a different team name. It is mandatory in study mode. Preparation
produces no test overlays. Changing split seeds or
fractions requires a new experiment version, not selective retries based on metrics.

## Small cones, occlusion, and checkpoints

The pilot small-cone threshold is the 25th percentile of normalized box area in
the **training split only**, with `area <= threshold` classified as small.
The audit also reports box pixel-area quantiles and original image resolutions.
This is a pilot convention; freeze the study metric before comparing policies.

Original object and image metadata are retained in the ignored canonical data.
The `truncated` tag covers overlapping cones and cones cut by image borders,
so it is **not an occlusion-only evaluation label**. Before the specialist PR,
define and annotate a validation subset distinguishing overlap occlusion from
border truncation; do not rename all truncated boxes to occluded.

No cone checkpoint was supplied or found in the project during PR 2. The config
records `not_provided`; no checkpoint is certified as free of dataset overlap.
PR 3 must record checkpoint origin, weight hash, class mapping, and training-data
provenance, or train from an auditable starting point. No detector results are
claimed by this data preparation PR.

## Review and tests

```bash
uv run --locked --no-editable pytest -q tests/test_data.py
uv run --locked --no-editable python -m apexfocus.data.download --help
uv run --locked --no-editable python -m apexfocus.data.prepare --help
```

Tests use small generated images and require neither network access nor CUDA.
They verify known coordinate conversions, preservation of empty annotations and
tags, source integrity failures, output immutability, grouped duplicate isolation,
pilot exclusion from test, and refusal of servers that ignore range requests.
Inspect the real pilot's audit and overlays as a separate acceptance check; a
passing fixture test is not evidence of annotation quality in real data.

## Verified pilot result (2026-10-12)

The official archive was downloaded and prepared locally on macOS/Python 3.14.5.
Frozen manifests, audit statistics, and visual-review notes are in
[`manifests/fsoco-pilot-v1/`](manifests/fsoco-pilot-v1/).

| Split | Images | Boxes | Small boxes | Source groups |
| --- | ---: | ---: | ---: | --- |
| Train | 125 | 1,847 | 462 | amz, ecurieaix, frt, ka, tuwr |
| Validation | 25 | 240 | 18 | mms |
| Test | 0 | 0 | 0 | Reserved for the later study |

- All 150 images decoded, matched annotation dimensions, and retained their
  original bytes in the prepared dataset. No invalid boxes or empty images were
  found; empty-image behavior is covered by fixtures.
- All 2,087 YOLO boxes were independently converted back to pixel coordinates;
  maximum rounding error was less than 0.000001 pixels.
- Five perceptual duplicate candidate pairs were detected, all within splits;
  no exact or pixel-identical duplicate pairs were detected in the selected pilot.
- Training-derived small-area threshold: `0.00005463343047166517` of full image area.
- Six team overlays were visually inspected. Original black borders and team
  watermarks are preserved; account for them when choosing the PR 3 crop grid.
- The 25-image validation set has only 18 small cones and no `unknown_cone` labels.
  It supports pipeline/feasibility debugging, not strong performance claims.
- A fresh regular package install and the complete test suite passed. EC2
  execution was not performed; this PR has no GPU-specific code.

In this local checkout, the verified environment is also available directly:

```bash
.venv-pr2/bin/python -m pytest -q
.venv-pr2/bin/python -m apexfocus.data.prepare --help
```

For a repeat conversion without overwriting the frozen version, copy
`configs/dataset.yaml`, change its `output` to a new directory, and run the prepare
command with that copied configuration. The existing source download can be reused.
