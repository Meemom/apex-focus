# apex-focus

MOPD-inspired active perception 

Motivated by [DAgger: A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a)

## Pilot dataset (PR 2)

See [data preparation and verification](data/README.md) for official FSOCO access,
the 150-image pilot, reproducible commands, split rules, and audit limitations.

```bash
export UV_PROJECT_ENVIRONMENT=.venv-apexfocus
uv sync --locked --no-editable
uv run --locked --no-editable python -m apexfocus.data.download --output data/raw/fsoco-pilot --count 150
uv run --locked --no-editable python -m apexfocus.data.prepare --config configs/dataset.yaml
uv run --locked --no-editable pytest -q
```

Use Python 3.14.5 and uv 0.13.0. Existing output directories are never overwritten.
The pilot is development-only; future test data must come from previously unused
source groups. No detector training or CUDA execution is required for this PR.

## Crop inference and feasibility (PR 3)

See [the PR 3 verification handoff](docs/PR3.md) for the protocol, tests, EC2 commands,
and current acceptance status. The `crop-inference` branch builds on PR 2 commit
`baa2561`; use `pilot-dataset` as its review base until PR 2 is merged.

On EC2, after installing the locked environment and preparing/transferring the pilot:

```bash
uv run --locked --no-editable apexfocus-env --device cuda
uv run --locked --no-editable python -m apexfocus.train_detector --config configs/detector.yaml
uv run --locked --no-editable python -m apexfocus.feasibility --config configs/experiment.yaml
```

The training command starts YOLO11n from random initialization, trains only on the
125 training images, and selects a checkpoint using the 25 validation images.
The feasibility command verifies its provenance and hash, compares five methods,
and writes metrics, overlays, timings, and a Gate A decision. Existing run directories
are never overwritten. Gate A remains **pending** until a complete CUDA validation
run supplies evidence; unit tests and a CPU smoke run cannot pass it.
