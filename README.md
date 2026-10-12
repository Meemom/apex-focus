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
