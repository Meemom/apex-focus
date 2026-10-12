# ApexFocus

Multi-teacher, student-rollout distillation for active racing perception. The
planned system selects two crops to improve cone detection with a frozen YOLO11n.
PR 1 provides the installable package and environment verification; dataset,
detector, and training commands are not implemented yet.

## Reproducible setup

Supported targets are Apple Silicon macOS 14+ for local development and x86-64 Linux
on NVIDIA GPU EC2 for CUDA work. Python **3.14.5** and uv **0.13.0** are pinned.
`pyproject.toml` declares dependencies; `uv.lock` fixes the resolved versions and
distribution hashes. The old unpinned `requirements.txt` has been removed.

Install uv into a dedicated bootstrap environment using an existing Python 3:

```bash
python3 -m venv "$HOME/.venvs/apexfocus-tools"
"$HOME/.venvs/apexfocus-tools/bin/python" -m pip install uv==0.13.0
export PATH="$HOME/.venvs/apexfocus-tools/bin:$PATH"
```

From the repository root, use a fresh environment to avoid inheriting the
scaffold's conflicting OpenCV installations:

```bash
uv python install 3.14.5
export UV_PROJECT_ENVIRONMENT=.venv-apexfocus
uv sync --locked
uv run --locked pytest -q
uv run --locked apexfocus-env --device cpu
```

Keep `UV_PROJECT_ENVIRONMENT` set in each shell running project commands.
`uv sync` installs the project and development tests. It also removes packages
outside the lock from its project environment. If an old `.venv` contains both
OpenCV variants, use a **new** environment: uninstalling either variant in place
can remove files shared by the other. To preserve an existing environment:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-clean
uv sync --locked
uv run --locked pytest -q
uv run --locked apexfocus-env --device cpu
```

Keep that variable set for subsequent commands using this environment. Only
`opencv-python` is installed; do not add `opencv-python-headless` alongside it.
No detector weights or datasets are downloaded by the verification command.

### Environment report and expected results

```bash
mkdir -p results/pr1
uv run --locked apexfocus-env --device cpu > results/pr1/environment-cpu.json
```

The JSON records Python/platform, package versions (including optional-package
absence), NVIDIA driver information when available, CUDA availability, PyTorch's
CUDA runtime, cuDNN, GPU names, and operation checks. Success returns exit code 0
and `"ok": true`. It checks a known matrix multiplication, Torchvision NMS, an
OpenCV resize, and the presence of exactly one supported OpenCV distribution.
No NVIDIA driver is expected on macOS. Requesting CUDA on an unavailable device
returns exit code 1 and a diagnostic JSON report; it never silently uses CPU.

## EC2 CUDA setup and verification

Use an **x86-64 Ubuntu GPU instance** with an NVIDIA GPU supported by the locked
PyTorch build, preferably Ampere or newer, and a compatible installed NVIDIA
driver. A current [AWS Deep Learning AMI](https://docs.aws.amazon.com/dlami/)
provides a starting point. Use a separate project virtual environment even if
the AMI includes preinstalled frameworks. Allow disk space for the CUDA wheels,
dataset, and generated caches; size training batches after measuring GPU memory.

The lock uses PyPI wheels: macOS receives its native build; Linux receives the
CUDA-enabled PyTorch distribution and its NVIDIA dependencies. See the
[uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).
The current lock includes a CUDA 13.0 runtime; confirm the actual runtime in the
report against the [NVIDIA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
Do not replace torch or torchvision with an arbitrary pip install: an alternate
CUDA build requires a deliberate dependency/lock change and fresh verification.

On EC2, first verify that the instance exposes its GPU:

```bash
nvidia-smi
```

For a minimal Ubuntu image, the standard OpenCV wheel may need these system
libraries (the project deliberately uses the same OpenCV variant as Ultralytics):

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

Clone the repository and check out the reviewed PR commit, replacing the SHA:

```bash
git clone https://github.com/Meemom/apex-focus.git
cd apex-focus
git checkout <reviewed-commit-sha>
```

Install the pinned uv bootstrap as above, then run:

```bash
uv python install 3.14.5
export UV_PROJECT_ENVIRONMENT=.venv-apexfocus
uv sync --locked
uv run --locked pytest -q
mkdir -p results/pr1
git rev-parse HEAD > results/pr1/commit.txt
uv run --locked apexfocus-env --device cuda > results/pr1/environment-cuda.json
```

**EC2 acceptance:** the command exits 0, `cuda_available` is true,
`requested_device` is `cuda`, GPU names are present, and all four checks pass.
The tensor and Torchvision NMS checks execute on CUDA. Preserve this report with
the commit ID before starting later GPU experiments. A successful local run
does not satisfy this acceptance check.

### Data, artifacts, transfer, and restart

Use `data/raw/` for downloaded online datasets, `data/processed/` for converted
data, `data/manifests/` for small versioned manifests, `checkpoints/` for weights,
`cache/` for generated caches, and `results/` for run outputs. Create these when
needed. Place the checkout on persistent EC2 storage, or explicitly copy artifacts
to persistent storage before terminating an instance. No AWS credentials belong
in the repository.

Prefer downloading datasets directly on EC2 using their official access process.
For development fixtures or returning reports, use a configured SSH host alias
such as `apexfocus-ec2` and adjust the remote project path:

```bash
rsync -av --partial data/raw/ apexfocus-ec2:~/apex-focus/data/raw/
rsync -av --partial apexfocus-ec2:~/apex-focus/results/ results/
```

These commands do not delete destination files. Do not transfer `.venv` between
macOS and Linux; recreate it from the lock. After reconnecting to an instance:

```bash
cd ~/apex-focus
export PATH="$HOME/.venvs/apexfocus-tools/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=.venv-apexfocus
uv sync --locked
uv run --locked apexfocus-env --device cuda
```

If you selected a custom `UV_PROJECT_ENVIRONMENT`, restore it too. Re-running PR 1
verification is safe; training checkpoint/resume commands will be added with the
training PRs rather than claiming resumability before a trainer exists.

## Optional tracking

W&B is absent from the default installation and is never imported by the
environment command. Its JSON stdout can be saved locally with shell redirection;
no account or network logging service is required.

```bash
uv sync --locked --extra tracking
```

When using that extra, include `--extra tracking` in subsequent `uv run` commands
to retain it. Training-specific structured logs and optional W&B integration
will land with training code.

## PR 1 review checklist

- Fresh install: `uv sync --locked` with an unused project environment.
- Tests: `uv run --locked pytest -q`.
- Local smoke: `uv run --locked apexfocus-env --device cpu`.
- EC2 smoke: `uv run --locked apexfocus-env --device cuda`.
- Check the saved JSON and commit ID; flag unperformed checks as pending.
- Scope: no model training, data acquisition, EC2 provisioning, or deployment.

The existing local `.gitignore` preference for untracked Markdown is preserved.
This README is already tracked; future Markdown reports must be explicitly
included when they are ready to commit.

### Implementation verification

Verified on macOS 26.4 arm64 with Python 3.14.5, using a fresh `.venv-pr1`:

- Locked installation succeeded; dependency compatibility check passed.
- All four tests passed, including duplicate-OpenCV rejection and no silent
  CPU fallback when CUDA is requested.
- CPU tensor, Torchvision NMS, and OpenCV operations passed.
- Exactly one OpenCV distribution was installed; W&B was absent.
- Source distribution and wheel built successfully; Ultralytics, Gymnasium,
  and PyYAML imports passed.

The existing `.venv` was preserved. To review the already-tested environment
without installing anything else, run from this checkout:

```bash
.venv-pr1/bin/pytest -q
.venv-pr1/bin/apexfocus-env --device cpu
```

EC2 installation and CUDA execution are **pending**. PR 1 is ready for local
review/testing; its CUDA acceptance criterion is not yet complete.
