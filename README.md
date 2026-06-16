# Secondsight

[![CI](https://github.com/vardhjain/Secondsight/actions/workflows/ci.yml/badge.svg)](https://github.com/vardhjain/Secondsight/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

A production-grade, configurable **Person Re-Identification (Re-ID)** system for the
[Market-1501](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html) dataset.

It implements a **ResNet-50 + BNNeck strong baseline** (Luo et al., *Bag of Tricks*,
CVPRW 2019) trained with:

- an identity-balanced **PK sampler** (`batch_size=64`, `num_instances=4` → P=16, K=4),
- **GeM pooling** and `last_stride=1` in ResNet `layer4`,
- **label-smoothed cross-entropy** + **batch-hard triplet** + optional **center loss**,
- **LR warmup** followed by a multistep decay schedule,
- **AMP** mixed-precision training,
- L2-normalized (cosine) features at evaluation with horizontal-flip **TTA**,
- **k-reciprocal re-ranking** (Zhong et al., CVPR 2017),
- **Grad-CAM** explainability and ranked-result visualizations.

The package is intentionally layered so that importing the top-level `reid` package
never pulls in heavy optional dependencies (`torchvision`, `cv2`, `gradio`,
`kagglehub`) — those are imported lazily by the submodules that need them.

## Project layout

```
src/reid/          # the importable library (src-layout)
  config.py        # typed, YAML-backed configuration dataclasses
  data/            # Market-1501 dataset, PK sampler, transforms
  losses/          # triplet, label-smooth CE, center, combined ReIDLoss
  models/          # ResNet-50 backbone, GeM pooling, BNNeck ReIDModel
  evaluation/      # CMC/mAP metrics, k-reciprocal re-ranking, evaluator
  engine/          # trainer and LR scheduler
  utils/           # distance, checkpoint, meters, logging, reproducibility
  visualization/   # Grad-CAM, ranking galleries, analysis plots
scripts/           # download_data, train, evaluate, visualize CLIs
app/               # gradio_app.py interactive probe-vs-gallery demo
configs/           # default.yaml and market1501_strong_baseline.yaml
tests/             # pytest suite (light tests + torchvision-gated heavy tests)
```

## Installation

The project uses a `src`-layout and is built with `hatchling`. With
[`uv`](https://github.com/astral-sh/uv):

```bash
make install          # uv sync --extra dev
```

or with plain pip:

```bash
pip install -e ".[dev]"
```

## Dataset

Download Market-1501 via `kagglehub`:

```bash
python scripts/download_data.py
```

Point the pipeline at the dataset root (the directory containing
`bounding_box_train/`, `query/`, and `bounding_box_test/`) either with the
`--data-root` flag or the `REID_DATA_ROOT` environment variable (see
`.env.example`).

## Usage

Common workflows are wrapped by the `Makefile` (override `CONFIG`, `DATA_ROOT`,
`OUTPUT_DIR`, `DEVICE`, `WEIGHTS` as needed):

```bash
make train   # python scripts/train.py    --config configs/market1501_strong_baseline.yaml ...
make eval    # python scripts/evaluate.py --config ... --weights outputs/best.pth ...
make demo    # python app/gradio_app.py   --config ... --weights outputs/best.pth
```

Or invoke the CLIs directly:

```bash
python scripts/train.py    --config configs/default.yaml --data-root /path/to/Market-1501-v15.09.15
python scripts/evaluate.py --config configs/default.yaml --weights outputs/best.pth
python scripts/visualize.py --help
python app/gradio_app.py   --weights outputs/best.pth
```

The installed console entry points (`reid-download`, `reid-train`,
`reid-evaluate`, `reid-visualize`) mirror these scripts.

## Configuration

Configuration is fully typed (`reid.config`) and round-trips to YAML.
`configs/default.yaml` mirrors the dataclass defaults exactly; the headline
recipe lives in `configs/market1501_strong_baseline.yaml`.

## Results

Measured on Market-1501 from a single training run (ResNet-50 + BNNeck strong
baseline, seed 42, 60 epochs, ~40 min on a Colab T4). The right-hand column lists
the figures reported for the same strong baseline by Luo et al. (*Bag of Tricks*,
CVPRW 2019) — this run lands right on it.

| Setting                   |  mAP   | Rank-1 | Rank-5 | Rank-10 | Reference (Luo et al., 2019) |
| ------------------------- | :----: | :----: | :----: | :-----: | :--------------------------: |
| Cosine + flip-TTA         | 85.04% | 94.21% | 98.25% | 98.90%  |     ~85.9 mAP / ~94.5 R-1    |
| + k-reciprocal re-ranking | 93.66% | 94.66% | 97.57% | 98.28%  |     ~94.2 mAP / ~95.4 R-1    |

> Single-run numbers (no seed averaging). k-reciprocal re-ranking trades a little
> deep-rank recall (Rank-5/10) for a large **+8.6 mAP** / Rank-1 gain, as expected.
> See [Reproducing the results](#reproducing-the-results) to regenerate them.

### Reproducing the results

Training takes roughly 40 minutes on a single modern GPU.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vardhjain/Secondsight/blob/main/notebooks/train_colab.ipynb)

Or run it locally, end to end:

```bash
make download                        # fetch Market-1501 via kagglehub
make train DEVICE=cuda               # train the strong baseline -> outputs/best.pth
make eval  WEIGHTS=outputs/best.pth  # CMC/mAP, then k-reciprocal re-ranking
make demo  WEIGHTS=outputs/best.pth  # interactive probe-vs-gallery Gradio demo
```

## Development

```bash
make lint          # ruff check .
make format        # ruff format . && ruff check --fix .
make format-check  # ruff format --check . && ruff check .
make typecheck     # mypy
make test          # pytest
make test-cov      # pytest with coverage
```

Tests are split into **light** tests (metrics, re-ranking, distance, config,
sampler, losses — need only `numpy`/`torch`/`Pillow`) and **heavy** tests
(model and the transform pipeline) that are guarded with `pytest.importorskip`
and skip cleanly when `torchvision` is not installed.

## Docker

```bash
make docker-build     # docker build -t secondsight .
make docker-run       # runs the Gradio demo on :7860, mounts ./outputs
```

## Model card

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for intended use, training data,
the evaluation protocol, limitations, and ethical considerations.

## Citation

If you use this software, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff).

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
