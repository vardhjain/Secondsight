<div align="center">

# 🔍 Secondsight

### Cross-camera person re-identification on Market-1501

Give it one cropped photo of a person and it searches a gallery of other photos, ranking every
candidate by how likely it is to be that same individual seen again on a different, non-overlapping
camera. Secondsight runs a production-grade **ResNet-50 + BNNeck** pipeline that reaches
**85.0 mAP / 94.2 Rank-1**, rising to **93.7 mAP** after k-reciprocal re-ranking.

**[▶ Open in Colab](https://colab.research.google.com/github/vardhjain/Secondsight/blob/main/notebooks/train_colab.ipynb)** &nbsp;·&nbsp; **[📊 Results](#results)** &nbsp;·&nbsp; **[🧠 Model card](docs/MODEL_CARD.md)**

[![CI](https://github.com/vardhjain/Secondsight/actions/workflows/ci.yml/badge.svg)](https://github.com/vardhjain/Secondsight/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

</div>

> [!NOTE]
> Person re-identification is dual-use and surveillance-adjacent. Please read the ethics and
> limitations section of the [model card](docs/MODEL_CARD.md) before using this work beyond
> research and benchmarking.

---

Under the hood Secondsight reproduces the well-known **ResNet-50 + BNNeck "strong baseline"** (Luo
et al., *Bag of Tricks*, CVPRW 2019) and wraps it in the kind of engineering you would expect from a
real project rather than a one-off research script. The network learns a 2048-dimensional embedding,
which is a compact numeric fingerprint, for each person crop, and it judges two crops to be the same
person when their embeddings sit close together under cosine distance.

The training recipe follows the modern Re-ID playbook. Every batch is drawn by an identity-balanced
**PK sampler** that picks 16 identities with 4 images each, for a batch of 64, which guarantees that
each batch holds enough same-person pairs for **batch-hard triplet mining** to actually work. The
ResNet-50 backbone keeps a higher-resolution feature map by using a stride of 1 in its final stage
(`last_stride=1`) and pools it with **GeM (generalized-mean) pooling**. Training blends three
complementary objectives, namely **label-smoothed cross-entropy**, a **batch-hard triplet loss**,
and an optional **center loss**. A short **learning-rate warmup** flows into a multistep decay
schedule, and **automatic mixed precision (AMP)** keeps the run fast and light on memory. At test
time the features are L2-normalized so that Euclidean distance becomes cosine similarity, each image
is averaged with its horizontal flip for a small **test-time augmentation (TTA)** gain, and an
optional **k-reciprocal re-ranking** pass (Zhong et al., CVPR 2017) pushes accuracy higher still. So
the model is not a black box, the repo also produces **Grad-CAM** attention maps and ranked-result
galleries that show what the model focuses on and where it succeeds or fails.

The package is deliberately layered to stay lightweight on import. Heavy optional dependencies such
as `torchvision`, `cv2`, `gradio`, and `kagglehub` load lazily inside only the submodules that
actually need them, so a plain `import reid` never pulls them in.

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
tests/             # pytest suite (light tests plus torchvision-gated heavy tests)
```

## Installation

The project uses a `src` layout and builds with `hatchling`. The quickest setup uses
[`uv`](https://github.com/astral-sh/uv), which creates the environment and installs the package
together with its development tools in a single step.

```bash
make install          # uv sync --extra dev
```

If you prefer plain pip, install the package in editable mode with its dev extras.

```bash
pip install -e ".[dev]"
```

## Dataset

Download Market-1501 with the bundled helper, which fetches it from Kaggle through `kagglehub`.

```bash
python scripts/download_data.py
```

Then point the pipeline at the dataset root, meaning the directory that holds `bounding_box_train/`,
`query/`, and `bounding_box_test/`. You can supply it with the `--data-root` flag or through the
`REID_DATA_ROOT` environment variable, which is documented in `.env.example`.

## Usage

The `Makefile` wraps the common workflows, and you can override `CONFIG`, `DATA_ROOT`, `OUTPUT_DIR`,
`DEVICE`, and `WEIGHTS` on the command line as needed.

```bash
make train   # python scripts/train.py    --config configs/market1501_strong_baseline.yaml ...
make eval    # python scripts/evaluate.py --config ... --weights outputs/best.pth ...
make demo    # python app/gradio_app.py   --config ... --weights outputs/best.pth
```

You can also call the CLIs directly.

```bash
python scripts/train.py    --config configs/default.yaml --data-root /path/to/Market-1501-v15.09.15
python scripts/evaluate.py --config configs/default.yaml --weights outputs/best.pth
python scripts/visualize.py --help
python app/gradio_app.py   --weights outputs/best.pth
```

After installation the console entry points `reid-download`, `reid-train`, `reid-evaluate`, and
`reid-visualize` mirror these same scripts.

## Configuration

Configuration is fully typed through `reid.config` and round-trips cleanly to and from YAML. The
file `configs/default.yaml` mirrors the dataclass defaults exactly, while the headline recipe behind
the results below lives in `configs/market1501_strong_baseline.yaml`.

## Results

These numbers were measured on Market-1501 from a single training run of the ResNet-50 + BNNeck
strong baseline (seed 42, 60 epochs, roughly 40 minutes on a Colab T4 GPU). The right-hand column
shows the figures Luo et al. (2019) reported for the same strong baseline, and this run lands right
on them.

| Setting                   |  mAP   | Rank-1 | Rank-5 | Rank-10 | Reference (Luo et al., 2019) |
| ------------------------- | :----: | :----: | :----: | :-----: | :--------------------------: |
| Cosine + flip-TTA         | 85.04% | 94.21% | 98.25% | 98.90%  |     ~85.9 mAP / ~94.5 R-1    |
| + k-reciprocal re-ranking | 93.66% | 94.66% | 97.57% | 98.28%  |     ~94.2 mAP / ~95.4 R-1    |

Running `reid-evaluate` on the trained checkpoint prints the same scorecard to the terminal.

```text
------------------------------------------------------
Setting              mAP    Rank-1    Rank-5   Rank-10
------------------------------------------------------
Baseline         85.04%    94.21%    98.25%    98.90%
Re-ranked        93.66%    94.66%    97.57%    98.28%
------------------------------------------------------
```

> These are single-run numbers with no seed averaging. The k-reciprocal re-ranking step trades a
> little deep-rank recall (Rank-5 and Rank-10) for a large gain in mAP and Rank-1, which is the
> expected behavior. See [Reproducing the results](#reproducing-the-results) to regenerate them
> yourself.

### Reproducing the results

A full run takes roughly 40 minutes on a single modern GPU. The easiest path is the Colab notebook,
which sets everything up for you on a free T4.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vardhjain/Secondsight/blob/main/notebooks/train_colab.ipynb)

You can also run the whole pipeline locally from start to finish.

```bash
make download                        # fetch Market-1501 via kagglehub
make train DEVICE=cuda               # train the strong baseline -> outputs/best.pth
make eval  WEIGHTS=outputs/best.pth  # CMC/mAP, then k-reciprocal re-ranking
make demo  WEIGHTS=outputs/best.pth  # interactive probe-vs-gallery Gradio demo
```

## Development

The same `Makefile` exposes the quality gates that run in CI.

```bash
make lint          # ruff check .
make format        # ruff format . && ruff check --fix .
make format-check  # ruff format --check . && ruff check .
make typecheck     # mypy
make test          # pytest
make test-cov      # pytest with coverage
```

The test suite is split into two groups. The light tests cover the metrics, re-ranking, distance,
config, sampler, and losses, and they need only `numpy`, `torch`, and `Pillow`. The heavy tests
cover the model and the transform pipeline, and they are guarded with `pytest.importorskip` so they
skip cleanly when `torchvision` is not installed.

## Docker

```bash
make docker-build     # docker build -t secondsight .
make docker-run       # runs the Gradio demo on :7860, mounts ./outputs
```

## Model card

The [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) file documents the intended use, training data,
evaluation protocol, limitations, and ethical considerations for the model.

## Citation

If you use this software, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for the full text.
