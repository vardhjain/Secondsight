<div align="center">

# 🔍 Secondsight

### Cross-camera person re-identification on Market-1501

Give it one cropped photo of a person and it searches a gallery of other photos, ranking every
candidate by how likely it is to be that same individual seen again on a different, non-overlapping
camera. Secondsight runs a carefully engineered **ResNet-50 + BNNeck** pipeline that reaches
**84.8 mAP / 93.8 Rank-1**, rising to **93.6 mAP / 95.1 Rank-1** after k-reciprocal re-ranking.

**[▶ Open in Colab](https://colab.research.google.com/github/vardhjain/Secondsight/blob/main/notebooks/train_colab.ipynb)** &nbsp;·&nbsp; **[📊 Results](#results)** &nbsp;·&nbsp; **[🤗 Live demo](#hugging-face-space)** &nbsp;·&nbsp; **[🧠 Model card](docs/MODEL_CARD.md)**

[![CI](https://github.com/vardhjain/Secondsight/actions/workflows/ci.yml/badge.svg)](https://github.com/vardhjain/Secondsight/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c.svg)](https://pytorch.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

</div>

> [!NOTE]
> Person re-identification is dual-use and surveillance-adjacent. Please read the ethics and
> limitations section of the [model card](docs/MODEL_CARD.md) before using this work beyond
> research and benchmarking.

---

Under the hood Secondsight closely follows the well-known **ResNet-50 + BNNeck "strong baseline"**
(Luo et al., *Bag of Tricks*, CVPRW 2019) and wraps it in the kind of engineering you would expect
from a real project rather than a one-off research script. The network learns a 2048-dimensional
embedding, which is a compact numeric fingerprint, for each person crop, and it judges two crops to
be the same person when their embeddings sit close together under cosine distance.

The training recipe follows the modern Re-ID playbook. Every batch is drawn by an identity-balanced
**PK sampler** that picks 16 identities with 4 images each, for a batch of 64, which guarantees that
each batch holds enough same-person pairs for **batch-hard triplet mining** to actually work. The
ResNet-50 backbone keeps a higher-resolution feature map by using a stride of 1 in its final stage
(`last_stride=1`) and pools it with **GeM (generalized-mean) pooling**. Training blends
**label-smoothed cross-entropy** with a **batch-hard triplet loss**, and the headline recipe also
switches on **center loss**. A short **learning-rate warmup** flows into a multistep decay schedule,
and **automatic mixed precision (AMP)** keeps the run fast and light on memory, while every loss term
is still computed in full float32 precision for stability. At test time the features are
L2-normalized so that Euclidean distance becomes cosine similarity, each image is averaged with its
horizontal flip for a small **test-time augmentation (TTA)** gain, and an optional **k-reciprocal
re-ranking** pass (Zhong et al., CVPR 2017) pushes accuracy higher still. So the model is not a
black box, the repo can also render **Grad-CAM** attention maps and ranked-result galleries that show
what the model focuses on and where it succeeds or fails.

The package is deliberately layered to stay lightweight. A core install brings only PyTorch,
torchvision, NumPy, Pillow, PyYAML, and tqdm. Plotting, the Gradio demo, and the Kaggle downloader
live in optional extras, and they are imported lazily inside only the submodules that need them, so a
plain `import reid` never pulls them in. If you call a feature whose extra is missing, the error
message tells you exactly which extra to install.

## Project layout

```
src/reid/          # the importable library (src-layout)
  cli/             # train, evaluate, visualize and download command-line tools
  config.py        # typed, YAML-backed configuration dataclasses
  data/            # Market-1501 dataset, validation split, PK sampler, transforms
  losses/          # triplet, label-smooth CE, center, combined ReIDLoss
  models/          # ResNet-50 backbone, GeM pooling, BNNeck ReIDModel
  evaluation/      # CMC/mAP metrics, k-reciprocal re-ranking, evaluator
  engine/          # trainer and LR scheduler
  utils/           # distance, checkpoint, meters, logging, reproducibility
  visualization/   # Grad-CAM, ranking galleries, analysis plots
scripts/           # thin wrappers that run the CLIs straight from a checkout
app/               # gradio_app.py interactive probe-vs-gallery demo
space/             # Hugging Face Space (two-crop similarity demo)
configs/           # default.yaml and market1501_strong_baseline.yaml
notebooks/         # Colab training notebook and the original research notebook
docs/              # model card and figures
tests/             # pytest suite (light tests plus torchvision-gated heavy tests)
```

## Installation

Secondsight supports Python 3.10 through 3.14. It uses a `src` layout, builds with `hatchling`, and
splits its dependencies into a small core plus optional extras. The `viz` extra adds the plotting
stack (matplotlib, seaborn, scikit-learn, OpenCV), `demo` adds Gradio, `data` adds the `kagglehub`
downloader, `all` combines those three, and `dev` adds the linting and testing tools.

The quickest setup uses [`uv`](https://github.com/astral-sh/uv), which creates the environment and
installs everything in a single step.

```bash
uv sync --extra all --extra dev               # machines with a CUDA GPU
uv sync --extra all --extra dev --extra cpu   # CPU-only machines (installs the CPU PyTorch wheels)
```

`make install` runs the same `uv sync` for you. If you prefer plain pip, install the package in
editable mode with the extras you need.

```bash
pip install -e ".[all,dev]"
```

## Dataset

Download Market-1501 with the bundled helper, which fetches it from Kaggle through `kagglehub` (part
of the `data` extra) and prints the dataset root on its last line.

```bash
reid-download
```

Then point the pipeline at that root, meaning the directory that holds `bounding_box_train/`,
`query/`, and `bounding_box_test/`. You can supply it with the `--data-root` flag or by exporting the
`REID_DATA_ROOT` environment variable in your shell. The `.env.example` file lists every variable the
project understands, but note that nothing loads a `.env` file automatically apart from
`docker compose`, so for local runs you export the variables yourself.

## Usage

Installing the package gives you four console commands, `reid-download`, `reid-train`,
`reid-evaluate`, and `reid-visualize`. They live inside the library (`reid.cli`), ship their default
configuration inside the wheel, and therefore work from any directory, not only from a checkout.

```bash
export REID_DATA_ROOT=/path/to/Market-1501-v15.09.15

reid-train     --output-dir outputs                          # strong-baseline recipe by default
reid-evaluate  --weights outputs/model_final.pth             # CMC/mAP, plus re-ranking
reid-visualize --weights outputs/model_final.pth --output figures
python app/gradio_app.py --weights outputs/model_final.pth   # interactive demo (needs the demo extra)
```

When `--config` is omitted the commands use the packaged `market1501_strong_baseline.yaml`. Pass
`--config configs/default.yaml` or your own YAML file to change the recipe, and run any command with
`--help` for the full list of flags. Inside a checkout, `python scripts/train.py` and its siblings are
thin wrappers around the same entry points.

The `Makefile` wraps the common workflows too, and you can override `CONFIG`, `DATA_ROOT`,
`OUTPUT_DIR`, `DEVICE`, and `WEIGHTS` on the command line.

```bash
make train DATA_ROOT=/path/to/Market-1501-v15.09.15
make eval  DATA_ROOT=/path/to/Market-1501-v15.09.15 WEIGHTS=outputs/model_final.pth
make demo  WEIGHTS=outputs/model_final.pth
```

## Configuration

Configuration is fully typed through `reid.config` and round-trips cleanly to and from YAML. The
file `configs/default.yaml` mirrors the dataclass defaults exactly, and it is identical to the
headline recipe in `configs/market1501_strong_baseline.yaml` except that center loss is switched off.
Numeric strings such as `5e-4` are coerced to the right type, unknown keys produce a warning instead
of being silently ignored, and the config is validated after command-line overrides are applied, so a
typo fails fast rather than an hour into training. Setting `device: auto` picks CUDA, then Apple MPS,
then the CPU.

### Validation split and honest model selection

Market-1501 has no official validation set, and choosing the "best" epoch by its test-set mAP would
quietly leak the test set into model selection. Secondsight avoids that in two ways. First, setting
`data.val_ids` to a positive number holds out that many training identities (chosen
deterministically from the seed among identities seen by at least two cameras) as a small query and
gallery split, and periodic evaluation and `best.pth` selection then use it instead of the test set.
With the default of `0`, periodic evaluation still runs on the test split, and the log says plainly
that it is for monitoring only. Second, and regardless of that setting, **the headline numbers always
come from the final-epoch weights** (`model_final.pth`) evaluated once on the test split, never from a
test-selected checkpoint. Training writes those metrics to `results.json` and the epoch-by-epoch log
to `history.json`.

## Results

These numbers were measured on Market-1501 from a single training run (seed 42, 60 epochs, roughly
40 minutes on a Colab T4 GPU). The right-hand column shows the figures Luo et al. (2019) report for
their original strong baseline, and this run comes within about one point of them while training
for half as many epochs.

They come from the final-epoch weights (`model_final.pth`), evaluated once on the test split, and
use the reference squared-distance form of k-reciprocal re-ranking. No checkpoint was chosen by its
test-set score.

| Setting                   |  mAP   | Rank-1 | Rank-5 | Rank-10 | Reference (Luo et al., 2019)¹ |
| ------------------------- | :----: | :----: | :----: | :-----: | :---------------------------: |
| Cosine + flip-TTA         | 84.84% | 93.79% | 98.19% | 98.84%  |     ~85.9 mAP / ~94.5 R-1     |
| + k-reciprocal re-ranking | 93.57% | 95.07% | 97.57% | 98.10%  |     ~94.2 mAP / ~95.4 R-1     |

¹ The reference figures were measured without flip test-time augmentation, so the comparison is
close but not strictly like for like.

Running `reid-evaluate` on the trained checkpoint prints the same scorecard to the terminal.

```text
------------------------------------------------------
Setting              mAP    Rank-1    Rank-5   Rank-10
------------------------------------------------------
Baseline         84.84%    93.79%    98.19%    98.84%
Re-ranked        93.57%    95.07%    97.57%    98.10%
------------------------------------------------------
```

> These are single-run numbers with no seed averaging. The k-reciprocal re-ranking step trades a
> little deep-rank recall (Rank-5 and Rank-10) for a large gain in mAP and a small gain in Rank-1,
> which is the expected behavior.

### How this recipe differs from the paper

The recipe is the Bag of Tricks strong baseline with a few deliberate changes, stated here so the
comparison stays honest. It pools with GeM instead of global average pooling. It trains for 60
epochs with learning-rate decay at epochs 30 and 50, where the paper trains for 120 epochs with decay
at 40 and 70. It evaluates with horizontal-flip test-time augmentation, which the reference evaluator
does not use. The results come from a single seed. Re-ranking uses the squared Euclidean distance,
exactly like the Zhong et al. and Bag of Tricks reference implementations, and the evaluation keeps
the standard Market-1501 rules, dropping junk images (identity `-1`) while keeping the distractor
crops (identity `0`) in the gallery. Average precision is the non-interpolated form used by the
reference code, which differs slightly from the trapezoidal variant in the original MATLAB devkit.

### Reproducing the results

The easiest path is the Colab notebook. A single **Run all** on a free T4 installs the package,
downloads Market-1501, trains the strong baseline, evaluates the final weights with re-ranking,
prints `results.json`, renders the figures, and packages everything as a zip for download.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vardhjain/Secondsight/blob/main/notebooks/train_colab.ipynb)

You can also run the whole pipeline locally from start to finish.

```bash
reid-download                                     # prints the Market-1501 root on its last line
export REID_DATA_ROOT=/path/printed/above
make train DEVICE=cuda                            # writes outputs/model_final.pth and results.json
make eval  WEIGHTS=outputs/model_final.pth        # CMC/mAP, then k-reciprocal re-ranking
make demo  WEIGHTS=outputs/model_final.pth        # interactive probe-vs-gallery Gradio demo
```

## Hugging Face Space

The [`space/`](space/) folder holds a small public demo for Hugging Face Spaces. A visitor uploads two
person crops and the Space reports how similar their embeddings are. It runs on a free CPU, hosts no
gallery, stores nothing, and its similarity threshold is not calibrated, so the verdict is a rough
guide rather than an identification. See [`space/DEPLOYING.md`](space/DEPLOYING.md) for the
deployment steps.

## Docker

The image runs the Gradio demo and listens on all interfaces inside the container, so the mapped port
is reachable from the host.

```bash
make docker-build     # docker build -t secondsight .
make docker-run       # runs the Gradio demo on :7860, mounts ./outputs
```

## Development

The same `Makefile` exposes the quality gates that run in CI, which tests every Python version from
3.10 to 3.14.

```bash
make lint          # ruff check .
make format        # ruff format . && ruff check --fix .
make format-check  # ruff format --check . && ruff check .
make typecheck     # mypy
make test          # pytest
make test-cov      # pytest with coverage
```

The test suite is split into two groups. The light tests cover the config system, dataset metadata,
PK sampler, losses, LR schedulers, distance utilities, metrics, re-ranking, checkpoint I/O, device
resolution, and the results table, and they need only NumPy, PyTorch, Pillow, and PyYAML. The heavy
tests cover the model, the transform pipeline, and the command-line tools, and they are guarded with
`pytest.importorskip` so they skip cleanly when `torchvision` is not installed. Tests never touch the
network, since a shared fixture blocks weight and dataset downloads.

## Model card

The [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) file documents the intended use, training data,
evaluation protocol, limitations, and ethical considerations for the model.

## Citation

If you use this software, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for the full text.
