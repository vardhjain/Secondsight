# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1) and `SECURITY.md`.
- Colab training notebook (`notebooks/train_colab.ipynb`) for one-click GPU
  reproduction, linked from the README.
- README status badges and a Results section with measured metrics. The baseline
  reaches **85.0 mAP / 94.2 Rank-1**, rising to **93.7 mAP / 94.7 Rank-1** with
  k-reciprocal re-ranking (ResNet-50 + BNNeck strong baseline, seed 42, 60 epochs).
- Unit tests for the LR schedulers, checkpoint (de)serialization, device
  resolution, and the results-table renderer.
- Model card (`docs/MODEL_CARD.md`) covering intended use, data, metrics,
  limitations, and ethical considerations.

### Fixed
- `RandomIdentitySampler.__len__` is now a stable, batch-aligned value computed
  once at construction time; it was previously mutated during iteration, so
  `len(dataloader)` could disagree across epochs.
- README `make install` command now matches the Makefile (`uv sync --extra dev`).
- **Periodic evaluation cache:** the trainer now resets the evaluator's feature
  cache before each in-loop evaluation, so mAP reflects the current epoch and
  best-by-mAP saves the genuine best checkpoint (previously it reused the first
  epoch's features and locked `best.pth` to the earliest evaluated epoch).
- Camera-id parsing uses `\d+` (future-proof beyond nine cameras); `DataLoader`
  `pin_memory` is enabled only when CUDA is actually present.
- Gradio demo skips unreadable gallery images and gained an optional `--auth`
  flag plus a warning when bound off-loopback without authentication.

### Planned
- Publish the trained checkpoint as a downloadable GitHub Release asset.
- Optional IBN-Net backbone weights and a cross-dataset (Market → DukeMTMC / MSMT17)
  domain-generalization evaluation script.
- GPU-accelerated k-reciprocal re-ranking.

## [0.1.0] - 2026-06-14

First public release. Refactors the original research notebook into a clean,
installable Python package with tests, CI/CD, Docker, a Gradio demo, and docs.

### Added
- `reid` package (src-layout) with subpackages: `config`, `data`, `models`,
  `losses`, `engine`, `evaluation`, `utils`, `visualization`.
- Typed, YAML-backed configuration (`reid.config.Config`) with `default.yaml`
  and `market1501_strong_baseline.yaml`.
- ResNet-50 + BNNeck model with `last_stride=1`, selectable pooling
  (`avg`/`gem`/`max`), and an optional IBN backbone hook.
- Identity-balanced `RandomIdentitySampler` (PK sampling) so batch-hard triplet
  mining is well-posed.
- Losses: label-smoothing cross-entropy, batch-hard / soft-margin triplet,
  optional center loss, and a combined `ReIDLoss`.
- `Trainer` with AMP mixed precision, LR warmup (`WarmupMultiStepLR` /
  `WarmupCosineLR`), periodic evaluation, and best-by-mAP checkpointing.
- `Evaluator` with cached feature extraction, flip-TTA, L2-normalized (cosine)
  retrieval, CMC/mAP metrics, and k-reciprocal re-ranking.
- Visualization: Grad-CAM, ranked-result galleries, CMC curve, AP-distribution,
  per-camera heatmap, t-SNE, and intra/inter-class distance plots.
- CLI scripts (`download_data`, `train`, `evaluate`, `visualize`) and a Gradio
  probe-vs-gallery demo app.
- Test suite (light CPU tests + `torchvision`-gated heavy tests), GitHub Actions
  CI (ruff + pytest, Python 3.10–3.12), Dockerfile, docker-compose, pre-commit,
  and project documentation.

[Unreleased]: https://github.com/vardhjain/Secondsight/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/vardhjain/Secondsight/releases/tag/v0.1.0
