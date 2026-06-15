# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Publish trained checkpoint and populate the measured "strong baseline" results row.
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

[Unreleased]: https://github.com/vardhjain/person-reid-market1501/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/vardhjain/person-reid-market1501/releases/tag/v0.1.0
