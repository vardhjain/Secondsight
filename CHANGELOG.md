# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Reproducibility
- Pin the Colab notebook to a specific reviewed source commit and install
  from `uv.lock` instead of cloning moving `main` and using floating pip
  dependencies. Bundle source, lockfile and runtime provenance with new runs.
- Provide an opt-in strict YAML configuration checker that rejects unknown
  field names without changing the backwards-compatible `Config` loader.
- Correct README promises of exact reproduction and the model card's
  unmeasured run-to-run variance estimate; add a reproducibility guide and
  CI regression tests for the notebook and configuration checks.

### Added
- CPU-only synthetic retrieval distance/re-ranking timing harness with machine-
  readable latency samples, environment metadata and conservative dense-array
  lower-bound estimates (not measured peak memory).
- Deterministic training ablation plan generator (pooling and center loss),
  evaluation-only flip-TTA/re-ranking grid, seed manifests and config hashes.
- Results report generator that marks absent evaluation files as pending, plus
  no-dataset regression tests and a fast CI smoke run for all three scripts.
- Experiment protocol explaining train/eval cost, benchmark test holdout,
  synthetic-versus-real measurements, and the steps for optional repeated seeds.
- Preserve CI smoke outputs as downloadable artifacts so reviewers can inspect
  a real synthetic timing JSON and an honest pending-data ablation report.

### Documentation
- Make the repository's first screen point to real release artifacts, the
  evaluation code, and a concise project case study for technical reviewers.
- Clarify that the v0.2.0 run periodically monitored the test set, that the
  current default evaluation protocol is stricter, and that the published
  results have not been rerun under the revised protocol.
- Remove "production-grade" claims from package metadata and the model card;
  describe the tested scope without implying external deployment validation.

### Fixed
- Prevent routine Dependabot uv updates from attempting incompatible PyTorch
  and torchvision wheel upgrades across CPU/CUDA and unsupported Python
  environments. Paired version upgrades are now documented as manual while
  security updates remain eligible.

### Changed
- Replace the Hugging Face Space's uncalibrated same-person/different-person
  threshold verdict with a descriptive cosine similarity score, its valid range,
  and an explicit warning against interpreting it as a match probability.
- Update the Space documentation and model card to reflect the score-only demo,
  and add regression tests for high, orthogonal, and opposite embeddings.

### Fixed
- Skip periodic evaluation entirely when no validation holdout is requested
  (`data.val_ids: 0`). Evaluate the test split only once, after training, and
  do not create `best.pth` without validation metrics.
- Require enough per-camera samples in validation identities for every query
  to retain a cross-camera positive in the validation gallery.
- Use the always-produced `model_final.pth` as the default weight path in
  local demo and Docker workflows.

## [0.2.0] - 2026-10-04

### Added
- Trained weights (`model_final.pth`) published as a GitHub Release asset, and
  Grad-CAM, re-ranking and CMC figures from the same run shown in the README.
- A Hugging Face Space demo in `space/` that compares two uploaded person crops
  on a free CPU, with optional clickable example pairs and a deployment guide
  (`space/DEPLOYING.md`).
- An optional validation split. Setting `data.val_ids` holds out that many
  training identities (seen by at least two cameras, chosen deterministically
  from the seed) as a small query and gallery set, which then drives periodic
  evaluation and `best.pth` selection instead of the test split.
- Training now writes `model_final.pth`, `results.json` (final-epoch test
  metrics) and `history.json` (per-epoch log).
- New config fields `data.val_ids` (default `0`) and `train.deterministic`
  (default `true`), and `device: auto`, which picks CUDA, then Apple MPS, then
  the CPU.
- `reid.config.default_config_path()` and a copy of the YAML configs packaged
  inside the wheel, so the console commands work outside a checkout.
- `reid.models.reid_model.load_trained_model()` and
  `reid.utils.checkpoint.read_checkpoint_config()`, which rebuild the exact
  architecture recorded in a checkpoint and load it strictly.
- Optional GPU acceleration and lower memory use for k-reciprocal re-ranking.
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1) and `SECURITY.md`.
- Colab training notebook (`notebooks/train_colab.ipynb`). A single Run all
  trains, evaluates the final weights, renders the figures and packages the
  outputs for download.
- README status badges and a Results section with measured metrics. The baseline
  reaches **84.8 mAP / 93.9 Rank-1**, rising to **93.7 mAP / 95.0 Rank-1** with
  k-reciprocal re-ranking (seed 42, 60 epochs, final-epoch weights, reference
  squared-distance re-ranking).
- Model card (`docs/MODEL_CARD.md`) covering intended use, data, metrics,
  limitations, and ethical considerations.
- Unit tests for the LR schedulers, checkpoint (de)serialization, device
  resolution, the results-table renderer and the command-line tools, plus a
  test fixture that blocks every network download.
- A CI job that builds the wheel and source distribution, and a Docker
  `HEALTHCHECK`.

### Changed
- The distribution is now named `Secondsight` (previously
  `person-reid-market1501`). The import package is still `reid`.
- The command-line tools moved into the library as `reid.cli`, and the
  `reid-train`, `reid-evaluate`, `reid-visualize` and `reid-download` entry
  points now work from any directory. The files in `scripts/` are thin wrappers,
  and the wheel ships only the `reid` package plus its configs.
- `--config` defaults to the packaged strong-baseline recipe.
- Dependencies are split into a small core (PyTorch, torchvision, NumPy, Pillow,
  PyYAML, tqdm) and the optional extras `viz`, `demo`, `data`, `all` and `dev`,
  plus the uv-only `cpu` and `cuda` extras that select the PyTorch wheels.
  pandas is no longer a dependency. A missing extra produces an error that names
  the extra to install.
- The minimum versions are now PyTorch 2.3 and torchvision 0.18, and Gradio 5.50
  for the demo extra.
- Python 3.10 through 3.14 are supported and tested in CI.
- Headline results always come from the final-epoch weights evaluated on the
  test split, never from a checkpoint selected on the test set. The trainer also
  evaluates on the final epoch, so `best.pth` exists whenever evaluation is on,
  including short smoke runs.
- k-reciprocal re-ranking uses the squared Euclidean distance, matching the
  Zhong et al. and Bag of Tricks reference implementations. Re-ranked numbers
  change slightly as a result.
- Loss terms are computed in float32 outside autocast, while the model forward
  pass still uses mixed precision.
- Checkpoints no longer embed machine-specific absolute paths.
- Config loading coerces numeric strings such as `5e-4`, warns about unknown
  keys, and validates more settings (PK sampler sizes, `last_stride`, periods,
  ranks and re-ranking parameters). The train command validates the config
  after applying command-line overrides.
- Data loaders keep their worker processes alive between epochs.
- Requesting the IBN backbone loads a hub model pinned to a fixed commit and
  fails loudly instead of silently falling back to plain ResNet-50.
- Inference paths (evaluation, visualization and both demos) build the model
  without downloading ImageNet weights and load checkpoints with
  `weights_only=True`.
- The Gradio demo's `--server-name` and `--server-port` fall back to
  `GRADIO_SERVER_NAME` and `GRADIO_SERVER_PORT`, so the Docker image is
  reachable from the host.
- The Hugging Face Space targets Gradio 6.

### Fixed
- `RandomIdentitySampler.__len__` is now a stable, batch-aligned value computed
  once at construction time. It was previously mutated during iteration, so
  `len(dataloader)` could disagree across epochs.
- README `make install` command now matches the Makefile.
- **Periodic evaluation cache.** The trainer now resets the evaluator's feature
  cache before each in-loop evaluation, so mAP reflects the current epoch and
  best-by-mAP saves the genuine best checkpoint (previously it reused the first
  epoch's features and locked `best.pth` to the earliest evaluated epoch).
- Camera-id parsing uses `\d+` (future-proof beyond nine cameras), and
  `DataLoader` `pin_memory` is enabled only when CUDA is actually present.
- Gradio demo skips unreadable gallery images and gained an optional `--auth`
  flag plus a warning when bound off-loopback without authentication.
- The Colab smoke-run tip (`--max-epochs 5`) no longer breaks the evaluation and
  download cells that follow it.

### Planned
- A cross-dataset domain-generalization evaluation script, targeting MSMT17
  (subject to its access terms). DukeMTMC-reID was withdrawn by its authors in
  2019 and will not be used.

## [0.1.0] - 2026-06-14

First public version. Refactors the original research notebook into a clean,
installable Python package with tests, CI/CD, Docker, a Gradio demo, and docs.
This version was never tagged, so it has no release page.

### Added
- `reid` package (src-layout) with the subpackages `config`, `data`, `models`,
  `losses`, `engine`, `evaluation`, `utils` and `visualization`.
- Typed, YAML-backed configuration (`reid.config.Config`) with `default.yaml`
  and `market1501_strong_baseline.yaml`.
- ResNet-50 + BNNeck model with `last_stride=1`, selectable pooling
  (`avg`/`gem`/`max`), and an optional IBN backbone hook.
- Identity-balanced `RandomIdentitySampler` (PK sampling) so batch-hard triplet
  mining is well-posed.
- Label-smoothing cross-entropy, batch-hard and soft-margin triplet losses,
  optional center loss, and a combined `ReIDLoss`.
- `Trainer` with AMP mixed precision, LR warmup (`WarmupMultiStepLR` /
  `WarmupCosineLR`), periodic evaluation, and best-by-mAP checkpointing.
- `Evaluator` with cached feature extraction, flip-TTA, L2-normalized (cosine)
  retrieval, CMC/mAP metrics, and k-reciprocal re-ranking.
- Visualization tools for Grad-CAM, ranked-result galleries, CMC curves, AP
  distributions, per-camera heatmaps, t-SNE, and intra/inter-class distances.
- CLI scripts (`download_data`, `train`, `evaluate`, `visualize`) and a Gradio
  probe-vs-gallery demo app.
- Test suite (light CPU tests + `torchvision`-gated heavy tests), GitHub Actions
  CI (ruff + pytest, Python 3.10–3.12), Dockerfile, docker-compose, pre-commit,
  and project documentation.

[Unreleased]: https://github.com/vardhjain/Secondsight/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/vardhjain/Secondsight/releases/tag/v0.2.0
[0.1.0]: https://github.com/vardhjain/Secondsight/commit/2aea8e4
