# Reproducibility notes

Secondsight separates **historical published evidence**, **current code**, and
**fresh experiment runs**. Keeping those distinct is necessary to interpret
metrics correctly.

## Published v0.2.0 model

The [v0.2.0 release](https://github.com/vardhjain/Secondsight/releases/tag/v0.2.0)
includes the original `model_final.pth`, `config.yaml`, `history.json`,
and `results.json`. Its reported Market-1501 metrics come from **one
60-epoch training run with seed 42**. The run used the final-epoch checkpoint
rather than a checkpoint selected for its test score, but its training script
**periodically monitored the test partition**. Those measurements are
historical, not newly rerun after the stricter validation protocol landed.

There are no demonstrated confidence intervals, independent cross-dataset
replications, or guaranteed bit-identical results across GPU types. Seeding
reduces variation but cannot force every CUDA operation to be deterministic:
the code calls `torch.use_deterministic_algorithms(..., warn_only=True)`.

## Re-run with the current workflow

The [Colab notebook](../notebooks/train_colab.ipynb) now checks out the
immutable source commit `4faaecb949736a4c6306716082fb2ccc5031c2ed`
(a reviewed Task 6 revision that includes the strict checker) rather than moving `main`. It uses `uv
sync --locked` against that checkout's `uv.lock` and activates the
`cuda`, `data` and `viz` extras.

When a complete run finishes, it writes `outputs/provenance.json` containing
the exact source commit, lockfile SHA-256, environment versions and coarse
dataset file counts. It is included with the downloadable output archive.
This makes *future* results traceable; it does not imply the new run must
match the released v0.2.0 numbers.

A fixed commit and lockfile do not pin **GPU driver, CUDA driver/runtime,
the Kaggle dataset snapshot, or hosted hardware**, nor guarantee the notebook
will execute on every future Colab image. Free GPU availability also varies.
If the pinned commit becomes unavailable, record the replacement commit and
revalidate the notebook rather than silently falling back to `main`.

## Validate configuration before consuming GPU time

The normal `Config.from_yaml` loader warns and ignores unknown YAML keys for
backward compatibility. That means a typo can leave a default in effect.

For experiment runs, use the strict checker first:

```bash
python scripts/validate_config.py \
  --config configs/market1501_strong_baseline.yaml
```

It rejects unknown sections and nested keys (including suggested corrections)
and validates the known field values. The notebook runs this check on the
selected recipe before training. This does **not** silently change behavior
for library clients relying on the existing, lenient loader.

For a local locked environment:

```bash
uv sync --locked --extra cpu --extra all --extra dev
uv run --no-sync python scripts/validate_config.py \
  --config configs/market1501_strong_baseline.yaml
uv run --no-sync pytest
```

The Colab notebook intentionally selects CUDA-capable wheels instead of the
CPU extra. Running it requires legitimate Market-1501 dataset access.

## Experiment reporting boundaries

The [ablation harness](EXPERIMENTS.md) generates configurations and can time
retrieval on synthetic data without training or paid compute. **Pending**
ablations are not results. A synthetic retrieval timing is not a measured
full-model latency, throughput, peak memory or real-data accuracy score.

For a portfolio review, cite the [recorded release results](https://github.com/vardhjain/Secondsight/releases/tag/v0.2.0)
with their methodological limitations; cite fresh experiment outputs only
when the experiment, environment and dataset provenance are present.
