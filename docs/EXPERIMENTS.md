# Reproducible ablations and retrieval benchmarks

These scripts provide a **measurement harness**, not newly measured Market-1501
accuracy. The published v0.2.0 mAP and Rank-1 metrics are from a single
historical training run; no new model-training scores are included here.

## 1. Plan ablations without a GPU

Use an environment with Secondsight and its dependencies installed (for
example `uv sync --locked --extra cpu --extra dev`). Run at the repository
root:

```bash
uv run --no-sync python scripts/plan_ablations.py \
  --output-dir outputs/ablations --seeds 42
```

This **does not train**. It creates an auditable `manifest.json` with
config-file SHA-256 hashes and exact CLI argument arrays. By default, it
plans three independent training variants from the same strong-baseline
recipe: `baseline`, `avg_pool` (GeM → average pooling), and
`no_center_loss` (center loss on → off), at seed 42.

For every resulting checkpoint, it generates **four evaluation-only**
settings: flip-TTA on/off crossed with re-ranking on/off. Changing TTA and
re-ranking **does not require retraining**. These comparisons form a
one-factor-at-a-time training study with a separate 2×2 evaluation factorial.
The plan deliberately disables the expensive in-training re-ranking; the
distinct evaluation commands provide both metrics afterwards.

To add repeated training seeds, pass `--seeds 42 123 456` (nine training
runs; potentially expensive). Only the **configuration generation** and
synthetic CPU benchmark run in CI. Neither downloads Market-1501, trains a
model, nor consumes a paid GPU.

### Run a planned experiment only when compute is available

First obtain Market-1501 under its license and export `REID_DATA_ROOT`
(see the top-level README). From the **same working directory** where you
created the plan, execute each `train_argv` command listed in
`outputs/ablations/manifest.json` manually. Then execute the four
`argv` evaluation commands for each completed training run.

The executable arrays avoid shell quoting assumptions but are not invoked
automatically. Each run writes its own `model_final.pth` and metric JSONs.
The `outputs/` folder is git-ignored. Do not commit model weights, benchmark
dataset crops, or large results directories.

For a small validation holdout selected from training identities, add
`--val-ids 50` to planning. This retains only eligible cross-camera identities
and reduces training population; compare only with runs using the same holdout
settings. **Important:** current `reid-train` still produces final-epoch
test metrics even in validation mode; it does not expose a validation-only
benchmark API. Therefore avoid inspecting final test results or running
alternative test sweeps during model/hyperparameter selection.

**Methodology:** preregister the comparisons before running them; use
training-only validation to select models and hyperparameters; reserve a
separate test evaluation for a final locked recipe. Repeated test-set
comparisons are *exploratory* and cannot justify a new independent final
benchmark claim. The historical v0.2.0 result was monitored on test during
training; the current code no longer does periodic test monitoring.

## 2. Produce the ablation report

```bash
uv run --no-sync python scripts/report_ablations.py \
  --manifest outputs/ablations/manifest.json \
  --output outputs/ablations/report.md
```

The report reads result JSONs already written by `reid-evaluate`. Missing
ones are marked **pending**, never replaced with guessed metrics. Recorded
mAP/Rank-1 values are matched to the correct plain vs. re-ranked variant;
mAP deltas compare the same seed and evaluation setting against the baseline.
Cross-seed differences are not silently treated as statistical significance.
Do not publish results unless the associated measurements have actually run.

## 3. CPU-only synthetic retrieval profiling

```bash
uv run --no-sync python scripts/benchmark_retrieval.py \
  --queries 64 --gallery 128 --dim 128 \
  --repeats 5 --warmups 1 --threads 1 \
  --output outputs/benchmarks/retrieval.json
```

This runs two actual algorithms using **synthetically generated L2-normalized
embeddings**: (a) query–gallery pairwise Euclidean distance and (b)
k-reciprocal re-ranking. It emits separate latency samples, median and p95
times, seed, input dimensions, platform/Python/PyTorch/NumPy versions and
thread count. It does **not** time image transforms, ResNet-50 inference, GPU
execution, or full-dataset Market-1501 evaluation.

The JSON also includes **analytical lower bounds** for dense arrays:
`4 × Q × G` bytes for the base distance result, and at least
`6 × (Q + G)² + 4 × Q × G` bytes for re-ranking's key dense arrays.
Actual peak RAM can be **much larger**, because intermediates, sorting
buffers, PyTorch tensors, and Python objects are excluded. These estimates
are not observations of resident memory. An explicit `--allow-large` is
required if the re-ranking lower bound exceeds 512 MiB.

Only compare timings on equivalent hardware, dependency versions, inputs,
warmups and thread counts. Avoid treating synthetic performance as a
production serving latency or accuracy claim.

## 4. Existing evidence and future work

Start with [the released v0.2.0 model artifacts](https://github.com/vardhjain/Secondsight/releases/tag/v0.2.0)
and [the technical case study](PROJECT_SHOWCASE.md). A rigorous follow-up
would run the preregistered ablation matrix on available compute, report
mean/variation across repeated seeds, and separately benchmark *model
inference* and realistic gallery sizes with measured peak RSS. None of those
results are claimed by this PR.
