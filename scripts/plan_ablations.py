"""Generate deterministic one-factor-at-a-time Secondsight ablation plans.

Planning writes YAML configurations and a JSON command manifest, but never
trains a model or evaluates the benchmark. No network or dataset is required.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from reid.config import Config

# The base recipe remains unchanged except for an evaluation-only re-ranking
# flag: training's final test evaluation skips costly re-ranking because the
# report explicitly evaluates both choices afterwards.
TRAIN_VARIANTS = ("baseline", "avg_pool", "no_center_loss")
EVAL_VARIANTS = (
    ("flip_rerank", True, True),
    ("flip_plain", True, False),
    ("no_flip_rerank", False, True),
    ("no_flip_plain", False, False),
)


def build_plan(
    config_path: Path, output_dir: Path, seeds: list[int], val_ids: int = 0
) -> dict[str, Any]:
    """Write config/command artifacts without executing any training.

    Raises:
        ValueError: For invalid seeds, holdout size, or a non-empty output directory.
    """
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct non-negative integers.")
    if val_ids < 0:
        raise ValueError("val_ids must be non-negative.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {output_dir}")

    base = Config.from_yaml(config_path).validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []

    for seed in sorted(seeds):
        for variant in TRAIN_VARIANTS:
            run_id = f"{variant}_seed{seed}"
            run_dir = output_dir / run_id
            run_dir.mkdir()
            train_cfg = copy.deepcopy(base)
            train_cfg.train.seed = seed
            train_cfg.data.val_ids = val_ids
            # Keep dataset paths private/local: supply REID_DATA_ROOT at run time.
            train_cfg.data.root = None
            train_cfg.train.output_dir = str(run_dir)
            train_cfg.eval.rerank = False  # Separate re-ranking from training.
            if variant == "avg_pool":
                train_cfg.model.pooling = "avg"
            elif variant == "no_center_loss":
                train_cfg.loss.center_loss = False
            train_cfg.validate()

            train_path = run_dir / "train.yaml"
            train_cfg.to_yaml(train_path)
            train_command = [
                "reid-train",
                "--config",
                str(train_path),
                "--output-dir",
                str(run_dir),
                "--no-rerank",
            ]
            evaluations: list[dict[str, Any]] = []
            for name, flip, rerank in EVAL_VARIANTS:
                eval_cfg = copy.deepcopy(train_cfg)
                eval_cfg.eval.flip_tta = flip
                eval_cfg.eval.rerank = rerank
                eval_path = run_dir / f"eval_{name}.yaml"
                results_path = run_dir / f"results_{name}.json"
                eval_cfg.to_yaml(eval_path)
                evaluations.append(
                    {
                        "name": name,
                        "flip_tta": flip,
                        "rerank": rerank,
                        "config": str(eval_path),
                        "config_sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest(),
                        "results": str(results_path),
                        "argv": [
                            "reid-evaluate",
                            "--config",
                            str(eval_path),
                            "--weights",
                            str(run_dir / "model_final.pth"),
                            "--results-json",
                            str(results_path),
                        ],
                    }
                )
            runs.append(
                {
                    "id": run_id,
                    "variant": variant,
                    "seed": seed,
                    "train_config": str(train_path),
                    "train_config_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
                    "weights": str(run_dir / "model_final.pth"),
                    "train_argv": train_command,
                    "evaluations": evaluations,
                }
            )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "base_config": str(config_path),
        "val_ids": val_ids,
        "train_variants": list(TRAIN_VARIANTS),
        "evaluation_variants": [name for name, _, _ in EVAL_VARIANTS],
        "runs": runs,
        "note": (
            "Commands are not run automatically. Set REID_DATA_ROOT and run them "
            "from the same working directory. Benchmark evaluation sweeps are "
            "exploratory; do not use test scores for hyperparameter selection."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Generate an ablation plan from an existing training config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/market1501_strong_baseline.yaml")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablations"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--val-ids", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.config, args.output_dir, args.seeds, args.val_ids)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(
        f"Planned {len(plan['runs'])} training runs and "
        f"{sum(len(run['evaluations']) for run in plan['runs'])} evaluations "
        f"in {args.output_dir} (nothing executed)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
