"""Train the Person Re-Identification strong baseline.

This CLI wires the full training pipeline together from a single YAML config.
It seeds all RNGs, builds the PK-sampled train loader with the query and
gallery loaders (plus an optional validation split held out from the training
identities), builds the ResNet-50 + BNNeck model, the combined loss, the
optimizer and the warmup scheduler, and runs the
:class:`reid.engine.trainer.Trainer` loop.

Model selection for ``best.pth`` uses the validation split when
``data.val_ids > 0``. Without it, periodic evaluation runs on the test split and
is monitoring only. The headline results are always the final-epoch weights
evaluated on the test split (with re-ranking unless ``--no-rerank``). They are
saved as ``model_final.pth`` and ``results.json`` and printed as a table.

Common settings can be overridden on the command line without editing the YAML
(data root, output dir, device, epoch count, AMP, seed, determinism and the
validation split size). Overrides are validated like the YAML itself.

Example:
    Train with the headline strong-baseline recipe::

        reid-train --data-root /path/to/Market-1501 --output-dir outputs/strong_baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

from reid.config import Config, default_config_path
from reid.data.build import build_dataloaders
from reid.engine.scheduler import build_scheduler
from reid.engine.trainer import Trainer
from reid.evaluation.evaluator import Evaluator
from reid.evaluation.reporting import format_results_table
from reid.losses.build import build_loss
from reid.models.reid_model import build_model
from reid.utils.checkpoint import save_model
from reid.utils.device import resolve_device
from reid.utils.logging import setup_logger
from reid.utils.reproducibility import set_seed

logger = logging.getLogger("reid.cli.train")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the training CLI.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="reid-train",
        description="Train the ResNet-50 + BNNeck Re-ID strong baseline on Market-1501.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to the YAML configuration file. Defaults to the shipped "
            "market1501_strong_baseline.yaml."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Path to the Market-1501 data root (the directory containing "
            "bounding_box_train/, query/ and bounding_box_test/). Overrides "
            "data.root from the config. Run `reid-download` to obtain it."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints and logs. Overrides train.output_dir.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('auto', 'cuda', 'mps' or 'cpu'). Overrides train.device.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Number of training epochs. Overrides train.max_epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Overrides train.seed.",
    )
    parser.add_argument(
        "--val-ids",
        type=int,
        default=None,
        help=(
            "Number of training identities held out for validation and best.pth "
            "selection (0 disables the split). Overrides data.val_ids."
        ),
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision even if enabled in the config.",
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_true",
        help=(
            "Disable deterministic kernels and enable cudnn.benchmark for faster, "
            "slightly less reproducible training. Overrides train.deterministic."
        ),
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the re-ranked variant in the final evaluation.",
    )
    return parser


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Apply command-line overrides onto a loaded config and re-validate it.

    Args:
        cfg: The config loaded from YAML.
        args: Parsed CLI arguments.

    Returns:
        The same (mutated) config for convenience.

    Raises:
        ValueError: If an overridden value fails :meth:`Config.validate`.
    """
    if args.data_root is not None:
        cfg.data.root = str(args.data_root)
    if args.output_dir is not None:
        cfg.train.output_dir = str(args.output_dir)
    if args.device is not None:
        cfg.train.device = args.device
    if args.max_epochs is not None:
        cfg.train.max_epochs = args.max_epochs
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.val_ids is not None:
        cfg.data.val_ids = args.val_ids
    if args.no_amp:
        cfg.train.amp = False
    if args.no_deterministic:
        cfg.train.deterministic = False
    return cfg.validate()


def _build_optimizer(cfg: Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build the main optimizer from configuration.

    Args:
        cfg: The full experiment configuration. Only ``cfg.optim`` is used.
        model: The model whose parameters are optimized.

    Returns:
        A configured optimizer (Adam or SGD).

    Raises:
        ValueError: If ``cfg.optim.name`` is not a supported optimizer.
    """
    name = cfg.optim.name.lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.optim.lr,
            momentum=0.9,
            weight_decay=cfg.optim.weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optim.name!r} (expected 'adam' or 'sgd').")


def _json_metrics(results: dict[str, Any]) -> dict[str, Any]:
    """Keep the scalar metrics of an evaluation result for ``results.json``.

    Args:
        results: The mapping returned by :meth:`Evaluator.evaluate`.

    Returns:
        The same mapping without the full CMC arrays, with plain floats.
    """
    return {k: float(v) for k, v in results.items() if not k.endswith("cmc")}


def main(argv: list[str] | None = None) -> int:
    """Run the end-to-end training pipeline.

    Args:
        argv: Optional list of command-line arguments (defaults to
            ``sys.argv[1:]``).

    Returns:
        Process exit code: ``0`` on success, non-zero on failure.
    """
    args = build_parser().parse_args(argv)

    config_path = args.config if args.config is not None else default_config_path()
    if not config_path.is_file():
        logger.error("Config file not found: %s", config_path)
        return 1
    try:
        cfg = _apply_overrides(Config.from_yaml(config_path), args)
    except (TypeError, ValueError) as exc:
        logger.error("Invalid configuration: %s", exc)
        return 1

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger("reid", output_dir=output_dir)

    if cfg.data.root is None:
        env_root = os.environ.get("REID_DATA_ROOT")
        if env_root:
            cfg.data.root = env_root
    if cfg.data.root is None:
        logger.error(
            "No dataset root configured. Pass --data-root, set data.root in the "
            "config, or set the REID_DATA_ROOT environment variable. "
            "Run `reid-download` to obtain the Market-1501 path."
        )
        return 1

    data_root = Path(cfg.data.root)
    if not data_root.is_dir():
        logger.error("Dataset root does not exist: %s", data_root)
        return 1

    set_seed(cfg.train.seed, deterministic=cfg.train.deterministic)
    device = resolve_device(cfg.train.device)
    cfg.train.device = str(device)

    logger.info("Loaded config from %s", config_path)
    cfg.to_yaml(output_dir / "config.yaml")

    # Build data, model, loss, optimizer and scheduler.
    loaders = build_dataloaders(cfg, root=data_root)
    num_classes = int(loaders["num_classes"])

    model = build_model(cfg, num_classes=num_classes).to(device)
    loss_fn = build_loss(cfg, num_classes=num_classes, feat_dim=model.feat_dim)
    optimizer = _build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    test_evaluator = Evaluator(
        model=model,
        query_loader=loaders["query_loader"],
        gallery_loader=loaders["gallery_loader"],
        device=device,
        cfg=cfg,
    )
    val_query = loaders.get("val_query_loader")
    val_gallery = loaders.get("val_gallery_loader")
    if cfg.data.val_ids > 0 and val_query is not None and val_gallery is not None:
        logger.info(
            "Selecting best.pth on a validation split of %d held-out training identities.",
            cfg.data.val_ids,
        )
        periodic_evaluator = Evaluator(
            model=model,
            query_loader=val_query,
            gallery_loader=val_gallery,
            device=device,
            cfg=cfg,
        )
    else:
        logger.info(
            "No validation split (data.val_ids=0), so periodic evaluation uses the test "
            "split for monitoring only. The reported results are the final-epoch weights."
        )
        periodic_evaluator = test_evaluator

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        train_loader=loaders["train_loader"],
        evaluator=periodic_evaluator,
        logger=logging.getLogger("reid.engine.trainer"),
    )

    summary = trainer.train()
    logger.info(
        "Training finished | best mAP=%.2f%% (epoch %s) | elapsed=%.1fs",
        100.0 * summary["best_mAP"],
        summary["best_epoch"],
        summary["elapsed_seconds"],
    )

    # Headline results: the final-epoch weights on the test split. When the
    # trainer monitored the test split, its forced final-epoch evaluation has
    # already cached features for exactly these weights, so they are reused.
    # A separate validation evaluator leaves the test evaluator's cache empty.
    logger.info("Running final evaluation on the test split...")
    do_rerank = not args.no_rerank and cfg.eval.rerank
    results = test_evaluator.evaluate(rerank=do_rerank)
    logger.info("Final results:\n%s", format_results_table(results))

    metrics = _json_metrics(results)
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved final metrics to %s", results_path)

    # Save a standalone, inference-ready weights file alongside the checkpoints.
    final_weights = output_dir / "model_final.pth"
    save_model(
        model,
        final_weights,
        num_classes=num_classes,
        mAP=metrics["mAP"],
        rank1=metrics["rank1"],
        config=cfg.to_portable_dict(),
    )
    logger.info("Saved final model weights to %s", final_weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
