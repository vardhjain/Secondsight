"""Train the Person Re-Identification strong baseline.

This CLI wires together the full training pipeline from a single YAML config:

#. seed all RNGs for reproducibility,
#. build the PK-sampled train loader plus query/gallery loaders,
#. build the ResNet-50 + BNNeck model, combined loss, optimizer and warmup
   scheduler,
#. run the :class:`reid.engine.trainer.Trainer` training loop with periodic
   evaluation and best-by-mAP checkpointing, and
#. run a final evaluation both with and without k-reciprocal re-ranking and
   print a results table.

A handful of common settings can be overridden on the command line without
editing the YAML (data root, output dir, device, epoch count, AMP, seed).

Example:
    Train with the headline strong-baseline recipe::

        python -m scripts.train \\
            --config configs/market1501_strong_baseline.yaml \\
            --data-root /path/to/Market-1501-v15.09.15 \\
            --output-dir outputs/strong_baseline
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

from reid.config import Config
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

_DEFAULT_CONFIG = "configs/market1501_strong_baseline.yaml"

logger = logging.getLogger("reid.scripts.train")


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
        default=Path(_DEFAULT_CONFIG),
        help="Path to the YAML configuration file.",
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
        help="Compute device, e.g. 'cuda' or 'cpu'. Overrides train.device.",
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
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision even if enabled in the config.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip the re-ranked variant in the final evaluation.",
    )
    return parser


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Apply command-line overrides onto a loaded config.

    Args:
        cfg: The config loaded from YAML.
        args: Parsed CLI arguments.

    Returns:
        The same (mutated) config for convenience.
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
    if args.no_amp:
        cfg.train.amp = False
    return cfg


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
    params = filter(lambda p: p.requires_grad, model.parameters())
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


def main(argv: list[str] | None = None) -> int:
    """Run the end-to-end training pipeline.

    Args:
        argv: Optional list of command-line arguments (defaults to
            ``sys.argv[1:]``).

    Returns:
        Process exit code: ``0`` on success, non-zero on failure.
    """
    args = build_parser().parse_args(argv)

    if not args.config.is_file():
        logger.error("Config file not found: %s", args.config)
        return 1
    cfg = Config.from_yaml(args.config)
    cfg = _apply_overrides(cfg, args)

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

    set_seed(cfg.train.seed, deterministic=True)
    device = resolve_device(cfg.train.device)
    cfg.train.device = str(device)

    logger.info("Loaded config from %s", args.config)
    cfg.to_yaml(output_dir / "config.yaml")

    # Build data, model, loss, optimizer and scheduler.
    loaders = build_dataloaders(cfg, root=data_root)
    num_classes = int(loaders["num_classes"])

    model = build_model(cfg, num_classes=num_classes).to(device)
    loss_fn = build_loss(cfg, num_classes=num_classes, feat_dim=model.feat_dim)
    optimizer = _build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    evaluator = Evaluator(
        model=model,
        query_loader=loaders["query_loader"],
        gallery_loader=loaders["gallery_loader"],
        device=device,
        cfg=cfg,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        train_loader=loaders["train_loader"],
        evaluator=evaluator,
        logger=logging.getLogger("reid.engine.trainer"),
    )

    summary = trainer.train()
    logger.info(
        "Training finished | best mAP=%.2f%% (epoch %s) | elapsed=%.1fs",
        100.0 * summary["best_mAP"],
        summary["best_epoch"],
        summary["elapsed_seconds"],
    )

    # Final evaluation on the latest weights (re-extract fresh features).
    logger.info("Running final evaluation...")
    evaluator.reset_cache()
    do_rerank = not args.no_rerank and cfg.eval.rerank
    results = evaluator.evaluate(rerank=do_rerank)

    logger.info("Final results:\n%s", format_results_table(results))

    # Save a standalone, inference-ready weights file alongside the checkpoints.
    final_weights = output_dir / "model_final.pth"
    save_model(
        model,
        final_weights,
        num_classes=num_classes,
        mAP=results["mAP"],
        rank1=results["rank1"],
        config=cfg.to_dict(),
    )
    logger.info("Saved final model weights to %s", final_weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
