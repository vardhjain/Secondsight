"""Evaluate a trained Person Re-Identification model on Market-1501.

This CLI loads a model from a checkpoint / weights file, builds the query and
gallery loaders, runs the :class:`reid.evaluation.evaluator.Evaluator`, and
prints a mAP / Rank-1 / Rank-5 / Rank-10 results table -- optionally including
the k-reciprocal re-ranked variant.

The weights file may be either a standalone model file produced by
:func:`reid.utils.checkpoint.save_model` (a ``{"state_dict": ...}`` mapping) or
a full training checkpoint produced by the trainer (a ``{"model": ...}``
mapping); :func:`reid.utils.checkpoint.load_model` handles both.

Example:
    Evaluate the best checkpoint with re-ranking::

        python -m scripts.evaluate \\
            --config configs/market1501_strong_baseline.yaml \\
            --weights outputs/strong_baseline/best.pth \\
            --data-root /path/to/Market-1501-v15.09.15 \\
            --rerank
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from reid.config import Config
from reid.data.build import build_dataloaders
from reid.evaluation.evaluator import Evaluator
from reid.evaluation.reporting import format_results_table
from reid.models.reid_model import build_model
from reid.utils.checkpoint import infer_num_classes_from_checkpoint, load_model
from reid.utils.device import resolve_device
from reid.utils.logging import setup_logger
from reid.utils.reproducibility import set_seed

_DEFAULT_CONFIG = "configs/market1501_strong_baseline.yaml"

logger = logging.getLogger("reid.scripts.evaluate")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the evaluation CLI.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="reid-evaluate",
        description="Evaluate a trained Re-ID model on the Market-1501 query/gallery split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(_DEFAULT_CONFIG),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to the trained model weights / checkpoint (.pth).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to the Market-1501 data root. Overrides data.root from the config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device, e.g. 'cuda' or 'cpu'. Overrides train.device.",
    )
    rerank_group = parser.add_mutually_exclusive_group()
    rerank_group.add_argument(
        "--rerank",
        dest="rerank",
        action="store_true",
        help="Enable k-reciprocal re-ranking (default follows the config).",
    )
    rerank_group.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="Disable k-reciprocal re-ranking.",
    )
    parser.set_defaults(rerank=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load a model and evaluate it on Market-1501.

    Args:
        argv: Optional list of command-line arguments (defaults to
            ``sys.argv[1:]``).

    Returns:
        Process exit code: ``0`` on success, non-zero on failure.
    """
    args = build_parser().parse_args(argv)
    setup_logger("reid")

    if not args.config.is_file():
        logger.error("Config file not found: %s", args.config)
        return 1
    cfg = Config.from_yaml(args.config)
    if args.data_root is not None:
        cfg.data.root = str(args.data_root)
    if args.device is not None:
        cfg.train.device = args.device

    if cfg.data.root is None:
        env_root = os.environ.get("REID_DATA_ROOT")
        if env_root:
            cfg.data.root = env_root
    if cfg.data.root is None:
        logger.error(
            "No dataset root configured. Pass --data-root, set data.root in the "
            "config, or set the REID_DATA_ROOT environment variable."
        )
        return 1

    data_root = Path(cfg.data.root)
    if not data_root.is_dir():
        logger.error("Dataset root does not exist: %s", data_root)
        return 1

    if not args.weights.exists():
        logger.error("Weights file not found: %s", args.weights)
        return 1

    set_seed(cfg.train.seed, deterministic=True)
    device = resolve_device(cfg.train.device)

    loaders = build_dataloaders(cfg, root=data_root)
    # Prefer the classifier size baked into the checkpoint so weights load
    # cleanly even though the gallery split alone cannot tell us the training
    # identity count.
    num_classes = infer_num_classes_from_checkpoint(
        args.weights, fallback=int(loaders["num_classes"])
    )

    model = build_model(cfg, num_classes=num_classes)
    load_model(model, args.weights, map_location="cpu")
    model = model.to(device)
    model.eval()

    evaluator = Evaluator(
        model=model,
        query_loader=loaders["query_loader"],
        gallery_loader=loaders["gallery_loader"],
        device=device,
        cfg=cfg,
    )

    results = evaluator.evaluate(rerank=args.rerank)
    logger.info("Evaluation results:\n%s", format_results_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
