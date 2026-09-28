r"""Evaluate a trained Person Re-Identification model on Market-1501.

This CLI rebuilds a model from a checkpoint, builds the query and gallery
loaders, runs the :class:`reid.evaluation.evaluator.Evaluator`, and prints a
table of mAP, Rank-1, Rank-5 and Rank-10, optionally with the k-reciprocal
re-ranked row as well.

The architecture and input size come from the config embedded in the
checkpoint (see :func:`reid.models.reid_model.load_trained_model`), so a
checkpoint is always evaluated with the network it was trained with. The
``--config`` file supplies everything else, such as the evaluation protocol and
batch size. Both trainer checkpoints (``best.pth``, ``last.pth``) and exported
weights (``model_final.pth``) are accepted, and they are loaded with PyTorch's
safe ``weights_only`` unpickler.

Example:
    Evaluate the final weights with re-ranking::

        reid-evaluate \
            --weights outputs/strong_baseline/model_final.pth \
            --data-root /path/to/Market-1501-v15.09.15 \
            --rerank
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from reid.cli import load_cli_config, resolve_data_root
from reid.data.build import build_dataloaders
from reid.evaluation.evaluator import Evaluator
from reid.evaluation.reporting import format_results_table
from reid.models.reid_model import load_trained_model
from reid.utils.device import resolve_device
from reid.utils.logging import setup_logger
from reid.utils.reproducibility import set_seed

logger = logging.getLogger("reid.cli.evaluate")


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
        default=None,
        help="YAML configuration file. Defaults to the packaged strong-baseline config.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Trained checkpoint or exported weights (.pth).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Market-1501 data root. Overrides data.root and the REID_DATA_ROOT variable.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('auto', 'cuda', 'mps' or 'cpu'). Overrides train.device.",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Optional path where the metrics (without the full CMC arrays) are saved.",
    )
    rerank_group = parser.add_mutually_exclusive_group()
    rerank_group.add_argument(
        "--rerank",
        dest="rerank",
        action="store_true",
        help="Also report k-reciprocal re-ranked metrics (default follows eval.rerank).",
    )
    rerank_group.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="Skip k-reciprocal re-ranking.",
    )
    parser.set_defaults(rerank=None)
    return parser


def scalar_metrics(results: dict[str, object]) -> dict[str, float]:
    """Keep only the scalar metrics of an evaluation result.

    Args:
        results: A dict returned by :meth:`Evaluator.evaluate`.

    Returns:
        The float-valued entries, without the full CMC arrays.
    """
    return {k: float(v) for k, v in results.items() if isinstance(v, (int, float))}


def main(argv: list[str] | None = None) -> int:
    """Load a model and evaluate it on Market-1501.

    Args:
        argv: Optional list of command-line arguments (defaults to
            ``sys.argv[1:]``).

    Returns:
        Process exit code, ``0`` on success and ``1`` on a usage or load error.
    """
    args = build_parser().parse_args(argv)
    setup_logger("reid")

    cfg = load_cli_config(args.config, logger)
    if cfg is None:
        return 1
    if args.device is not None:
        cfg.train.device = args.device
    data_root = resolve_data_root(cfg, args.data_root, logger)
    if data_root is None:
        return 1
    if not args.weights.is_file():
        logger.error("Weights file not found: %s", args.weights)
        return 1

    try:
        model, cfg = load_trained_model(args.weights, cfg)
    except (RuntimeError, ValueError, OSError) as exc:
        logger.error("Could not load %s: %s", args.weights, exc)
        return 1

    set_seed(cfg.train.seed, deterministic=cfg.train.deterministic)
    device = resolve_device(cfg.train.device)
    model = model.to(device).eval()

    loaders = build_dataloaders(cfg, root=data_root, include_train=False)
    evaluator = Evaluator(
        model=model,
        query_loader=loaders["query_loader"],
        gallery_loader=loaders["gallery_loader"],
        device=device,
        cfg=cfg,
    )
    results = evaluator.evaluate(rerank=args.rerank)
    logger.info("Evaluation results:\n%s", format_results_table(results))

    if args.results_json is not None:
        args.results_json.parent.mkdir(parents=True, exist_ok=True)
        args.results_json.write_text(
            json.dumps(scalar_metrics(results), indent=2) + "\n", encoding="utf-8"
        )
        logger.info("Saved metrics to %s", args.results_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
