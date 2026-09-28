r"""Generate qualitative and quantitative Re-ID analysis figures.

This CLI loads a trained model and writes the figures used in the project
README to the ``--output`` directory (default ``docs/images``). They are the
CMC curve, the per-query AP histogram, the cross-camera Rank-1 heatmap, the
intra-class versus inter-class distance distributions, a t-SNE projection of
the embedding space, a top-k ranked retrieval gallery, Rank-1 success and
failure panels, a before-versus-after re-ranking comparison, and Grad-CAM
attention overlays.

Every figure is computed from one feature extraction that follows the
evaluation protocol in the config (flip test-time augmentation and L2
normalization), so the qualitative panels agree with the reported metrics.
Random choices are seeded by ``--seed`` (default ``train.seed``), so reruns
reproduce the same figures.

The plotting libraries come from the ``viz`` extra
(``pip install "secondsight[viz]"``). A figure whose dependency is missing, for
example Grad-CAM without OpenCV or t-SNE without scikit-learn, is skipped with
a logged warning, and the camera heatmap falls back to plain matplotlib when
seaborn is absent. A figure that fails for any other reason is logged and the
remaining figures are still written.

Example:
    Render every figure for a trained checkpoint::

        reid-visualize \
            --weights outputs/strong_baseline/model_final.pth \
            --data-root /path/to/Market-1501-v15.09.15 \
            --output docs/images
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from reid.cli import load_cli_config, resolve_data_root
from reid.data.build import build_dataloaders
from reid.evaluation.evaluator import extract_features
from reid.evaluation.metrics import compute_ap_per_query, compute_cmc_map
from reid.evaluation.rerank import re_ranking
from reid.models.reid_model import build_model, load_trained_model
from reid.utils.device import resolve_device
from reid.utils.distance import compute_distance_matrix
from reid.utils.logging import setup_logger
from reid.utils.reproducibility import set_seed

_DEFAULT_OUTPUT = "docs/images"

logger = logging.getLogger("reid.cli.visualize")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the visualization CLI.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="reid-visualize",
        description="Generate Grad-CAM, CMC, t-SNE and ranking figures for a trained Re-ID model.",
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
        default=None,
        help=(
            "Trained checkpoint or exported weights (.pth). When omitted a randomly "
            "initialized model is used, which only makes sense for a smoke test."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Market-1501 data root. Overrides data.root and the REID_DATA_ROOT variable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(_DEFAULT_OUTPUT),
        help="Directory where the figures are written.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device ('auto', 'cuda', 'mps' or 'cpu'). Overrides train.device.",
    )
    parser.add_argument(
        "--num-gradcam",
        type=int,
        default=5,
        help="Number of query images to render Grad-CAM overlays for.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the sampled queries and identities. Defaults to train.seed.",
    )
    rerank_group = parser.add_mutually_exclusive_group()
    rerank_group.add_argument(
        "--rerank",
        dest="rerank",
        action="store_true",
        help="Draw the before-versus-after re-ranking figure (default follows eval.rerank).",
    )
    rerank_group.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="Skip the re-ranking figure.",
    )
    parser.set_defaults(rerank=None)
    return parser


def _safe_plot(name: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
    """Run one plotting function without letting its failure stop the others.

    Args:
        name: Human-readable name of the figure, used in log messages.
        func: The plotting callable to invoke.
        *args: Positional arguments forwarded to ``func``.
        **kwargs: Keyword arguments forwarded to ``func``.

    Returns:
        ``True`` when the figure was produced, ``False`` when it was skipped or
        failed.
    """
    try:
        figure = func(*args, **kwargs)
    except ImportError as exc:
        logger.warning("Skipping %s (missing dependency): %s", name, exc)
        return False
    except Exception:  # noqa: BLE001 (keep generating the remaining figures)
        logger.exception("Failed to generate the %s figure.", name)
        return False
    if figure is None:
        logger.warning("Nothing to draw for the %s figure; it was not written.", name)
        return False
    logger.info("Saved the %s figure.", name)
    return True


def main(argv: list[str] | None = None) -> int:
    r"""Generate the full analysis figure suite for a trained model.

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

    if args.weights is not None:
        if not args.weights.is_file():
            logger.error("Weights file not found: %s", args.weights)
            return 1
        try:
            model, cfg = load_trained_model(args.weights, cfg)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.error("Could not load %s: %s", args.weights, exc)
            return 1
        logger.info("Loaded weights from %s", args.weights)
    else:
        logger.warning("No --weights given; using a randomly initialized model.")
        model = build_model(cfg, num_classes=1, pretrained=False)

    seed = cfg.train.seed if args.seed is None else args.seed
    rerank = cfg.eval.rerank if args.rerank is None else args.rerank
    set_seed(seed, deterministic=cfg.train.deterministic)
    device = resolve_device(cfg.train.device)
    model = model.to(device).eval()

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing figures to %s", output_dir.resolve())

    loaders = build_dataloaders(cfg, root=data_root, include_train=False)
    query_set = loaders["query_loader"].dataset
    gallery_set = loaders["gallery_loader"].dataset

    # Extract features once, following the evaluation protocol, and reuse them
    # for every distance-based figure.
    logger.info("Extracting query and gallery features...")
    qf, q_pids, q_camids = extract_features(
        model,
        loaders["query_loader"],
        device,
        flip_tta=cfg.eval.flip_tta,
        feat_norm=cfg.eval.feat_norm,
    )
    gf, g_pids, g_camids = extract_features(
        model,
        loaders["gallery_loader"],
        device,
        flip_tta=cfg.eval.flip_tta,
        feat_norm=cfg.eval.feat_norm,
    )
    distmat = compute_distance_matrix(qf, gf, metric="euclidean").cpu().numpy()
    ids = (q_pids, g_pids, q_camids, g_camids)

    cmc, mean_ap = compute_cmc_map(distmat, *ids, max_rank=cfg.eval.max_rank)
    aps = compute_ap_per_query(distmat, *ids)
    logger.info("mAP=%.2f%% | Rank-1=%.2f%%", 100.0 * mean_ap, 100.0 * float(cmc[0]))

    from reid.visualization import analysis, gradcam, ranking

    _safe_plot("CMC curve", analysis.plot_cmc_curve, cmc, save_path=output_dir / "cmc_curve.png")
    _safe_plot(
        "AP distribution",
        analysis.plot_ap_distribution,
        aps,
        save_path=output_dir / "ap_distribution.png",
    )
    _safe_plot(
        "camera heatmap",
        analysis.plot_camera_heatmap,
        distmat,
        *ids,
        save_path=output_dir / "camera_heatmap.png",
    )
    _safe_plot(
        "distance distributions",
        analysis.plot_distance_distributions,
        distmat,
        *ids,
        save_path=output_dir / "distance_distributions.png",
        seed=seed,
    )
    _safe_plot(
        "t-SNE",
        analysis.plot_tsne,
        qf.numpy(),
        q_pids,
        save_path=output_dir / "tsne.png",
        seed=seed,
    )
    _safe_plot(
        "ranked results",
        ranking.visualize_ranked_results,
        distmat,
        query_set,
        gallery_set,
        *ids,
        save_path=output_dir / "ranked_results.png",
        seed=seed,
    )
    _safe_plot(
        "success/failure",
        ranking.plot_success_failure,
        distmat,
        query_set,
        gallery_set,
        *ids,
        save_path=output_dir / "success_failure.png",
        seed=seed,
    )
    if rerank:
        logger.info("Running k-reciprocal re-ranking for the before/after figure...")
        rr_distmat = re_ranking(
            qf,
            gf,
            k1=cfg.eval.rerank_k1,
            k2=cfg.eval.rerank_k2,
            lambda_value=cfg.eval.rerank_lambda,
        )
        _safe_plot(
            "re-ranking impact",
            ranking.plot_rerank_impact,
            distmat,
            rr_distmat,
            query_set,
            gallery_set,
            *ids,
            save_path=output_dir / "rerank_impact.png",
            seed=seed,
        )
    _safe_plot(
        "Grad-CAM",
        gradcam.plot_gradcam_samples,
        model,
        query_set,
        args.num_gradcam,
        save_path=output_dir / "gradcam.png",
        device=device,
        seed=seed,
    )

    logger.info("All figures written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
