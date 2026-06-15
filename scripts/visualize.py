"""Generate qualitative and quantitative Re-ID analysis figures.

This CLI loads a trained model and produces a suite of figures used in the
project report / README:

* **Grad-CAM** attention overlays on a handful of query images,
* the **CMC curve**,
* the per-query **AP histogram**,
* the cross-camera **Rank-1 heatmap** (angle analysis),
* a **t-SNE** projection of the embedding space,
* curated **success vs. failure** Rank-1 panels, and
* intra- vs. inter-class **distance distributions**.

All figures are written to the ``--output`` directory (default
``docs/images``). Heavy/optional dependencies (``cv2`` for Grad-CAM,
``scikit-learn`` for t-SNE, ``seaborn`` for the heatmap) are imported lazily by
the underlying visualization functions, and individual figures are skipped with
a logged warning if a dependency is missing, so the rest of the suite still
runs.

Example:
    Render every figure for a trained checkpoint::

        python -m scripts.visualize \\
            --config configs/market1501_strong_baseline.yaml \\
            --weights outputs/strong_baseline/best.pth \\
            --data-root /path/to/Market-1501-v15.09.15 \\
            --output docs/images
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch

from reid.config import Config
from reid.data.dataset import Market1501
from reid.data.transforms import build_transforms
from reid.evaluation.evaluator import extract_features
from reid.evaluation.metrics import compute_ap_per_query, compute_cmc_map
from reid.models.reid_model import build_model
from reid.utils.checkpoint import infer_num_classes_from_checkpoint, load_model
from reid.utils.device import resolve_device
from reid.utils.distance import compute_distance_matrix
from reid.utils.logging import setup_logger
from reid.utils.reproducibility import set_seed

_DEFAULT_CONFIG = "configs/market1501_strong_baseline.yaml"
_DEFAULT_OUTPUT = "docs/images"

logger = logging.getLogger("reid.scripts.visualize")


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
        default=Path(_DEFAULT_CONFIG),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=(
            "Path to trained model weights (.pth). If omitted a randomly "
            "initialized model is used (figures will be uninformative)."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to the Market-1501 data root. Overrides data.root from the config.",
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
        help="Compute device, e.g. 'cuda' or 'cpu'. Overrides train.device.",
    )
    parser.add_argument(
        "--num-gradcam",
        type=int,
        default=5,
        help="Number of query images to render Grad-CAM overlays for.",
    )
    return parser


def _generate_gradcam(
    model: torch.nn.Module,
    query_set: Market1501,
    device: torch.device,
    output_dir: Path,
    num_samples: int,
) -> None:
    """Render Grad-CAM attention overlays for a sample of query images.

    Args:
        model: The trained Re-ID model.
        query_set: Query dataset (yielding normalized tensors).
        device: Device on which to run the forward/backward passes.
        output_dir: Directory to write the figure into.
        num_samples: Number of query images to visualize.
    """
    try:
        import matplotlib.pyplot as plt

        from reid.visualization.gradcam import GradCAM, overlay_heatmap
        from reid.visualization.ranking import _denormalize_to_image
    except ImportError as exc:  # pragma: no cover - optional dependency (cv2)
        logger.warning("Skipping Grad-CAM (missing dependency): %s", exc)
        return

    num_samples = min(num_samples, len(query_set))
    if num_samples <= 0:
        return

    rng = np.random.default_rng(0)
    indices = rng.choice(len(query_set), size=num_samples, replace=False)

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2.4, 5.2))
    axes = np.atleast_2d(axes)

    target_layer = model.backbone.layer4
    with GradCAM(model, target_layer) as cam:
        for col, idx in enumerate(indices):
            img_t, pid, camid = query_set[int(idx)]
            input_tensor = img_t.unsqueeze(0).to(device)
            heatmap = cam(input_tensor)

            original = _denormalize_to_image(img_t)
            overlay = overlay_heatmap(original, heatmap, alpha=0.5)

            axes[0, col].imshow(original)
            axes[0, col].set_title(f"ID {int(pid)}\nCam {int(camid)}", fontsize=9)
            axes[0, col].axis("off")

            axes[1, col].imshow(overlay)
            axes[1, col].set_title("Grad-CAM", fontsize=9)
            axes[1, col].axis("off")

    fig.suptitle("Grad-CAM Attention", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_path = output_dir / "gradcam.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Grad-CAM figure to %s", save_path)


def _safe_plot(name: str, func, *args, **kwargs) -> None:
    """Run a plotting function, logging and swallowing optional-dep failures.

    Args:
        name: Human-readable name of the figure (for logging).
        func: The plotting callable to invoke.
        *args: Positional arguments forwarded to ``func``.
        **kwargs: Keyword arguments forwarded to ``func``.
    """
    try:
        func(*args, **kwargs)
        logger.info("Saved %s figure.", name)
    except ImportError as exc:  # pragma: no cover - optional dependency
        logger.warning("Skipping %s (missing dependency): %s", name, exc)
    except Exception:  # noqa: BLE001 - keep generating the remaining figures
        logger.exception("Failed to generate %s figure.", name)


def main(argv: list[str] | None = None) -> int:
    """Generate the full analysis figure suite for a trained model.

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

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.train.seed, deterministic=True)
    device = resolve_device(cfg.train.device)

    # Build datasets with the test transform (deterministic, no augmentation).
    test_transform = build_transforms(cfg.data, is_train=False)
    query_set = Market1501(data_root, subset="query", transform=test_transform)
    gallery_set = Market1501(data_root, subset="gallery", transform=test_transform)

    # The classifier size is irrelevant for feature extraction; size it from the
    # checkpoint (if any) so weights load cleanly, else fall back to the gallery
    # identity count.
    num_classes = infer_num_classes_from_checkpoint(
        args.weights, fallback=max(1, gallery_set.num_classes)
    )
    model = build_model(cfg, num_classes=num_classes)

    if args.weights is not None and args.weights.exists():
        load_model(model, args.weights, map_location="cpu")
        logger.info("Loaded weights from %s", args.weights)
    else:
        if args.weights is not None:
            logger.warning("Weights file %s not found; using random initialization.", args.weights)
        else:
            logger.warning("No --weights provided; using random initialization.")
    model = model.to(device)
    model.eval()

    from torch.utils.data import DataLoader

    query_loader = DataLoader(
        query_set, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers
    )
    gallery_loader = DataLoader(
        gallery_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    # Extract features once and reuse them for every distance-based figure.
    logger.info("Extracting query features...")
    qf, q_pids, q_camids = extract_features(
        model, query_loader, device, flip_tta=cfg.eval.flip_tta, feat_norm=cfg.eval.feat_norm
    )
    logger.info("Extracting gallery features...")
    gf, g_pids, g_camids = extract_features(
        model, gallery_loader, device, flip_tta=cfg.eval.flip_tta, feat_norm=cfg.eval.feat_norm
    )

    distmat = compute_distance_matrix(qf, gf, metric="euclidean").cpu().numpy()

    cmc, mean_ap = compute_cmc_map(
        distmat, q_pids, g_pids, q_camids, g_camids, max_rank=cfg.eval.max_rank
    )
    aps = compute_ap_per_query(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info("mAP=%.2f%% | Rank-1=%.2f%%", 100.0 * mean_ap, 100.0 * float(cmc[0]))

    # Import the analysis plotters lazily (matplotlib only at this point).
    from reid.visualization import analysis, ranking

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
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        save_path=output_dir / "camera_heatmap.png",
    )
    _safe_plot(
        "distance distributions",
        analysis.plot_distance_distributions,
        distmat,
        q_pids,
        g_pids,
        save_path=output_dir / "distance_distributions.png",
    )
    _safe_plot(
        "t-SNE",
        analysis.plot_tsne,
        qf.numpy(),
        q_pids,
        save_path=output_dir / "tsne.png",
    )
    _safe_plot(
        "ranked results",
        ranking.visualize_ranked_results,
        distmat,
        query_set,
        gallery_set,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        save_path=output_dir / "ranked_results.png",
    )
    _safe_plot(
        "success/failure",
        ranking.plot_success_failure,
        model,
        query_set,
        gallery_set,
        device,
        save_path=output_dir / "success_failure.png",
    )

    _generate_gradcam(model, query_set, device, output_dir, args.num_gradcam)

    logger.info("All figures written to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
