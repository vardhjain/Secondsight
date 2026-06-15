"""Interactive Gradio demo: probe-vs-gallery person re-identification.

This app lets a user upload a *probe* (query) person crop and retrieves the
most visually similar people from the Market-1501 gallery, ranked by cosine
similarity of their learned embeddings. It is designed to be runnable in a
recruiter-facing portfolio context with minimal friction:

* **No trained weights?** The app warns and falls back to a randomly
  initialized model, so the UI is still fully interactive (matches will simply
  be uninformative).
* **No dataset?** The app launches with a clear, in-UI message instructing the
  user to run ``reid-download`` and re-launch with ``--data-root`` (or the
  ``REID_DATA_ROOT`` environment variable).

Weights are resolved from ``--weights`` or the ``REID_WEIGHTS`` environment
variable; the dataset root from ``--data-root`` or ``REID_DATA_ROOT``.

``gradio`` is imported lazily inside :func:`build_demo` / :func:`main` so that
importing this module (e.g. for testing helper functions) does not require the
dependency.

Example:
    Launch the demo against a trained checkpoint::

        python -m app.gradio_app \\
            --weights outputs/strong_baseline/best.pth \\
            --data-root /path/to/Market-1501-v15.09.15
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.nn.functional import normalize

from reid.config import Config
from reid.data.dataset import Market1501
from reid.data.transforms import build_transforms
from reid.models.reid_model import build_model
from reid.utils.checkpoint import infer_num_classes_from_checkpoint, load_model
from reid.utils.device import resolve_device
from reid.utils.logging import setup_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import gradio as gr
    from PIL import Image

_DEFAULT_CONFIG = "configs/market1501_strong_baseline.yaml"
_ENV_WEIGHTS = "REID_WEIGHTS"
_ENV_DATA_ROOT = "REID_DATA_ROOT"
# Cap the indexed gallery so the demo stays responsive on CPU.
_DEFAULT_GALLERY_LIMIT = 2000

logger = logging.getLogger("reid.app.gradio")


class ReIDDemoEngine:
    """Backend for the Gradio demo: model, gallery index and search.

    The engine loads (or randomly initializes) the model, optionally indexes a
    subset of the Market-1501 gallery by extracting L2-normalized embeddings,
    and answers nearest-neighbor queries for an uploaded probe image.

    Attributes:
        cfg: The experiment configuration.
        device: Device used for inference.
        model: The Re-ID model (trained or randomly initialized).
        has_weights: Whether trained weights were successfully loaded.
        gallery: The indexed gallery dataset, or ``None`` if unavailable.
        gallery_features: L2-normalized gallery embeddings, or ``None``.
        status_message: A human-readable description of the engine's state.
    """

    def __init__(
        self,
        cfg: Config,
        weights: Path | None,
        data_root: Path | None,
        device: torch.device,
        gallery_limit: int = _DEFAULT_GALLERY_LIMIT,
    ) -> None:
        """Initialize the engine, loading weights and indexing the gallery.

        Args:
            cfg: The experiment configuration.
            weights: Optional path to trained weights. ``None`` or a missing
                file triggers a random-initialization fallback.
            data_root: Optional path to the Market-1501 data root. ``None`` or a
                missing directory disables gallery search.
            device: Device on which to run inference.
            gallery_limit: Maximum number of gallery images to index.
        """
        self.cfg = cfg
        self.device = device
        self.transform = build_transforms(cfg.data, is_train=False)

        self.gallery: Market1501 | None = None
        self.gallery_features: torch.Tensor | None = None
        self.has_weights = False
        self.status_message = ""

        num_classes = self._index_gallery(data_root, gallery_limit)
        self._build_model(weights, num_classes)
        if self.gallery is not None:
            self._extract_gallery_features()

        self.status_message = self._compose_status()

    def _index_gallery(self, data_root: Path | None, gallery_limit: int) -> int:
        """Load the gallery dataset if a valid data root is provided.

        Args:
            data_root: Path to the Market-1501 data root, or ``None``.
            gallery_limit: Maximum number of gallery images to index.

        Returns:
            A class count usable for sizing the classifier head (the gallery's
            unique identity count, or a small positive default when no gallery
            is available). The classifier is unused at inference time.
        """
        if data_root is None:
            logger.warning("No data root provided; gallery search is disabled.")
            return 1
        if not data_root.is_dir():
            logger.warning("Data root %s not found; gallery search is disabled.", data_root)
            return 1
        try:
            gallery = Market1501(data_root, subset="gallery", transform=self.transform)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Failed to load gallery from %s: %s", data_root, exc)
            return 1

        # Optionally subsample to keep the demo responsive.
        if gallery_limit and len(gallery) > gallery_limit:
            rng = np.random.default_rng(42)
            keep = sorted(rng.choice(len(gallery), size=gallery_limit, replace=False).tolist())
            gallery.img_paths = [gallery.img_paths[i] for i in keep]
            gallery.pids = [gallery.pids[i] for i in keep]
            gallery.camids = [gallery.camids[i] for i in keep]
            logger.info("Subsampled gallery to %d images for the demo.", len(gallery))

        self.gallery = gallery
        return max(1, gallery.num_classes)

    def _build_model(self, weights: Path | None, num_classes: int) -> None:
        """Build the model and load weights if available.

        Args:
            weights: Optional path to trained weights.
            num_classes: Classifier size (unused at inference but required to
                construct the head).
        """
        # If a checkpoint is available, size the classifier from it so weights
        # load cleanly.
        resolved_classes = infer_num_classes_from_checkpoint(weights, fallback=num_classes)
        model = build_model(self.cfg, num_classes=resolved_classes)

        if weights is not None and weights.exists():
            try:
                load_model(model, weights, map_location="cpu")
                self.has_weights = True
                logger.info("Loaded model weights from %s", weights)
            except Exception:  # noqa: BLE001 - never crash the demo on bad weights
                logger.exception("Failed to load weights from %s; using random init.", weights)
        else:
            if weights is not None:
                logger.warning("Weights file %s not found; using random initialization.", weights)
            else:
                logger.warning("No weights provided; using random initialization.")

        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def _embed(self, image: Image.Image) -> torch.Tensor:
        """Embed a single PIL image into an L2-normalized feature vector.

        Args:
            image: An RGB ``PIL.Image``.

        Returns:
            A CPU tensor of shape ``(1, feat_dim)`` (L2-normalized).
        """
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        feat = self.model.extract_features(tensor)
        if self.cfg.eval.feat_norm:
            feat = normalize(feat, p=2, dim=1)
        return feat.cpu()

    @torch.no_grad()
    def _extract_gallery_features(self) -> None:
        """Extract and cache L2-normalized embeddings for the indexed gallery."""
        assert self.gallery is not None
        from torch.utils.data import DataLoader

        loader = DataLoader(
            self.gallery,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=0,
        )
        feats: list[torch.Tensor] = []
        for images, _pids, _camids in loader:
            images = images.to(self.device)
            batch_feat = self.model.extract_features(images)
            if self.cfg.eval.feat_norm:
                batch_feat = normalize(batch_feat, p=2, dim=1)
            feats.append(batch_feat.cpu())
        self.gallery_features = torch.cat(feats, dim=0)
        logger.info("Indexed %d gallery embeddings.", self.gallery_features.shape[0])

    def _compose_status(self) -> str:
        """Build the human-readable status banner shown in the UI.

        Returns:
            A Markdown status string describing weight and gallery readiness.
        """
        parts: list[str] = []
        if self.has_weights:
            parts.append("Model: trained weights loaded.")
        else:
            parts.append(
                "Model: **random initialization** (no valid weights) - "
                "matches will be uninformative. Pass `--weights` or set "
                "`REID_WEIGHTS` to a trained checkpoint."
            )
        if self.gallery is not None and self.gallery_features is not None:
            parts.append(f"Gallery: {self.gallery_features.shape[0]} images indexed.")
        else:
            parts.append(
                "Gallery: **not available**. Run `reid-download` to fetch "
                "Market-1501, then relaunch with `--data-root` (or set "
                "`REID_DATA_ROOT`)."
            )
        return "  \n".join(parts)

    @torch.no_grad()
    def search(self, image: Image.Image | None, topk: int) -> tuple[list[tuple[Any, str]], str]:
        """Retrieve the top-k most similar gallery images for a probe.

        Args:
            image: The uploaded probe image, or ``None`` if nothing was given.
            topk: Number of matches to return.

        Returns:
            A tuple ``(gallery_items, message)`` where ``gallery_items`` is a
            list of ``(PIL.Image, caption)`` pairs suitable for a Gradio gallery
            component and ``message`` is a status string.
        """
        if image is None:
            return [], "Please upload a probe image."
        if self.gallery is None or self.gallery_features is None:
            return [], (
                "Gallery is not available. Run `reid-download` and relaunch with "
                "`--data-root` (or set `REID_DATA_ROOT`)."
            )

        from PIL import Image as PILImage

        probe_feat = self._embed(image)
        # Cosine similarity == dot product of L2-normalized features.
        sims = (probe_feat @ self.gallery_features.t()).squeeze(0).numpy()
        topk = int(max(1, min(topk, len(self.gallery))))
        order = np.argsort(-sims)[:topk]

        items: list[tuple[Any, str]] = []
        for rank, gid in enumerate(order, start=1):
            path = self.gallery.img_paths[int(gid)]
            pid = self.gallery.pids[int(gid)]
            camid = self.gallery.camids[int(gid)]
            caption = f"#{rank} | ID {pid} | Cam {camid} | sim {sims[int(gid)]:.3f}"
            items.append((PILImage.open(path).convert("RGB"), caption))

        suffix = "" if self.has_weights else " (random-init model - results are not meaningful)"
        return items, f"Showing top-{topk} matches by cosine similarity.{suffix}"


def build_demo(engine: ReIDDemoEngine) -> gr.Blocks:
    """Build the Gradio Blocks UI bound to a demo engine.

    Args:
        engine: A constructed :class:`ReIDDemoEngine`.

    Returns:
        A :class:`gradio.Blocks` application ready to ``launch``.
    """
    import gradio as gr

    with gr.Blocks(title="Person Re-ID - Market-1501") as demo:
        gr.Markdown(
            "# Person Re-Identification Demo\n"
            "Upload a person crop (the *probe*) and retrieve the most similar "
            "people from the Market-1501 gallery, ranked by cosine similarity "
            "of their learned embeddings."
        )
        gr.Markdown(engine.status_message)

        with gr.Row():
            with gr.Column(scale=1):
                probe = gr.Image(type="pil", label="Probe image", height=320)
                topk = gr.Slider(
                    minimum=1, maximum=20, value=10, step=1, label="Number of matches (top-k)"
                )
                search_btn = gr.Button("Search gallery", variant="primary")
            with gr.Column(scale=2):
                results = gr.Gallery(
                    label="Top matches",
                    columns=5,
                    height=420,
                    object_fit="contain",
                )
                message = gr.Markdown()

        def _on_search(img: Image.Image | None, k: int) -> tuple[list[tuple[Any, str]], str]:
            return engine.search(img, int(k))

        search_btn.click(fn=_on_search, inputs=[probe, topk], outputs=[results, message])
        probe.upload(fn=_on_search, inputs=[probe, topk], outputs=[results, message])

    return demo


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the Gradio app.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Launch the interactive probe-vs-gallery Re-ID demo.",
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
        help=f"Path to trained weights (.pth). Falls back to the {_ENV_WEIGHTS} env var.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"Path to the Market-1501 data root. Falls back to the {_ENV_DATA_ROOT} env var.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device, e.g. 'cuda' or 'cpu'. Overrides train.device.",
    )
    parser.add_argument(
        "--gallery-limit",
        type=int,
        default=_DEFAULT_GALLERY_LIMIT,
        help="Maximum number of gallery images to index (0 = no limit).",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind the Gradio server to.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Port for the Gradio server.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link.",
    )
    return parser


def _resolve_path(cli_value: Path | None, env_var: str) -> Path | None:
    """Resolve a path from a CLI value, falling back to an environment variable.

    Args:
        cli_value: The value supplied on the command line, or ``None``.
        env_var: Name of the environment variable to fall back to.

    Returns:
        The resolved :class:`~pathlib.Path`, or ``None`` if neither is set.
    """
    if cli_value is not None:
        return cli_value
    env_value = os.environ.get(env_var)
    return Path(env_value) if env_value else None


def main(argv: list[str] | None = None) -> int:
    """Build and launch the Gradio demo.

    Args:
        argv: Optional list of command-line arguments (defaults to
            ``sys.argv[1:]``).

    Returns:
        Process exit code (``0`` on a clean shutdown).
    """
    args = build_parser().parse_args(argv)
    setup_logger("reid")

    if not args.config.is_file():
        logger.error("Config file not found: %s", args.config)
        return 1
    cfg = Config.from_yaml(args.config)
    if args.device is not None:
        cfg.train.device = args.device
    device = resolve_device(cfg.train.device)

    weights = _resolve_path(args.weights, _ENV_WEIGHTS)
    data_root = _resolve_path(args.data_root, _ENV_DATA_ROOT)
    if data_root is None and cfg.data.root is not None:
        data_root = Path(cfg.data.root)

    engine = ReIDDemoEngine(
        cfg=cfg,
        weights=weights,
        data_root=data_root,
        device=device,
        gallery_limit=args.gallery_limit,
    )
    logger.info(engine.status_message.replace("**", ""))

    demo = build_demo(engine)
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
