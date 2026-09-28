"""Interactive Gradio demo: probe-vs-gallery person re-identification.

This app lets a user upload a *probe* (query) person crop and retrieves the
most visually similar people from the Market-1501 gallery, ranked by cosine
similarity of their learned embeddings. It is designed to launch with minimal
setup, even when weights or data are missing.

* **No trained weights?** The app warns and falls back to a randomly
  initialized model, so the UI is still fully interactive (matches will simply
  be uninformative).
* **No dataset?** The app launches with a clear, in-UI message instructing the
  user to run ``reid-download`` and re-launch with ``--data-root`` (or the
  ``REID_DATA_ROOT`` environment variable).

Weights are resolved from ``--weights`` or the ``REID_WEIGHTS`` environment
variable, and the dataset root from ``--data-root`` or ``REID_DATA_ROOT``. The
architecture and input size come from the config embedded in the checkpoint,
and embeddings use the same flip test-time augmentation and normalization as
the evaluation protocol. The server binds to ``--server-name`` and
``--server-port``, falling back to ``GRADIO_SERVER_NAME`` and
``GRADIO_SERVER_PORT`` and then to ``127.0.0.1:7860``, so the Docker image
(which sets ``GRADIO_SERVER_NAME=0.0.0.0``) is reachable from the host.

``gradio`` comes from the ``demo`` extra and is imported only inside
:func:`build_demo` and :func:`main`, so importing this module (for example to
test the helpers) does not require it.

Example:
    Launch the demo against a trained checkpoint::

        python -m app.gradio_app \\
            --weights outputs/strong_baseline/model_final.pth \\
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

from reid.config import Config, default_config_path
from reid.data.dataset import Market1501
from reid.data.transforms import build_transforms
from reid.models.reid_model import ReIDModel, build_model, load_trained_model
from reid.utils.device import resolve_device
from reid.utils.logging import setup_logger

if TYPE_CHECKING:  # pragma: no cover (typing only)
    import gradio as gr
    from PIL import Image

_ENV_WEIGHTS = "REID_WEIGHTS"
_ENV_DATA_ROOT = "REID_DATA_ROOT"
_ENV_SERVER_NAME = "GRADIO_SERVER_NAME"
_ENV_SERVER_PORT = "GRADIO_SERVER_PORT"
_DEFAULT_SERVER_NAME = "127.0.0.1"
_DEFAULT_SERVER_PORT = 7860
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
# Cap the indexed gallery so the demo stays responsive on CPU.
_DEFAULT_GALLERY_LIMIT = 2000

logger = logging.getLogger("reid.app.gradio")


class ReIDDemoEngine:
    """Backend for the Gradio demo: model, gallery index and search.

    The engine loads (or randomly initializes) the model, optionally indexes a
    subset of the Market-1501 gallery by extracting embeddings, and answers
    nearest-neighbor queries for an uploaded probe image.

    Attributes:
        cfg: The effective configuration. When trained weights are loaded, its
            model section and input size come from the checkpoint.
        device: Device used for inference.
        transform: The evaluation transform for the effective input size.
        model: The Re-ID model (trained or randomly initialized).
        has_weights: Whether trained weights were successfully loaded.
        gallery: The indexed gallery dataset, or ``None`` if unavailable.
        gallery_features: Gallery embeddings, or ``None``.
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
            cfg: The experiment configuration. It is not modified.
            weights: Optional path to trained weights. ``None`` or a missing
                file triggers a random-initialization fallback.
            data_root: Optional path to the Market-1501 data root. ``None`` or a
                missing directory disables gallery search.
            device: Device on which to run inference.
            gallery_limit: Maximum number of gallery images to index
                (``0`` means no limit).
        """
        self.cfg = cfg
        self.device = device
        self.gallery: Market1501 | None = None
        self.gallery_features: torch.Tensor | None = None
        self.has_weights = False
        self.status_message = ""

        # The model comes first because a checkpoint can change the input size,
        # which the transform and therefore the gallery depend on.
        self._build_model(weights)
        self.transform = build_transforms(self.cfg.data, is_train=False)
        self._index_gallery(data_root, gallery_limit)
        if self.gallery is not None:
            self._extract_gallery_features()

        self.status_message = self._compose_status()

    def _index_gallery(self, data_root: Path | None, gallery_limit: int) -> None:
        """Load the gallery dataset if a valid data root is provided.

        Args:
            data_root: Path to the Market-1501 data root, or ``None``.
            gallery_limit: Maximum number of gallery images to index.
        """
        if data_root is None:
            logger.warning("No data root provided; gallery search is disabled.")
            return
        if not data_root.is_dir():
            logger.warning("Data root %s not found; gallery search is disabled.", data_root)
            return
        try:
            gallery = Market1501(data_root, subset="gallery", transform=self.transform)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Failed to load gallery from %s: %s", data_root, exc)
            return

        # Optionally subsample to keep the demo responsive.
        if gallery_limit and len(gallery) > gallery_limit:
            rng = np.random.default_rng(42)
            keep = sorted(rng.choice(len(gallery), size=gallery_limit, replace=False).tolist())
            gallery.img_paths = [gallery.img_paths[i] for i in keep]
            gallery.pids = [gallery.pids[i] for i in keep]
            gallery.camids = [gallery.camids[i] for i in keep]
            logger.info("Subsampled gallery to %d images for the demo.", len(gallery))

        self.gallery = gallery

    def _build_model(self, weights: Path | None) -> None:
        """Load the trained model, or fall back to a randomly initialized one.

        Trained weights are loaded strictly through
        :func:`~reid.models.reid_model.load_trained_model`, which also adopts
        the architecture and input size stored in the checkpoint. The fallback
        model is built without ImageNet weights, so the demo never downloads
        anything and the status banner can honestly call it random.

        Args:
            weights: Optional path to trained weights.
        """
        model: ReIDModel | None = None
        if weights is not None and weights.is_file():
            try:
                model, self.cfg = load_trained_model(weights, self.cfg)
                self.has_weights = True
                logger.info("Loaded model weights from %s", weights)
            except Exception:  # noqa: BLE001 (never crash the demo on bad weights)
                logger.exception("Failed to load weights from %s; using random init.", weights)
        elif weights is not None:
            logger.warning("Weights file %s not found; using random initialization.", weights)
        else:
            logger.warning("No weights provided; using random initialization.")

        if model is None:
            # The classifier head is unused at inference, so its size is arbitrary.
            model = build_model(self.cfg, num_classes=1, pretrained=False)
        self.model = model.to(self.device).eval()

    def _features(self, images: torch.Tensor) -> torch.Tensor:
        """Embed a batch following the evaluation protocol in ``cfg.eval``.

        Args:
            images: Normalized image batch of shape ``[B, 3, H, W]`` on the
                engine's device.

        Returns:
            A CPU float tensor of shape ``[B, feat_dim]``. It is averaged with
            the features of the horizontally flipped batch when ``flip_tta`` is
            set and L2-normalized when ``feat_norm`` is set.
        """
        feat = self.model.extract_features(images).float()
        if self.cfg.eval.flip_tta:
            flipped = self.model.extract_features(torch.flip(images, dims=[3])).float()
            feat = (feat + flipped) / 2.0
        if self.cfg.eval.feat_norm:
            feat = normalize(feat, p=2, dim=1)
        return feat.cpu()

    @torch.no_grad()
    def _embed(self, image: Image.Image) -> torch.Tensor:
        """Embed a single PIL image with the evaluation protocol.

        Args:
            image: An RGB ``PIL.Image``.

        Returns:
            A CPU tensor of shape ``(1, feat_dim)``.
        """
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        return self._features(tensor)

    @torch.no_grad()
    def _extract_gallery_features(self) -> None:
        """Extract and cache embeddings for the indexed gallery."""
        assert self.gallery is not None
        from torch.utils.data import DataLoader

        loader = DataLoader(
            self.gallery,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=0,
        )
        feats = [self._features(images.to(self.device)) for images, _pids, _camids in loader]
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
                "Model: **random initialization** (no valid weights), so "
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
        order = np.argsort(-sims, kind="stable")[:topk]

        items: list[tuple[Any, str]] = []
        for rank, gid in enumerate(order, start=1):
            path = self.gallery.img_paths[int(gid)]
            try:
                img = PILImage.open(path).convert("RGB")
            except (OSError, ValueError):
                logger.warning("Skipping unreadable gallery image %s", path)
                continue
            pid = self.gallery.pids[int(gid)]
            camid = self.gallery.camids[int(gid)]
            caption = f"#{rank} | ID {pid} | Cam {camid} | sim {sims[int(gid)]:.3f}"
            items.append((img, caption))

        suffix = "" if self.has_weights else " (random-init model, so results are not meaningful)"
        return items, f"Showing top-{topk} matches by cosine similarity.{suffix}"


def build_demo(engine: ReIDDemoEngine) -> gr.Blocks:
    """Build the Gradio Blocks UI bound to a demo engine.

    Args:
        engine: A constructed :class:`ReIDDemoEngine`.

    Returns:
        A :class:`gradio.Blocks` application ready to ``launch``.

    Raises:
        ImportError: If gradio is not installed.
    """
    try:
        import gradio as gr
    except ImportError as exc:
        msg = 'The demo requires gradio. Install it with pip install "secondsight[demo]"'
        raise ImportError(msg) from exc

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


def parse_auth(value: str) -> tuple[str, str]:
    """Parse a ``USER:PASS`` credential for ``--auth``.

    Only the first colon separates the two parts, so the password may itself
    contain colons.

    Args:
        value: The raw ``--auth`` value.

    Returns:
        A ``(user, password)`` tuple.

    Raises:
        argparse.ArgumentTypeError: If the colon is missing or the user or the
            password is empty, which would otherwise allow a blank password.
    """
    user, sep, password = value.partition(":")
    if not sep or not user or not password:
        msg = "--auth must be USER:PASS with a non-empty user and password."
        raise argparse.ArgumentTypeError(msg)
    return user, password


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
        default=None,
        help="YAML configuration file. Defaults to the packaged strong-baseline config.",
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
        help="Compute device ('auto', 'cuda', 'mps' or 'cpu'). Overrides train.device.",
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
        default=None,
        help=(
            f"Host interface to bind. Defaults to the {_ENV_SERVER_NAME} env var, "
            f"else {_DEFAULT_SERVER_NAME}."
        ),
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=None,
        help=(
            f"Server port. Defaults to the {_ENV_SERVER_PORT} env var, else {_DEFAULT_SERVER_PORT}."
        ),
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link.",
    )
    parser.add_argument(
        "--auth",
        type=parse_auth,
        default=None,
        metavar="USER:PASS",
        help="Require a login as 'user:password'. Recommended whenever the server "
        "binds to a non-loopback interface or --share is used.",
    )
    return parser


def resolve_server(name: str | None, port: int | None) -> tuple[str, int]:
    """Resolve the server address from the CLI, the environment or defaults.

    Args:
        name: The ``--server-name`` value, or ``None``.
        port: The ``--server-port`` value, or ``None``.

    Returns:
        A ``(server_name, server_port)`` tuple. Missing values fall back to
        ``GRADIO_SERVER_NAME`` and ``GRADIO_SERVER_PORT``, then to
        ``127.0.0.1`` and ``7860``.

    Raises:
        ValueError: If ``GRADIO_SERVER_PORT`` is set but is not an integer.
    """
    server_name = name or os.environ.get(_ENV_SERVER_NAME) or _DEFAULT_SERVER_NAME
    if port is not None:
        return server_name, port
    env_port = os.environ.get(_ENV_SERVER_PORT)
    return server_name, int(env_port) if env_port else _DEFAULT_SERVER_PORT


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
        Process exit code (``0`` on a clean shutdown, ``1`` on a setup error).
    """
    args = build_parser().parse_args(argv)
    setup_logger("reid")

    config_path = args.config if args.config is not None else default_config_path()
    if not config_path.is_file():
        logger.error("Config file not found: %s", config_path)
        return 1
    try:
        server_name, server_port = resolve_server(args.server_name, args.server_port)
    except ValueError:
        logger.error("%s must be an integer port number.", _ENV_SERVER_PORT)
        return 1

    cfg = Config.from_yaml(config_path)
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

    auth: tuple[str, str] | None = args.auth
    if (args.share or server_name not in _LOOPBACK) and auth is None:
        logger.warning(
            "Exposing the demo on %s without authentication; pass --auth "
            "USER:PASS to require a login.",
            "a public share link" if args.share else server_name,
        )

    demo = build_demo(engine)
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=args.share,
        auth=auth,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
