"""Feature extraction and end-to-end evaluation for re-identification models.

This module provides:

The function :func:`extract_features` runs a trained model over a dataloader
to produce the feature matrix together with the per-image person ids and camera
ids. It optionally averages the features of an image and its horizontal flip
(test-time augmentation) and L2-normalises the result so that Euclidean
distance becomes equivalent to cosine distance.

The class :class:`Evaluator` is a thin orchestrator that extracts query and
gallery features once (caching them), builds the distance matrix, optionally
applies k-reciprocal re-ranking, and reports CMC and mAP metrics.

Only ``torch``/``numpy`` plus this package's lightweight utilities are needed at
import time; ``torchvision`` is *not* imported here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import normalize

from reid.evaluation.metrics import compute_cmc_map
from reid.evaluation.rerank import re_ranking
from reid.utils.distance import compute_distance_matrix

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader

    from reid.config import Config

__all__ = ["extract_features", "Evaluator"]

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
    flip_tta: bool = False,
    feat_norm: bool = True,
) -> tuple[Tensor, np.ndarray, np.ndarray]:
    """Extract features for every image yielded by ``loader``.

    The model is switched to eval mode and run under ``torch.no_grad``. Each
    batch is expected to yield ``(images, pids, camids)`` as produced by the
    :class:`~reid.data.dataset.Market1501` dataset.

    Args:
        model: A trained re-identification model. In eval mode its ``forward``
            (or :meth:`extract_features`) must return a feature tensor of shape
            ``(batch, feat_dim)``.
        loader: Dataloader over a query or gallery split.
        device: Device on which to run inference (e.g. ``"cuda"`` / ``"cpu"``).
        flip_tta: If ``True``, average the features of each image and its
            horizontally flipped copy (test-time augmentation).
        feat_norm: If ``True``, L2-normalise the final feature vectors so that
            Euclidean distance is equivalent to cosine distance.

    Returns:
        A tuple ``(features, pids, camids)`` where ``features`` is a CPU tensor
        of shape ``(num_images, feat_dim)`` and ``pids`` / ``camids`` are
        ``int64`` NumPy arrays of length ``num_images``.

    Raises:
        ValueError: If ``loader`` yields no batches.
    """
    model.eval()
    device = torch.device(device)

    features: list[Tensor] = []
    pids: list[int] = []
    camids: list[int] = []

    for batch in loader:
        images, batch_pids, batch_camids = batch
        images = images.to(device, non_blocking=True)

        feats = _forward(model, images)
        if flip_tta:
            flipped = torch.flip(images, dims=[3])
            feats = feats + _forward(model, flipped)
            feats = feats / 2.0

        if feat_norm:
            feats = normalize(feats, p=2, dim=1)

        features.append(feats.cpu())
        pids.extend(_to_int_list(batch_pids))
        camids.extend(_to_int_list(batch_camids))

    if not features:
        raise ValueError("The dataloader yielded no batches; cannot extract features.")
    feature_matrix = torch.cat(features, dim=0)
    pid_array = np.asarray(pids, dtype=np.int64)
    camid_array = np.asarray(camids, dtype=np.int64)
    return feature_matrix, pid_array, camid_array


def _forward(model: nn.Module, images: Tensor) -> Tensor:
    """Run a forward pass and return the eval-time feature tensor.

    Prefers an explicit ``extract_features`` method on the model (the BNNeck
    post-bottleneck feature); falls back to a plain ``forward`` call, which in
    eval mode also returns the post-BN feature for this project's model.

    Args:
        model: The re-identification model.
        images: A batch of input images of shape ``(batch, 3, H, W)``.

    Returns:
        A float feature tensor of shape ``(batch, feat_dim)``.
    """
    extractor = getattr(model, "extract_features", None)
    feats = extractor(images) if callable(extractor) else model(images)
    if isinstance(feats, (tuple, list)):
        feats = feats[-1]
    return feats.float()


def _to_int_list(values: Tensor | np.ndarray | Iterable[int]) -> list[int]:
    """Convert a batch of ids (tensor / array / sequence) to a list of ints.

    Args:
        values: Per-sample identity or camera ids from a dataloader batch.

    Returns:
        A plain Python ``list[int]``.
    """
    if isinstance(values, Tensor):
        return [int(v) for v in values.detach().cpu().long().tolist()]
    return [int(v) for v in values]


class Evaluator:
    """Evaluate a model on a query/gallery split with optional re-ranking.

    Features are extracted lazily on the first :meth:`evaluate` call and cached,
    so repeated evaluations (e.g. with and without re-ranking) do not re-run the
    backbone.

    Attributes:
        model: The model under evaluation.
        query_loader: Dataloader over the query split.
        gallery_loader: Dataloader over the gallery split.
        device: Device used for feature extraction.
        cfg: The full project configuration (drives TTA, norm, re-rank, ...).
    """

    def __init__(
        self,
        model: nn.Module,
        query_loader: DataLoader,
        gallery_loader: DataLoader,
        device: torch.device | str,
        cfg: Config,
    ) -> None:
        """Initialise the evaluator.

        Args:
            model: A trained re-identification model.
            query_loader: Dataloader over the query split.
            gallery_loader: Dataloader over the gallery split.
            device: Device on which to extract features.
            cfg: The full :class:`~reid.config.Config` configuration object.
        """
        self.model = model
        self.query_loader = query_loader
        self.gallery_loader = gallery_loader
        self.device = torch.device(device)
        self.cfg = cfg

        # Feature cache (populated by _ensure_features).
        self._qf: Tensor | None = None
        self._gf: Tensor | None = None
        self._q_pids: np.ndarray | None = None
        self._g_pids: np.ndarray | None = None
        self._q_camids: np.ndarray | None = None
        self._g_camids: np.ndarray | None = None

    def reset_cache(self) -> None:
        """Drop the cached query/gallery features.

        Call this after the model weights change (e.g. between training epochs)
        so the next :meth:`evaluate` re-extracts features.
        """
        self._qf = None
        self._gf = None
        self._q_pids = None
        self._g_pids = None
        self._q_camids = None
        self._g_camids = None

    def _ensure_features(self) -> None:
        """Extract and cache query/gallery features if not already cached."""
        if self._qf is not None and self._gf is not None:
            return

        eval_cfg = self.cfg.eval
        logger.info("Extracting query features...")
        self._qf, self._q_pids, self._q_camids = extract_features(
            self.model,
            self.query_loader,
            self.device,
            flip_tta=eval_cfg.flip_tta,
            feat_norm=eval_cfg.feat_norm,
        )
        logger.info("Extracting gallery features...")
        self._gf, self._g_pids, self._g_camids = extract_features(
            self.model,
            self.gallery_loader,
            self.device,
            flip_tta=eval_cfg.flip_tta,
            feat_norm=eval_cfg.feat_norm,
        )
        logger.info(
            "Extracted %d query and %d gallery features (dim=%d).",
            self._qf.shape[0],
            self._gf.shape[0],
            self._qf.shape[1],
        )

    def evaluate(self, rerank: bool | None = None) -> dict[str, object]:
        """Run the full evaluation pipeline and return the metrics.

        Args:
            rerank: Whether to additionally compute re-ranked metrics. If
                ``None`` (default) the value of ``cfg.eval.rerank`` is used.

        Returns:
            A dict with keys ``mAP``, ``rank1``, ``rank5``, ``rank10`` and
            ``cmc`` (the full CMC array). When re-ranking is enabled the same
            metrics are additionally returned with a ``rerank_`` prefix
            (``rerank_mAP``, ``rerank_rank1``, ...).
        """
        if rerank is None:
            rerank = self.cfg.eval.rerank

        self._ensure_features()
        qf, gf = self._qf, self._gf
        q_pids, g_pids = self._q_pids, self._g_pids
        q_camids, g_camids = self._q_camids, self._g_camids
        if (
            qf is None
            or gf is None
            or q_pids is None
            or g_pids is None
            or q_camids is None
            or g_camids is None
        ):
            raise RuntimeError("Feature extraction did not populate the evaluator cache.")

        max_rank = self.cfg.eval.max_rank
        if max_rank < 1:
            raise ValueError(f"cfg.eval.max_rank must be >= 1, got {max_rank}")

        logger.info("Computing distance matrix and base metrics...")
        distmat = compute_distance_matrix(qf, gf, metric="euclidean")
        cmc, mean_ap = compute_cmc_map(
            distmat.cpu().numpy(), q_pids, g_pids, q_camids, g_camids, max_rank=max_rank
        )
        results: dict[str, object] = _summarise(cmc, mean_ap, prefix="")

        if rerank:
            logger.info("Running k-reciprocal re-ranking...")
            rr_distmat = re_ranking(
                qf,
                gf,
                k1=self.cfg.eval.rerank_k1,
                k2=self.cfg.eval.rerank_k2,
                lambda_value=self.cfg.eval.rerank_lambda,
            )
            rr_cmc, rr_map = compute_cmc_map(
                rr_distmat, q_pids, g_pids, q_camids, g_camids, max_rank=max_rank
            )
            results.update(_summarise(rr_cmc, rr_map, prefix="rerank_"))

        return results


def _summarise(cmc: np.ndarray, mean_ap: float, prefix: str) -> dict[str, object]:
    """Build the metrics dict for one CMC curve, with keys prefixed by ``prefix``.

    When the CMC curve is shorter than 10 (because :func:`compute_cmc_map`
    clamped ``max_rank`` to a gallery smaller than 10, or a small ``max_rank``
    was configured), Rank-5 and Rank-10 fall back to the last CMC value. For a
    clamped curve this is exact, because every query has already been scored
    against the whole gallery, so the CMC cannot rise beyond its final value.

    Args:
        cmc: CMC curve as returned by :func:`compute_cmc_map`.
        mean_ap: Mean average precision.
        prefix: Key prefix, either ``""`` or ``"rerank_"``.

    Returns:
        A dict with the ``mAP``, ``rank1``, ``rank5``, ``rank10`` and ``cmc``
        entries, each key carrying ``prefix``.
    """

    def at_rank(k: int) -> float:
        return float(cmc[k - 1]) if len(cmc) >= k else float(cmc[-1])

    return {
        f"{prefix}mAP": float(mean_ap),
        f"{prefix}rank1": at_rank(1),
        f"{prefix}rank5": at_rank(5),
        f"{prefix}rank10": at_rank(10),
        f"{prefix}cmc": cmc,
    }
