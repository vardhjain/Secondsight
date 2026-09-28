"""Quantitative analysis plots for Person Re-Identification.

This module produces the analytical figures that accompany a Re-ID evaluation.
They are the Cumulative Matching Characteristic (CMC) curve, the distribution
of per-query Average Precision (AP), a cross-camera Rank-1 accuracy heatmap, a
t-SNE projection of the learned embedding space, and the intra-class versus
inter-class distance distributions.

The camera-pair heatmap is a port of cell 9 of the original research notebook.
The remaining plots are standard Re-ID diagnostics.

Matplotlib and scikit-learn come from the optional ``viz`` extra and are
imported inside the functions that need them, so importing this module only
requires ``numpy``. The heatmap uses ``seaborn`` when it is installed and falls
back to plain matplotlib otherwise.

Every plotting function returns the matplotlib figure it drew. When
``save_path`` is given the figure is written to disk and closed first.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from reid.visualization._common import finalize_figure, pyplot, require

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from matplotlib.figure import Figure


def plot_cmc_curve(cmc: np.ndarray, save_path: str | Path | None = None) -> Figure:
    """Plot the Cumulative Matching Characteristic (CMC) curve.

    The CMC curve reports, for each rank ``k``, the probability that the
    correct match appears within the top ``k`` retrieved gallery images. A
    curve that rises steeply and saturates near 1.0 indicates strong retrieval.

    Args:
        cmc: 1-D array of cumulative match rates indexed by rank (``cmc[0]`` is
            Rank-1 accuracy). Values are expected in ``[0, 1]``.
        save_path: Optional path to save the figure. If ``None`` the figure is
            shown interactively.

    Returns:
        The matplotlib figure.
    """
    plt = pyplot()

    cmc = np.asarray(cmc, dtype=np.float64)
    ranks = np.arange(1, len(cmc) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ranks, cmc, marker="o", markersize=3, linewidth=2, color="#1f77b4")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Matching Rate")
    ax.set_title("Cumulative Matching Characteristic (CMC) Curve")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(1, max(2, len(cmc)))
    ax.grid(visible=True, linestyle="--", alpha=0.4)

    # Annotate a few headline ranks where available.
    for rank in (1, 5, 10):
        if rank <= len(cmc):
            ax.annotate(
                f"R{rank}: {cmc[rank - 1]:.1%}",
                xy=(rank, cmc[rank - 1]),
                xytext=(rank + 0.5, max(0.0, cmc[rank - 1] - 0.08)),
                fontsize=9,
            )

    fig.tight_layout()
    return finalize_figure(fig, save_path)


def plot_ap_distribution(aps: np.ndarray, save_path: str | Path | None = None) -> Figure:
    """Plot a histogram of per-query Average Precision (AP) scores.

    The spread of per-query AP shows whether performance is uniform across
    queries or dominated by a few easy or hard cases. The mean AP (the mAP) is
    marked with a vertical line.

    Args:
        aps: 1-D array of per-query AP values in ``[0, 1]``.
        save_path: Optional path to save the figure. If ``None`` the figure is
            shown interactively.

    Returns:
        The matplotlib figure.
    """
    plt = pyplot()

    aps = np.asarray(aps, dtype=np.float64)
    mean_ap = float(np.mean(aps)) if aps.size else 0.0

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(aps, bins=30, range=(0.0, 1.0), color="#2ca02c", alpha=0.8, edgecolor="white")
    ax.axvline(mean_ap, color="red", linestyle="--", linewidth=2, label=f"mAP = {mean_ap:.1%}")
    ax.set_xlabel("Average Precision (per query)")
    ax.set_ylabel("Number of Queries")
    ax.set_title("Distribution of Per-Query Average Precision")
    ax.legend()
    ax.grid(visible=True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    return finalize_figure(fig, save_path)


def _camera_pair_rank1(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    cameras: list[int],
) -> np.ndarray:
    """Compute Rank-1 accuracy for every ordered pair of distinct cameras.

    Args:
        distmat: Query-to-gallery distance matrix of shape ``[num_q, num_g]``.
        q_pids: Query person IDs.
        g_pids: Gallery person IDs.
        q_camids: Query camera IDs.
        g_camids: Gallery camera IDs.
        cameras: Sorted list of camera IDs defining the matrix axes.

    Returns:
        A ``[n_cam, n_cam]`` float array. Entries are ``NaN`` on the diagonal
        and wherever no query identity appears in the gallery camera.
    """
    n_cam = len(cameras)
    matrix = np.full((n_cam, n_cam), np.nan, dtype=np.float64)
    for i, q_cam in enumerate(cameras):
        q_idxs = np.flatnonzero(q_camids == q_cam)
        if q_idxs.size == 0:
            continue
        for j, g_cam in enumerate(cameras):
            if q_cam == g_cam:
                continue
            g_idxs = np.flatnonzero(g_camids == g_cam)
            if g_idxs.size == 0:
                continue
            sub_g_pids = g_pids[g_idxs]
            sub_q_pids = q_pids[q_idxs]
            # Only count queries whose identity exists in this gallery camera.
            has_match = np.isin(sub_q_pids, sub_g_pids)
            if not has_match.any():
                continue
            top1 = np.argmin(distmat[np.ix_(q_idxs[has_match], g_idxs)], axis=1)
            matrix[i, j] = float(np.mean(sub_g_pids[top1] == sub_q_pids[has_match]))
    return matrix


def plot_camera_heatmap(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    save_path: str | Path | None = None,
) -> Figure:
    """Plot a cross-camera Rank-1 accuracy heatmap.

    For every ordered pair of distinct cameras ``(query_cam, gallery_cam)``
    this computes the Rank-1 accuracy of queries from the query camera matched
    only against gallery images from the gallery camera. The diagonal (same
    camera) is left blank because cross-camera matching is the quantity of
    interest. This is a port of cell 9 of the original notebook.

    Args:
        distmat: Query-to-gallery distance matrix of shape ``[num_q, num_g]``.
        q_pids: Query person IDs, shape ``[num_q]``.
        g_pids: Gallery person IDs, shape ``[num_g]``.
        q_camids: Query camera IDs, shape ``[num_q]``.
        g_camids: Gallery camera IDs, shape ``[num_g]``.
        save_path: Optional path to save the figure. If ``None`` the figure is
            shown interactively.

    Returns:
        The matplotlib figure.
    """
    plt = pyplot()
    try:
        import seaborn as sns
    except ImportError:
        sns = None

    distmat = np.asarray(distmat)
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)

    cameras = sorted(set(q_camids.tolist()) | set(g_camids.tolist()))
    n_cam = len(cameras)
    matrix = _camera_pair_rank1(distmat, q_pids, g_pids, q_camids, g_camids, cameras)

    fig, ax = plt.subplots(figsize=(10, 8))
    if sns is not None:
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1%",
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            xticklabels=cameras,
            yticklabels=cameras,
            ax=ax,
            cbar_kws={"label": "Rank-1 Accuracy"},
        )
    else:
        # Plain matplotlib fallback when seaborn is unavailable.
        masked = np.ma.masked_invalid(matrix)
        cmap = plt.get_cmap("RdYlGn").with_extremes(bad="lightgray")
        im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(n_cam))
        ax.set_yticks(range(n_cam))
        ax.set_xticklabels(cameras)
        ax.set_yticklabels(cameras)
        fig.colorbar(im, ax=ax, label="Rank-1 Accuracy")
        for i in range(n_cam):
            for j in range(n_cam):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.1%}", ha="center", va="center", fontsize=8)

    ax.set_title("Rank-1 Accuracy by Camera Pair (Angle Analysis)")
    ax.set_xlabel("Gallery Camera ID")
    ax.set_ylabel("Query Camera ID")

    fig.tight_layout()
    return finalize_figure(fig, save_path)


def plot_tsne(
    features: np.ndarray,
    pids: np.ndarray,
    num_ids: int = 20,
    save_path: str | Path | None = None,
    *,
    seed: int | None = 0,
) -> Figure:
    """Plot a 2-D t-SNE projection of the embedding space.

    A random subset of ``num_ids`` identities is selected and their feature
    vectors are projected to two dimensions with t-SNE. Tight, well separated
    clusters indicate a discriminative embedding. Each selected identity gets
    its own categorical colour, so different people never share a colour when
    at most 20 identities are shown.

    Args:
        features: Feature matrix of shape ``[N, D]`` (a numpy array or anything
            accepted by ``np.asarray``, including CPU torch tensors).
        pids: Person IDs of shape ``[N]`` aligned with ``features``.
        num_ids: Number of distinct identities to sample and visualize.
        save_path: Optional path to save the figure. If ``None`` the figure is
            shown interactively.
        seed: Seed for the identity sampling and for t-SNE itself, so the
            figure is reproducible. ``None`` draws fresh randomness.

    Returns:
        The matplotlib figure.

    Raises:
        ImportError: If matplotlib or scikit-learn is not installed.
    """
    plt = pyplot()
    manifold = require("sklearn.manifold", "plot_tsne")
    colors = require("matplotlib.colors", "plot_tsne")

    features = np.asarray(features, dtype=np.float32)
    pids = np.asarray(pids)

    unique_pids = np.unique(pids)
    rng = np.random.default_rng(seed)
    n_select = min(num_ids, len(unique_pids))
    selected_pids = np.sort(rng.choice(unique_pids, size=n_select, replace=False))

    mask = np.isin(pids, selected_pids)
    sub_features = features[mask]
    sub_pids = pids[mask]
    labels = np.searchsorted(selected_pids, sub_pids)

    # t-SNE perplexity must be smaller than the number of samples.
    n_samples = sub_features.shape[0]
    perplexity = float(max(1, min(30, n_samples - 1)))
    tsne = manifold.TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=0 if seed is None else seed,
    )
    embedded = tsne.fit_transform(sub_features)

    if n_select <= 20:
        cmap = colors.ListedColormap(plt.get_cmap("tab20").colors[: max(1, n_select)])
    else:
        cmap = plt.get_cmap("hsv", n_select)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(
        embedded[:, 0],
        embedded[:, 1],
        c=labels,
        cmap=cmap,
        vmin=-0.5,
        vmax=n_select - 0.5,
        s=18,
        alpha=0.8,
    )
    ax.set_title(f"t-SNE of Embedding Space ({n_select} identities, one colour each)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")

    fig.tight_layout()
    return finalize_figure(fig, save_path)


def plot_distance_distributions(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray | None = None,
    g_camids: np.ndarray | None = None,
    save_path: str | Path | None = None,
    *,
    max_neg: int = 1_000_000,
    seed: int | None = 0,
) -> Figure:
    """Plot intra-class versus inter-class distance distributions.

    Query and gallery pairs are split into pairs sharing an identity (positive)
    and pairs with different identities (negative). A clean Re-ID embedding
    separates the two histograms, with positive distances concentrated near
    zero. When camera IDs are given, same-identity pairs from the same camera
    are dropped, exactly as the Market-1501 protocol drops them, so near
    duplicates do not inflate the positive peak. Negatives are subsampled to at
    most ``max_neg`` pairs to keep memory bounded on the full test split.

    Args:
        distmat: Query-to-gallery distance matrix of shape ``[num_q, num_g]``.
        q_pids: Query person IDs, shape ``[num_q]``.
        g_pids: Gallery person IDs, shape ``[num_g]``.
        q_camids: Optional query camera IDs, shape ``[num_q]``.
        g_camids: Optional gallery camera IDs, shape ``[num_g]``.
        save_path: Optional path to save the figure. If ``None`` the figure is
            shown interactively.
        max_neg: Maximum number of negative pairs drawn into the histogram.
        seed: Seed for the negative subsampling.

    Returns:
        The matplotlib figure.
    """
    plt = pyplot()

    distmat = np.asarray(distmat, dtype=np.float32)
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)

    same_id = q_pids[:, None] == g_pids[None, :]
    positive = same_id
    if q_camids is not None and g_camids is not None:
        same_cam = np.asarray(q_camids)[:, None] == np.asarray(g_camids)[None, :]
        positive = same_id & ~same_cam
    pos_dists = distmat[positive]

    neg_idx = np.flatnonzero(~same_id)
    if neg_idx.size > max_neg:
        rng = np.random.default_rng(seed)
        neg_idx = rng.choice(neg_idx, size=max_neg, replace=False)
    neg_dists = distmat.ravel()[neg_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    if pos_dists.size:
        ax.hist(
            pos_dists,
            bins=60,
            density=True,
            alpha=0.6,
            color="#2ca02c",
            label="Same identity (positive)",
        )
    if neg_dists.size:
        ax.hist(
            neg_dists,
            bins=60,
            density=True,
            alpha=0.6,
            color="#d62728",
            label="Different identity (negative)",
        )
    ax.set_xlabel("Pairwise Distance")
    ax.set_ylabel("Density")
    ax.set_title("Intra-class vs. Inter-class Distance Distributions")
    if pos_dists.size or neg_dists.size:
        ax.legend()
    ax.grid(visible=True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    return finalize_figure(fig, save_path)
