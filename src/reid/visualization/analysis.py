"""Quantitative analysis plots for Person Re-Identification.

This module produces the analytical figures that accompany a Re-ID evaluation:
the Cumulative Matching Characteristic (CMC) curve, the distribution of
per-query Average Precision (AP), a cross-camera Rank-1 accuracy heatmap, a
t-SNE projection of the learned embedding space, and intra- vs. inter-class
distance distributions.

The camera-pair heatmap is a faithful port of cell 9 of the original research
notebook. The remaining plots are standard Re-ID diagnostics.

Optional dependencies (``seaborn`` for the heatmap, ``scikit-learn`` for
t-SNE) are imported lazily inside the functions that need them, so importing
this module only requires ``matplotlib`` and ``numpy``. Matplotlib is likewise
imported inside each function to avoid forcing a backend at package import.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _finalize(fig, save_path: str | Path | None) -> None:
    """Saves a figure to disk or shows it interactively.

    Args:
        fig: The matplotlib figure to finalize.
        save_path: Destination path; if ``None`` the figure is shown and left
            open, otherwise it is written to disk and closed.
    """
    import matplotlib.pyplot as plt

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_cmc_curve(cmc: np.ndarray, save_path: str | Path | None = None) -> None:
    """Plots the Cumulative Matching Characteristic (CMC) curve.

    The CMC curve reports, for each rank ``k``, the probability that the
    correct match appears within the top-``k`` retrieved gallery images. A
    curve that rises steeply and saturates near 1.0 indicates strong retrieval
    performance.

    Args:
        cmc: 1-D array of cumulative match rates indexed by rank (``cmc[0]`` is
            Rank-1 accuracy). Values are expected in ``[0, 1]``.
        save_path: Optional path to save the figure. If ``None`` the figure is
            shown interactively.

    Returns:
        None.
    """
    import matplotlib.pyplot as plt

    cmc = np.asarray(cmc, dtype=np.float64)
    ranks = np.arange(1, len(cmc) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ranks, cmc, marker="o", markersize=3, linewidth=2, color="#1f77b4")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Matching Rate")
    ax.set_title("Cumulative Matching Characteristic (CMC) Curve")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(1, len(cmc))
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
    _finalize(fig, save_path)


def plot_ap_distribution(aps: np.ndarray, save_path: str | Path | None = None) -> None:
    """Plots a histogram of per-query Average Precision (AP) scores.

    The spread of per-query AP reveals whether performance is uniform across
    queries or dominated by a few easy/hard cases. The mean AP (i.e. mAP) is
    marked with a vertical line.

    Args:
        aps: 1-D array of per-query AP values in ``[0, 1]``.
        save_path: Optional path to save the figure. If ``None`` the figure is
            shown interactively.

    Returns:
        None.
    """
    import matplotlib.pyplot as plt

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
    _finalize(fig, save_path)


def plot_camera_heatmap(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Plots a cross-camera Rank-1 accuracy heatmap.

    For every ordered pair of distinct cameras ``(query_cam, gallery_cam)``,
    this computes the Rank-1 retrieval accuracy restricted to queries from the
    query camera matched against gallery images from the gallery camera. The
    diagonal (same camera) is left blank because cross-camera matching is the
    quantity of interest. This is a port of cell 9 of the original notebook.

    Args:
        distmat: Query-to-gallery distance matrix of shape ``[num_q, num_g]``.
        q_pids: Query person IDs, shape ``[num_q]``.
        g_pids: Gallery person IDs, shape ``[num_g]``.
        q_camids: Query camera IDs, shape ``[num_q]``.
        g_camids: Gallery camera IDs, shape ``[num_g]``.
        save_path: Optional path to save the figure. If ``None`` the figure is
            shown interactively.

    Returns:
        None.
    """
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        _has_seaborn = True
    except ImportError:  # pragma: no cover - optional dependency
        _has_seaborn = False

    distmat = np.asarray(distmat)
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)

    cameras = sorted(set(q_camids.tolist()) | set(g_camids.tolist()))
    n_cam = len(cameras)
    matrix = np.full((n_cam, n_cam), np.nan, dtype=np.float64)

    for i, q_cam in enumerate(cameras):
        for j, g_cam in enumerate(cameras):
            if q_cam == g_cam:
                continue

            q_idxs = np.where(q_camids == q_cam)[0]
            g_idxs = np.where(g_camids == g_cam)[0]
            if len(q_idxs) == 0 or len(g_idxs) == 0:
                continue

            sub_dist = distmat[q_idxs][:, g_idxs]
            sub_q_pids = q_pids[q_idxs]
            sub_g_pids = g_pids[g_idxs]
            ranking = np.argsort(sub_dist, axis=1)

            correct = 0
            total = 0
            for k in range(len(q_idxs)):
                target_pid = sub_q_pids[k]
                # Only count queries whose identity exists in this gallery cam.
                if target_pid not in sub_g_pids:
                    continue
                total += 1
                top_idx = ranking[k, 0]
                if sub_g_pids[top_idx] == target_pid:
                    correct += 1

            if total > 0:
                matrix[i, j] = correct / total

    fig, ax = plt.subplots(figsize=(10, 8))
    if _has_seaborn:
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".1%",
            cmap="RdYlGn",
            xticklabels=cameras,
            yticklabels=cameras,
            ax=ax,
            cbar_kws={"label": "Rank-1 Accuracy"},
        )
    else:
        # Matplotlib fallback when seaborn is unavailable.
        masked = np.ma.masked_invalid(matrix)
        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad(color="lightgray")
        im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(range(n_cam))
        ax.set_yticks(range(n_cam))
        ax.set_xticklabels(cameras)
        ax.set_yticklabels(cameras)
        fig.colorbar(im, ax=ax, label="Rank-1 Accuracy")
        for i in range(n_cam):
            for j in range(n_cam):
                if not np.isnan(matrix[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{matrix[i, j]:.1%}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

    ax.set_title("Rank-1 Accuracy by Camera Pair (Angle Analysis)")
    ax.set_xlabel("Gallery Camera ID")
    ax.set_ylabel("Query Camera ID")

    fig.tight_layout()
    _finalize(fig, save_path)


def plot_tsne(
    features: np.ndarray,
    pids: np.ndarray,
    num_ids: int = 20,
    save_path: str | Path | None = None,
) -> None:
    """Plots a 2-D t-SNE projection of the embedding space.

    A random subset of ``num_ids`` identities is selected and their feature
    vectors are projected to two dimensions with t-SNE. Tight, well-separated
    clusters indicate that the model has learned a discriminative embedding.

    Args:
        features: Feature matrix of shape ``[N, D]`` (numpy array or anything
            convertible via ``np.asarray``; torch tensors are accepted).
        pids: Person IDs of shape ``[N]`` aligning with ``features``.
        num_ids: Number of distinct identities to sample and visualize.
        save_path: Optional path to save the figure. If ``None`` the figure is
            shown interactively.

    Returns:
        None.

    Raises:
        ImportError: If scikit-learn is not installed.
    """
    import matplotlib.pyplot as plt

    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:  # pragma: no cover - optional dependency
        msg = "plot_tsne requires scikit-learn. Install it with `pip install scikit-learn`."
        raise ImportError(msg) from exc

    features = np.asarray(features, dtype=np.float32)
    pids = np.asarray(pids)

    unique_pids = np.unique(pids)
    rng = np.random.default_rng()
    n_select = min(num_ids, len(unique_pids))
    selected_pids = rng.choice(unique_pids, size=n_select, replace=False)

    mask = np.isin(pids, selected_pids)
    sub_features = features[mask]
    sub_pids = pids[mask]

    # t-SNE perplexity must be smaller than the number of samples.
    n_samples = sub_features.shape[0]
    perplexity = float(max(2, min(30, n_samples - 1)))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=42)
    embedded = tsne.fit_transform(sub_features)

    fig, ax = plt.subplots(figsize=(9, 8))
    scatter = ax.scatter(
        embedded[:, 0],
        embedded[:, 1],
        c=sub_pids,
        cmap="tab20",
        s=18,
        alpha=0.8,
    )
    ax.set_title(f"t-SNE of Embedding Space ({n_select} identities)")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    fig.colorbar(scatter, ax=ax, label="Person ID")

    fig.tight_layout()
    _finalize(fig, save_path)


def plot_distance_distributions(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Plots intra-class vs. inter-class distance distributions.

    Distances between query/gallery pairs are split into two groups: pairs
    sharing the same identity (positive / intra-class) and pairs with different
    identities (negative / inter-class). A clean Re-ID embedding produces a
    large separation between the two histograms, with positive distances
    concentrated near zero.

    Args:
        distmat: Query-to-gallery distance matrix of shape ``[num_q, num_g]``.
        q_pids: Query person IDs, shape ``[num_q]``.
        g_pids: Gallery person IDs, shape ``[num_g]``.
        save_path: Optional path to save the figure. If ``None`` the figure is
            shown interactively.

    Returns:
        None.
    """
    import matplotlib.pyplot as plt

    distmat = np.asarray(distmat, dtype=np.float64)
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)

    # Boolean mask: True where query/gallery identities match.
    same_id = q_pids[:, None] == g_pids[None, :]
    pos_dists = distmat[same_id]
    neg_dists = distmat[~same_id]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        pos_dists,
        bins=60,
        density=True,
        alpha=0.6,
        color="#2ca02c",
        label="Same identity (positive)",
    )
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
    ax.legend()
    ax.grid(visible=True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    _finalize(fig, save_path)
