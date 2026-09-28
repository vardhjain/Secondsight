"""Ranked-result visualization utilities for Person Re-Identification.

This module renders qualitative Re-ID results. It draws galleries of the top-k
matches for a set of query images, curated Rank-1 success and failure panels,
and a before-versus-after comparison showing queries whose Rank-1 match is
corrected by k-reciprocal re-ranking. These plots are the most directly
readable artifacts of a Re-ID system because they show, image by image,
whether the model retrieves the correct identity across cameras.

:func:`plot_success_failure` is ported from cell 11 of the original research
notebook and :func:`plot_rerank_impact` from cell 12. Both now work from a
precomputed distance matrix, so they rank with exactly the protocol behind the
reported metrics. :func:`visualize_ranked_results` is new in the package.

Every ranking follows the Market-1501 protocol and ignores gallery images that
share both identity and camera with the query. Matplotlib comes from the
optional ``viz`` extra and is imported inside each function. Every plotting
function returns the matplotlib figure it drew (closed when saved), or ``None``
when there was nothing to draw.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from reid.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from reid.visualization._common import finalize_figure, pyplot

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from torch import Tensor
    from torch.utils.data import Dataset


def _denormalize_to_image(tensor: Tensor) -> np.ndarray:
    """Invert ImageNet normalization and convert a CHW tensor to an image.

    Args:
        tensor: A normalized image tensor of shape ``[3, H, W]``.

    Returns:
        An RGB image as a ``float32`` numpy array of shape ``[H, W, 3]`` with
        values clipped to ``[0, 1]``.
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    img = (tensor.detach().cpu() * std + mean).permute(1, 2, 0).numpy()
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _get_display_image(dataset: Dataset, idx: int) -> np.ndarray:
    """Return a viewable RGB image for a dataset index.

    The dataset's transform yields a normalized tensor, and this helper inverts
    that normalization so the result can be passed straight to ``imshow``.

    Args:
        dataset: A dataset yielding ``(image, pid, camid)`` samples, such as
            :class:`~reid.data.dataset.Market1501`.
        idx: Index of the sample to render.

    Returns:
        An RGB image as a ``float32`` numpy array of shape ``[H, W, 3]`` in
        ``[0, 1]``.
    """
    img = dataset[idx][0]
    if isinstance(img, torch.Tensor):
        return _denormalize_to_image(img)
    # Fall back gracefully if a raw PIL image or array is returned.
    arr = np.asarray(img).astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _valid_ranking(
    dist_row: np.ndarray,
    q_pid: int,
    q_cam: int,
    g_pids: np.ndarray,
    g_camids: np.ndarray,
) -> np.ndarray:
    """Rank the gallery for one query, dropping same-identity same-camera entries.

    Args:
        dist_row: Distances from the query to every gallery image.
        q_pid: Query identity.
        q_cam: Query camera.
        g_pids: Gallery identities.
        g_camids: Gallery cameras.

    Returns:
        Gallery indices sorted by ascending distance, with the trivial matches
        removed as the Market-1501 protocol requires.
    """
    order = np.argsort(dist_row, kind="stable")
    remove = (g_pids[order] == q_pid) & (g_camids[order] == q_cam)
    return order[~remove]


def _style_match(ax: Axes, color: str) -> None:
    """Draw a coloured border around an image axis and hide its ticks.

    Args:
        ax: A matplotlib ``Axes``.
        color: Border colour.
    """
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.5)
    ax.set_xticks([])
    ax.set_yticks([])


def visualize_ranked_results(
    distmat: np.ndarray,
    query_dataset: Dataset,
    gallery_dataset: Dataset,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    topk: int = 10,
    num_query: int = 5,
    save_path: str | Path | None = None,
    *,
    seed: int | None = 0,
) -> Figure | None:
    """Plot the top-k gallery matches for a sample of query images.

    For each selected query the gallery is ranked by ascending distance and the
    top-k matches are shown, outlined in green when the identity matches the
    query and in red otherwise.

    Args:
        distmat: Query-to-gallery distance matrix of shape ``[num_q, num_g]``,
            where smaller values denote closer matches.
        query_dataset: Dataset providing the query images, indexed in the same
            order as the rows of ``distmat``.
        gallery_dataset: Dataset providing the gallery images, indexed in the
            same order as the columns of ``distmat``.
        q_pids: Query person IDs, shape ``[num_q]``.
        g_pids: Gallery person IDs, shape ``[num_g]``.
        q_camids: Query camera IDs, shape ``[num_q]``.
        g_camids: Gallery camera IDs, shape ``[num_g]``.
        topk: Number of top gallery matches to display per query.
        num_query: Number of query images to visualize (randomly sampled).
        save_path: Optional path to save the figure. If ``None`` the figure is
            shown interactively.
        seed: Seed for the query sampling. ``None`` draws fresh randomness.

    Returns:
        The matplotlib figure, or ``None`` when there are no queries.
    """
    plt = pyplot()

    distmat = np.asarray(distmat)
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)

    num_q = distmat.shape[0]
    num_query = min(num_query, num_q)
    if num_query <= 0:
        return None
    topk = max(0, topk)

    rng = np.random.default_rng(seed)
    query_indices = rng.choice(num_q, size=num_query, replace=False)

    # One column for the query plus topk columns for the matches.
    n_cols = topk + 1
    fig, axes = plt.subplots(
        num_query, n_cols, figsize=(n_cols * 1.6, num_query * 3.2), squeeze=False
    )

    for row, q_idx in enumerate(query_indices):
        q_pid = q_pids[q_idx]
        ranked = _valid_ranking(distmat[q_idx], q_pid, q_camids[q_idx], g_pids, g_camids)

        ax_q = axes[row, 0]
        ax_q.imshow(_get_display_image(query_dataset, int(q_idx)))
        ax_q.set_title(f"Query\nID {q_pid}", fontsize=9, fontweight="bold")
        ax_q.axis("off")

        for k in range(topk):
            ax_m = axes[row, k + 1]
            if k >= len(ranked):
                ax_m.axis("off")
                continue
            g_idx = int(ranked[k])
            color = "green" if g_pids[g_idx] == q_pid else "red"
            ax_m.imshow(_get_display_image(gallery_dataset, g_idx))
            ax_m.set_title(f"R{k + 1}\nID {g_pids[g_idx]}", fontsize=8, color=color)
            _style_match(ax_m, color)

    fig.suptitle("Top-k Ranked Retrieval Results", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return finalize_figure(fig, save_path)


def plot_success_failure(
    distmat: np.ndarray,
    query_dataset: Dataset,
    gallery_dataset: Dataset,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    num_success: int = 5,
    num_fail: int = 5,
    save_path: str | Path | None = None,
    *,
    seed: int | None = 0,
    max_scan: int = 1000,
) -> Figure | None:
    """Plot curated Rank-1 success and failure cases.

    A random subset of queries is scanned until the requested number of correct
    (success) and incorrect (failure) Rank-1 retrievals has been collected.
    Each case shows the query next to its top-1 gallery match, coloured green
    for a success and red for a failure. Because the ranking comes from the
    same distance matrix as the reported metrics, a query shown as a failure is
    exactly a Rank-1 miss in the reported numbers.

    Args:
        distmat: Query-to-gallery distance matrix of shape ``[num_q, num_g]``.
        query_dataset: Dataset providing the query images (rows of ``distmat``).
        gallery_dataset: Dataset providing the gallery images (columns).
        q_pids: Query person IDs, shape ``[num_q]``.
        g_pids: Gallery person IDs, shape ``[num_g]``.
        q_camids: Query camera IDs, shape ``[num_q]``.
        g_camids: Gallery camera IDs, shape ``[num_g]``.
        num_success: Number of success cases to collect and display.
        num_fail: Number of failure cases to collect and display.
        save_path: Optional path to save the figure. If ``None`` the figure is
            shown interactively.
        seed: Seed for the order in which queries are scanned.
        max_scan: Maximum number of queries to scan.

    Returns:
        The matplotlib figure, or ``None`` when no case was found.
    """
    plt = pyplot()

    distmat = np.asarray(distmat)
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)

    num_q = distmat.shape[0]
    rng = np.random.default_rng(seed)
    scan_indices = rng.permutation(num_q)[: min(max_scan, num_q)]

    successes: list[tuple[int, int]] = []
    failures: list[tuple[int, int]] = []
    for q_idx in scan_indices:
        if len(successes) >= num_success and len(failures) >= num_fail:
            break
        ranked = _valid_ranking(distmat[q_idx], q_pids[q_idx], q_camids[q_idx], g_pids, g_camids)
        if ranked.size == 0:
            continue
        case = (int(q_idx), int(ranked[0]))
        if g_pids[ranked[0]] == q_pids[q_idx]:
            if len(successes) < num_success:
                successes.append(case)
        elif len(failures) < num_fail:
            failures.append(case)

    cases = [(c, "SUCCESS", "green") for c in successes] + [(c, "FAILURE", "red") for c in failures]
    if not cases:
        return None

    fig, axes = plt.subplots(len(cases), 2, figsize=(8, len(cases) * 3.5), squeeze=False)
    for row, ((q_idx, g_idx), label, color) in enumerate(cases):
        ax_q = axes[row, 0]
        ax_q.imshow(_get_display_image(query_dataset, q_idx))
        ax_q.set_title(
            f"{label}: Query\nID: {q_pids[q_idx]} | Cam: {q_camids[q_idx]}",
            color=color,
            fontweight="bold",
            fontsize=10,
        )
        ax_q.axis("off")

        ax_g = axes[row, 1]
        ax_g.imshow(_get_display_image(gallery_dataset, g_idx))
        ax_g.set_title(
            f"Rank-1 Match\nID: {g_pids[g_idx]} | Cam: {g_camids[g_idx]}",
            color=color,
            fontweight="bold",
            fontsize=10,
        )
        ax_g.axis("off")

    fig.tight_layout()
    return finalize_figure(fig, save_path)


def plot_rerank_impact(
    dist_orig: np.ndarray,
    dist_rerank: np.ndarray,
    query_dataset: Dataset,
    gallery_dataset: Dataset,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    num_examples: int = 3,
    save_path: str | Path | None = None,
    *,
    seed: int | None = 0,
) -> Figure | None:
    """Plot queries whose Rank-1 match is fixed by re-ranking.

    The function looks for queries whose top-1 gallery match is wrong under the
    original distance and right under the re-ranked distance, and shows each
    one as the query, the wrong match before re-ranking (red) and the correct
    match after re-ranking (green). It is a port of cell 12 of the original
    notebook.

    Args:
        dist_orig: Original query-to-gallery distance matrix ``[num_q, num_g]``.
        dist_rerank: Re-ranked distance matrix with the same shape.
        query_dataset: Dataset providing the query images (rows).
        gallery_dataset: Dataset providing the gallery images (columns).
        q_pids: Query person IDs, shape ``[num_q]``.
        g_pids: Gallery person IDs, shape ``[num_g]``.
        q_camids: Query camera IDs, shape ``[num_q]``.
        g_camids: Gallery camera IDs, shape ``[num_g]``.
        num_examples: Maximum number of corrected queries to show.
        save_path: Optional path to save the figure. If ``None`` the figure is
            shown interactively.
        seed: Seed for the order in which queries are scanned.

    Returns:
        The matplotlib figure, or ``None`` when re-ranking corrected no query.

    Raises:
        ValueError: If the two distance matrices have different shapes.
    """
    plt = pyplot()

    dist_orig = np.asarray(dist_orig)
    dist_rerank = np.asarray(dist_rerank)
    if dist_orig.shape != dist_rerank.shape:
        msg = f"Distance matrices differ in shape: {dist_orig.shape} vs {dist_rerank.shape}"
        raise ValueError(msg)
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)

    rng = np.random.default_rng(seed)
    fixed: list[tuple[int, int, int]] = []
    for q_idx in rng.permutation(dist_orig.shape[0]):
        if len(fixed) >= num_examples:
            break
        q_pid, q_cam = q_pids[q_idx], q_camids[q_idx]
        before = _valid_ranking(dist_orig[q_idx], q_pid, q_cam, g_pids, g_camids)
        after = _valid_ranking(dist_rerank[q_idx], q_pid, q_cam, g_pids, g_camids)
        if before.size == 0 or after.size == 0:
            continue
        if g_pids[before[0]] != q_pid and g_pids[after[0]] == q_pid:
            fixed.append((int(q_idx), int(before[0]), int(after[0])))

    if not fixed:
        return None

    fig, axes = plt.subplots(len(fixed), 3, figsize=(10, len(fixed) * 3.5), squeeze=False)
    for row, (q_idx, before_idx, after_idx) in enumerate(fixed):
        panels = (
            (query_dataset, q_idx, f"Query (ID {q_pids[q_idx]})", "black"),
            (gallery_dataset, before_idx, f"Before: WRONG\n(ID {g_pids[before_idx]})", "red"),
            (gallery_dataset, after_idx, f"After: CORRECT\n(ID {g_pids[after_idx]})", "green"),
        )
        for col, (dataset, idx, title, color) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(_get_display_image(dataset, idx))
            ax.set_title(title, color=color, fontweight="bold", fontsize=10)
            ax.axis("off")

    fig.suptitle("Rank-1 Corrected by Re-Ranking", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return finalize_figure(fig, save_path)
