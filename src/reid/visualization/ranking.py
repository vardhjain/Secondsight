"""Ranked-result visualization utilities for Person Re-Identification.

This module renders qualitative Re-ID results: galleries of the top-k matches
for a set of query images, and curated success-versus-failure panels. These
plots are the most recruiter-legible artifacts of a Re-ID system because they
show, image-by-image, whether the model retrieves the correct identity across
cameras.

The logic is ported from cells 8 and 11 of the original research notebook,
generalized to operate on any :class:`reid.data.dataset.Market1501` query and
gallery datasets and to optionally persist figures to disk.

Matplotlib is imported lazily (inside functions) to avoid forcing a GUI
backend at package import time; figures are saved with ``savefig`` and the
default backend is respected so the same code works in scripts and notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import Tensor, nn

    from reid.data.dataset import Market1501

# ImageNet normalization statistics, used to invert the test-time transform so
# tensors can be rendered as human-viewable images.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _denormalize_to_image(tensor: Tensor) -> np.ndarray:
    """Inverts ImageNet normalization and converts a CHW tensor to an image.

    Args:
        tensor: A normalized image tensor of shape ``[3, H, W]``.

    Returns:
        An RGB image as a ``float32`` numpy array of shape ``[H, W, 3]`` with
        values clipped to ``[0, 1]``.
    """
    mean = torch.tensor(_IMAGENET_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=tensor.dtype).view(3, 1, 1)
    img = tensor.detach().cpu() * std + mean
    img = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _get_display_image(dataset: Market1501, idx: int) -> np.ndarray:
    """Returns a viewable RGB image for a dataset index.

    The dataset's transform yields a normalized tensor; this helper inverts
    that normalization so the result can be passed straight to ``imshow``.

    Args:
        dataset: A :class:`~reid.data.dataset.Market1501` instance.
        idx: Index of the sample to render.

    Returns:
        An RGB image as a ``float32`` numpy array of shape ``[H, W, 3]`` in
        ``[0, 1]``.
    """
    sample = dataset[idx]
    img = sample[0]
    if isinstance(img, torch.Tensor):
        return _denormalize_to_image(img)
    # Fall back gracefully if a raw PIL image / array is returned.
    arr = np.asarray(img).astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def visualize_ranked_results(
    distmat: np.ndarray,
    query_dataset: Market1501,
    gallery_dataset: Market1501,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    topk: int = 10,
    num_query: int = 5,
    save_path: str | Path | None = None,
) -> None:
    """Plots top-k gallery matches for a sample of query images.

    For each selected query, the gallery is ranked by ascending distance and
    the top-k matches are shown. Following the standard Market-1501 protocol,
    gallery images with the *same identity and same camera* as the query are
    excluded from the ranking (these are trivial matches). Each retrieved
    match is outlined in green if its identity equals the query's (correct) and
    red otherwise (incorrect).

    Args:
        distmat: Query-to-gallery distance matrix of shape ``[num_q, num_g]``;
            smaller values denote closer matches.
        query_dataset: Dataset providing the query images for display. Indexed
            in the same order as the rows of ``distmat``.
        gallery_dataset: Dataset providing the gallery images for display.
            Indexed in the same order as the columns of ``distmat``.
        q_pids: Query person IDs, shape ``[num_q]``.
        g_pids: Gallery person IDs, shape ``[num_g]``.
        q_camids: Query camera IDs, shape ``[num_q]``.
        g_camids: Gallery camera IDs, shape ``[num_g]``.
        topk: Number of top gallery matches to display per query.
        num_query: Number of query images to visualize (randomly sampled).
        save_path: Optional path to save the figure. If ``None`` the figure is
            shown interactively via ``plt.show()``.

    Returns:
        None. The figure is either saved to ``save_path`` or shown.
    """
    import matplotlib.pyplot as plt

    distmat = np.asarray(distmat)
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)

    num_q = distmat.shape[0]
    num_query = min(num_query, num_q)
    if num_query <= 0:
        return

    rng = np.random.default_rng()
    query_indices = rng.choice(num_q, size=num_query, replace=False)

    # One column for the query plus topk columns for the matches.
    n_cols = topk + 1
    fig, axes = plt.subplots(
        num_query,
        n_cols,
        figsize=(n_cols * 1.6, num_query * 3.2),
    )
    axes = np.atleast_2d(axes)

    for row, q_idx in enumerate(query_indices):
        q_pid = q_pids[q_idx]
        q_cam = q_camids[q_idx]

        order = np.argsort(distmat[q_idx])
        # Exclude trivial same-id/same-cam gallery entries.
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_cam)
        ranked = order[~remove]

        # Render the query.
        ax_q = axes[row, 0]
        ax_q.imshow(_get_display_image(query_dataset, int(q_idx)))
        ax_q.set_title(f"Query\nID {q_pid}", fontsize=9, fontweight="bold")
        ax_q.axis("off")

        # Render the top-k matches.
        for k in range(topk):
            ax_m = axes[row, k + 1]
            if k < len(ranked):
                g_idx = int(ranked[k])
                is_correct = bool(g_pids[g_idx] == q_pid)
                color = "green" if is_correct else "red"
                ax_m.imshow(_get_display_image(gallery_dataset, g_idx))
                ax_m.set_title(f"R{k + 1}\nID {g_pids[g_idx]}", fontsize=8, color=color)
                # Colored border to flag correctness.
                for spine in ax_m.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2.5)
                ax_m.set_xticks([])
                ax_m.set_yticks([])
            else:
                ax_m.axis("off")

    fig.suptitle("Top-k Ranked Retrieval Results", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    _finalize(fig, save_path)


def plot_success_failure(
    model: nn.Module,
    query_dataset: Market1501,
    gallery_dataset: Market1501,
    device: str | torch.device,
    num_success: int = 5,
    num_fail: int = 5,
    save_path: str | Path | None = None,
) -> None:
    """Plots curated Rank-1 success and failure cases.

    Gallery features are extracted once, then a random subset of queries is
    scanned until the requested number of correct (success) and incorrect
    (failure) Rank-1 retrievals have been collected. Each case shows the query
    image alongside its top-1 gallery match, color-coded green (success) or red
    (failure). The matching protocol excludes same-identity/same-camera gallery
    entries, consistent with the Market-1501 evaluation rules.

    This is a faithful port of cell 11 of the original notebook, refactored to
    accept an explicit ``device`` and optional ``save_path``.

    Args:
        model: The trained Re-ID model. Set to eval mode internally.
        query_dataset: Query dataset yielding ``(tensor, pid, camid)`` samples.
        gallery_dataset: Gallery dataset yielding ``(tensor, pid, camid)``
            samples.
        device: Torch device (or device string) for feature extraction.
        num_success: Number of success cases to collect and display.
        num_fail: Number of failure cases to collect and display.
        save_path: Optional path to save the figure. If ``None`` the figure is
            shown interactively.

    Returns:
        None. The figure is either saved to ``save_path`` or shown.
    """
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    model.eval()
    device = torch.device(device) if isinstance(device, str) else device

    # Extract gallery features in batches for speed.
    gallery_loader = DataLoader(gallery_dataset, batch_size=256, shuffle=False)
    gf_list: list[Tensor] = []
    g_pids_list: list[int] = []
    g_camids_list: list[int] = []
    with torch.no_grad():
        for inputs, pids, cams in gallery_loader:
            inputs = inputs.to(device)
            feat = model(inputs)
            gf_list.append(feat.cpu())
            g_pids_list.extend([int(p) for p in pids.numpy()])
            g_camids_list.extend([int(c) for c in cams.numpy()])
    gf = torch.cat(gf_list, dim=0)
    g_pids = np.asarray(g_pids_list)
    g_camids = np.asarray(g_camids_list)

    # Scan a random subset of queries for hard examples.
    num_q = len(query_dataset)
    scan_size = min(1000, num_q)
    rng = np.random.default_rng()
    scan_indices = rng.choice(num_q, size=scan_size, replace=False)

    successes: list[tuple[int, int, int, int, int, int]] = []
    failures: list[tuple[int, int, int, int, int, int]] = []

    gf_sq = gf.pow(2).sum(dim=1)  # precompute gallery squared norms

    with torch.no_grad():
        for idx in scan_indices:
            if len(successes) >= num_success and len(failures) >= num_fail:
                break

            img_t, q_pid, q_camid = query_dataset[int(idx)]
            q_pid, q_camid = int(q_pid), int(q_camid)
            img_t = img_t.unsqueeze(0).to(device)
            q_feat = model(img_t).cpu()

            # Squared Euclidean distance to the whole gallery: |q|^2 + |g|^2 - 2 q.g
            q_sq = q_feat.pow(2).sum(dim=1)
            dist = q_sq + gf_sq - 2.0 * (q_feat @ gf.t()).squeeze(0)
            dist = dist.numpy()

            order = np.argsort(dist)
            best_match_idx = -1
            for cand in order:
                if g_pids[cand] == q_pid and g_camids[cand] == q_camid:
                    continue
                best_match_idx = int(cand)
                break
            if best_match_idx == -1:
                continue

            pred_pid = int(g_pids[best_match_idx])
            info = (
                int(idx),
                best_match_idx,
                q_pid,
                pred_pid,
                q_camid,
                int(g_camids[best_match_idx]),
            )
            if pred_pid == q_pid and len(successes) < num_success:
                successes.append(info)
            elif pred_pid != q_pid and len(failures) < num_fail:
                failures.append(info)

    all_cases = successes + failures
    labels = ["SUCCESS"] * len(successes) + ["FAILURE"] * len(failures)
    colors = ["green"] * len(successes) + ["red"] * len(failures)
    total_rows = len(all_cases)

    if total_rows == 0:
        return

    fig, axes = plt.subplots(total_rows, 2, figsize=(8, total_rows * 3.5))
    axes = np.atleast_2d(axes)

    for i, (q_idx, g_idx, q_pid, pred_pid, q_cam, g_cam) in enumerate(all_cases):
        ax_q = axes[i, 0]
        ax_q.imshow(_get_display_image(query_dataset, q_idx))
        ax_q.set_title(
            f"{labels[i]}: Query\nID: {q_pid} | Cam: {q_cam}",
            color=colors[i],
            fontweight="bold",
            fontsize=10,
        )
        ax_q.axis("off")

        ax_g = axes[i, 1]
        ax_g.imshow(_get_display_image(gallery_dataset, g_idx))
        ax_g.set_title(
            f"Rank-1 Match\nID: {pred_pid} | Cam: {g_cam}",
            color=colors[i],
            fontweight="bold",
            fontsize=10,
        )
        ax_g.axis("off")

    fig.tight_layout()
    _finalize(fig, save_path)


def _finalize(fig, save_path: str | Path | None) -> None:
    """Saves a figure to disk or shows it, then closes it when saving.

    Args:
        fig: The matplotlib figure to finalize.
        save_path: Destination path; if ``None`` the figure is shown.
    """
    import matplotlib.pyplot as plt

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
