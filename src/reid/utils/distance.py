"""Distance / similarity utilities for feature matching.

These helpers compute pairwise distance matrices between two sets of feature
vectors. They are used both during training (the triplet loss relies on
:func:`euclidean_distance`) and at evaluation time (the ranking step). Only
``torch`` is required, so this module is safe to import in lightweight code
paths.
"""

from __future__ import annotations

import torch
from torch import Tensor


def euclidean_distance(x: Tensor, y: Tensor) -> Tensor:
    """Compute the pairwise Euclidean distance matrix between ``x`` and ``y``.

    Uses the numerically-stable expansion
    ``||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b`` followed by a clamp before the
    square root, so small negative values caused by floating-point error never
    produce ``NaN``.

    Args:
        x: Tensor of shape ``[M, D]``.
        y: Tensor of shape ``[N, D]``.

    Returns:
        Tensor of shape ``[M, N]`` where entry ``(i, j)`` is the Euclidean
        distance between ``x[i]`` and ``y[j]``.
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


def cosine_distance(x: Tensor, y: Tensor) -> Tensor:
    """Compute the pairwise cosine distance matrix between ``x`` and ``y``.

    Cosine distance is defined as ``1 - cosine_similarity``. Inputs are
    L2-normalized internally, so the result lies in ``[0, 2]``.

    Args:
        x: Tensor of shape ``[M, D]``.
        y: Tensor of shape ``[N, D]``.

    Returns:
        Tensor of shape ``[M, N]`` of cosine distances.
    """
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
    y_norm = torch.nn.functional.normalize(y, p=2, dim=1)
    sim = torch.matmul(x_norm, y_norm.t())
    return 1.0 - sim


def compute_distance_matrix(x: Tensor, y: Tensor, metric: str = "euclidean") -> Tensor:
    """Dispatch to the requested pairwise distance function.

    Args:
        x: Tensor of shape ``[M, D]``.
        y: Tensor of shape ``[N, D]``.
        metric: Either ``"euclidean"`` or ``"cosine"``.

    Returns:
        Tensor of shape ``[M, N]`` of pairwise distances.

    Raises:
        ValueError: If ``metric`` is not a supported value.
    """
    metric = metric.lower()
    if metric == "euclidean":
        return euclidean_distance(x, y)
    if metric == "cosine":
        return cosine_distance(x, y)
    raise ValueError(f"Unsupported distance metric: {metric!r}. Expected 'euclidean' or 'cosine'.")


__all__ = ["euclidean_distance", "cosine_distance", "compute_distance_matrix"]
