"""Tests for k-reciprocal re-ranking (:mod:`reid.evaluation.rerank`).

Light tests: only ``numpy`` and ``torch`` are required. Re-ranking is hard to
pin to exact values, so these assertions focus on the contract guarantees:
output shape, dtype, and numerical robustness (finite, no ``NaN`` -- the
negative-distance clamp before the square root must hold). A small sanity check
confirms that re-ranking preserves the correct nearest neighbour on an easily
separable, well-clustered toy problem.
"""

from __future__ import annotations

import numpy as np
import torch

from reid.evaluation.rerank import re_ranking


def _random_features(num_query: int, num_gallery: int, dim: int, seed: int = 0) -> tuple:
    """Create reproducible random query/gallery feature tensors.

    Args:
        num_query: Number of query vectors.
        num_gallery: Number of gallery vectors.
        dim: Feature dimensionality.
        seed: RNG seed.

    Returns:
        A ``(qf, gf)`` tuple of float32 tensors.
    """
    torch.manual_seed(seed)
    return torch.randn(num_query, dim), torch.randn(num_gallery, dim)


def test_rerank_shape_and_dtype() -> None:
    """Output is a NumPy array of shape ``(num_query, num_gallery)``."""
    qf, gf = _random_features(6, 20, 32)
    dist = re_ranking(qf, gf, k1=6, k2=3, lambda_value=0.3)
    assert isinstance(dist, np.ndarray)
    assert dist.shape == (6, 20)
    assert dist.dtype == np.float32


def test_rerank_is_finite_no_nan() -> None:
    """Re-ranked distances contain no ``NaN`` / ``inf`` values."""
    qf, gf = _random_features(8, 30, 16, seed=1)
    dist = re_ranking(qf, gf, k1=8, k2=3, lambda_value=0.3)
    assert np.isfinite(dist).all()
    assert not np.isnan(dist).any()


def test_rerank_k2_equals_one_disables_query_expansion() -> None:
    """``k2 == 1`` (no query expansion) still yields a valid finite matrix."""
    qf, gf = _random_features(5, 18, 24, seed=2)
    dist = re_ranking(qf, gf, k1=6, k2=1, lambda_value=0.3)
    assert dist.shape == (5, 18)
    assert np.isfinite(dist).all()


def test_rerank_preserves_correct_match_on_clustered_data() -> None:
    """On well-separated clusters re-ranking keeps the right nearest neighbour.

    Each query sits inside a tight cluster that has exactly one matching gallery
    item plus several gallery distractors from the same cluster, so the
    re-ranked top-1 should remain within the correct cluster.
    """
    torch.manual_seed(3)
    dim = 16
    # Two clusters centred far apart; build queries and gallery from each.
    centre_a = torch.zeros(dim)
    centre_b = torch.zeros(dim)
    centre_b[0] = 50.0  # large separation between clusters.

    qf = torch.stack([centre_a + 0.01 * torch.randn(dim), centre_b + 0.01 * torch.randn(dim)])
    gallery = []
    gallery_cluster = []
    for _ in range(8):
        gallery.append(centre_a + 0.01 * torch.randn(dim))
        gallery_cluster.append(0)
    for _ in range(8):
        gallery.append(centre_b + 0.01 * torch.randn(dim))
        gallery_cluster.append(1)
    gf = torch.stack(gallery)
    gallery_cluster_arr = np.asarray(gallery_cluster)

    dist = re_ranking(qf, gf, k1=4, k2=2, lambda_value=0.3)
    top1 = dist.argmin(axis=1)
    # Query 0 belongs to cluster 0, query 1 to cluster 1.
    assert gallery_cluster_arr[top1[0]] == 0
    assert gallery_cluster_arr[top1[1]] == 1
