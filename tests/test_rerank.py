"""Tests for k-reciprocal re-ranking (:mod:`reid.evaluation.rerank`).

Light tests that need only ``numpy`` and ``torch``. The golden tests compare
:func:`re_ranking` against a faithful NumPy port of the Bag of Tricks reference
implementation (``utils/re_ranking.py`` in ``michuanhaohao/reid-strong-baseline``),
which is embedded below, and check the ``lambda_value == 1`` case exactly.
Contract tests cover output shape and dtype, numerical robustness (negative
squared distances from floating-point error must be clamped to zero, so the
output stays finite), tiny populations and argument validation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reid.evaluation.rerank import re_ranking


def _reference_re_ranking(
    probe: np.ndarray, gallery: np.ndarray, k1: int, k2: int, lambda_value: float
) -> np.ndarray:
    """NumPy port of the Bag of Tricks reference ``re_ranking``.

    The logic mirrors the reference line by line, including its ``float16``
    intermediates. The only change is that the squared distance is computed in
    NumPy instead of with ``torch.addmm_``.
    """
    query_num = probe.shape[0]
    all_num = query_num + gallery.shape[0]
    feat = np.concatenate([probe, gallery]).astype(np.float32)
    sq = np.power(feat, 2).sum(axis=1, keepdims=True)
    original_dist = sq + sq.T - 2.0 * feat @ feat.T
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    v = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    half = int(np.around(k1 / 2)) + 1
    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, : k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :half]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index, :half]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            overlap = np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)
            if len(overlap) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        v[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        v_qe = np.zeros_like(v, dtype=np.float16)
        for i in range(all_num):
            v_qe[i, :] = np.mean(v[initial_rank[i, :k2], :], axis=0)
        v = v_qe
    inv_index = [np.where(v[:, i] != 0)[0] for i in range(gallery_num)]
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        ind_non_zero = np.where(v[i, :] != 0)[0]
        ind_images = [inv_index[ind] for ind in ind_non_zero]
        for j in range(len(ind_non_zero)):
            temp_min[0, ind_images[j]] = temp_min[0, ind_images[j]] + np.minimum(
                v[i, ind_non_zero[j]], v[ind_images[j], ind_non_zero[j]]
            )
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    return final_dist[:query_num, query_num:]


def _random_features(
    num_query: int, num_gallery: int, dim: int, seed: int = 0, normalize: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create reproducible random query and gallery feature tensors."""
    gen = torch.Generator().manual_seed(seed)
    qf = torch.randn(num_query, dim, generator=gen)
    gf = torch.randn(num_gallery, dim, generator=gen)
    if normalize:
        qf = torch.nn.functional.normalize(qf, dim=1)
        gf = torch.nn.functional.normalize(gf, dim=1)
    return qf, gf


@pytest.mark.parametrize(
    ("k1", "k2", "lam", "seed"),
    [(4, 2, 0.3, 0), (4, 1, 0.3, 1), (6, 3, 0.5, 2), (20, 6, 0.3, 3), (8, 4, 0.0, 4)],
)
def test_rerank_matches_reference(k1: int, k2: int, lam: float, seed: int) -> None:
    """Re-ranking agrees with the Bag of Tricks reference within float16 tolerance."""
    qf, gf = _random_features(6, 20, 8, seed=seed, normalize=True)
    ours = re_ranking(qf, gf, k1=k1, k2=k2, lambda_value=lam)
    ref = _reference_re_ranking(qf.numpy(), gf.numpy(), k1, k2, lam)
    np.testing.assert_allclose(ours, ref.astype(np.float32), atol=2e-3)


def test_rerank_lambda_one_equals_row_normalized_squared_distance() -> None:
    """With ``lambda_value == 1`` the output is the row-normalized squared distance."""
    qf, gf = _random_features(3, 5, 8, seed=5)
    feat = torch.cat([qf, gf]).double()
    full = (torch.cdist(feat, feat) ** 2).numpy()
    expected = (full / full.max(axis=1, keepdims=True))[:3, 3:]
    dist = re_ranking(qf, gf, k1=4, k2=1, lambda_value=1.0)
    np.testing.assert_allclose(dist, expected, atol=1e-5)


def test_rerank_lambda_zero_is_bounded_jaccard() -> None:
    """With ``lambda_value == 0`` the output is a Jaccard distance in ``[0, 1]``.

    The bounds allow a small slack because ``V`` is stored in float16, exactly as
    in the reference implementation.
    """
    qf, gf = _random_features(4, 12, 8, seed=6, normalize=True)
    dist = re_ranking(qf, gf, k1=4, k2=2, lambda_value=0.0)
    assert float(dist.min()) >= -1e-3
    assert float(dist.max()) <= 1.0 + 1e-3


def test_rerank_device_argument_gives_same_result() -> None:
    """Passing an explicit device does not change the result."""
    qf, gf = _random_features(4, 10, 8, seed=7, normalize=True)
    base = re_ranking(qf, gf, k1=4, k2=2, lambda_value=0.3)
    on_cpu = re_ranking(qf, gf, k1=4, k2=2, lambda_value=0.3, device="cpu")
    np.testing.assert_array_equal(base, on_cpu)


def test_rerank_shape_and_dtype() -> None:
    """Output is a float32 NumPy array of shape ``(num_query, num_gallery)``."""
    qf, gf = _random_features(6, 20, 32)
    dist = re_ranking(qf, gf, k1=6, k2=3, lambda_value=0.3)
    assert isinstance(dist, np.ndarray)
    assert dist.shape == (6, 20)
    assert dist.dtype == np.float32


def test_rerank_is_finite() -> None:
    """Re-ranked distances contain no ``NaN`` or ``inf`` values."""
    qf, gf = _random_features(8, 30, 16, seed=1)
    dist = re_ranking(qf, gf, k1=8, k2=3, lambda_value=0.3)
    assert np.isfinite(dist).all()


def test_rerank_duplicate_features_stay_finite() -> None:
    """Identical vectors (zero distance, prone to negative rounding) stay finite."""
    qf, _ = _random_features(3, 1, 16, seed=8, normalize=True)
    gf = torch.cat([qf, qf])
    dist = re_ranking(qf, gf, k1=2, k2=2, lambda_value=0.3)
    assert np.isfinite(dist).all()


def test_rerank_k_larger_than_population() -> None:
    """Neighbourhood sizes larger than the population still give a valid matrix."""
    qf, gf = _random_features(2, 3, 8, seed=9)
    dist = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
    assert dist.shape == (2, 3)
    assert np.isfinite(dist).all()


def test_rerank_k2_equals_one_disables_query_expansion() -> None:
    """``k2 == 1`` (no query expansion) still yields a valid finite matrix."""
    qf, gf = _random_features(5, 18, 24, seed=2)
    dist = re_ranking(qf, gf, k1=6, k2=1, lambda_value=0.3)
    assert dist.shape == (5, 18)
    assert np.isfinite(dist).all()


@pytest.mark.parametrize(
    "kwargs",
    [{"k1": 0}, {"k2": 0}, {"lambda_value": -0.1}, {"lambda_value": 1.5}],
)
def test_rerank_rejects_invalid_arguments(kwargs: dict[str, float]) -> None:
    """Out-of-range neighbourhood sizes or blend weights raise ``ValueError``."""
    qf, gf = _random_features(2, 4, 8)
    with pytest.raises(ValueError):
        re_ranking(qf, gf, **kwargs)  # type: ignore[arg-type]


def test_rerank_rejects_mismatched_feature_dims() -> None:
    """Query and gallery features must share the feature dimension."""
    with pytest.raises(ValueError):
        re_ranking(torch.randn(2, 8), torch.randn(4, 6))


def test_rerank_preserves_correct_match_on_clustered_data() -> None:
    """On well-separated clusters re-ranking keeps the right nearest neighbour.

    Each query sits inside a tight cluster, and the gallery holds 8 items per
    cluster, so the re-ranked top-1 should remain within the query's cluster.
    """
    torch.manual_seed(3)
    dim = 16
    centre_a = torch.zeros(dim)
    centre_b = torch.zeros(dim)
    centre_b[0] = 50.0  # large separation between clusters.

    qf = torch.stack([centre_a + 0.01 * torch.randn(dim), centre_b + 0.01 * torch.randn(dim)])
    gallery = [centre_a + 0.01 * torch.randn(dim) for _ in range(8)]
    gallery += [centre_b + 0.01 * torch.randn(dim) for _ in range(8)]
    gf = torch.stack(gallery)
    gallery_cluster = np.asarray([0] * 8 + [1] * 8)

    dist = re_ranking(qf, gf, k1=4, k2=2, lambda_value=0.3)
    top1 = dist.argmin(axis=1)
    assert gallery_cluster[top1[0]] == 0
    assert gallery_cluster[top1[1]] == 1
