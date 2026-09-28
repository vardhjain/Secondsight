"""k-reciprocal encoding re-ranking for person re-identification.

This module implements the re-ranking algorithm of Zhong et al.,
*"Re-ranking Person Re-identification with k-reciprocal Encoding"* (CVPR 2017),
following the reference implementation shipped with Luo et al.'s Bag of Tricks
strong baseline (``utils/re_ranking.py`` in ``michuanhaohao/reid-strong-baseline``).
Re-ranking refines the initial query-gallery distance matrix by blending the
original distance with a Jaccard distance computed over k-reciprocal
nearest-neighbour sets, which typically yields a large mAP improvement.

Like both reference implementations, the algorithm works on the *squared*
Euclidean distance. Negative squared distances caused by floating-point error
are clamped to zero. The results match the reference within float tolerance,
while the implementation differs in a few ways that save memory and time:

* Only the first ``max(k1 + 1, k2)`` neighbour indices of every row are kept,
  as ``int32``, instead of a full ``N x N`` ``int64`` argsort.
* Normalisation is done in place and only the query-gallery block of the final
  blend is materialised.
* The pairwise distance can optionally be computed on an accelerator via the
  ``device`` argument, while the sparse, index-heavy Jaccard step runs in NumPy.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import Tensor

__all__ = ["re_ranking"]

logger = logging.getLogger(__name__)

# Rows are argsorted in chunks of this many so the full ``N x N`` index matrix is
# never materialised at once.
_RANK_CHUNK = 2048


def _squared_distance(feat: Tensor) -> np.ndarray:
    """Return the clamped pairwise squared Euclidean distance as float32 NumPy.

    Args:
        feat: Feature tensor of shape ``(N, D)`` on any device.

    Returns:
        A float32 array of shape ``(N, N)`` with non-negative entries.
    """
    sq_sum = feat.pow(2).sum(dim=1, keepdim=True)
    distmat = sq_sum + sq_sum.t()
    distmat.addmm_(feat, feat.t(), beta=1, alpha=-2)
    distmat.clamp_(min=0.0)
    return distmat.cpu().numpy().astype(np.float32, copy=False)


def _truncated_rank(dist: np.ndarray, width: int) -> np.ndarray:
    """Return the first ``width`` columns of the row-wise argsort, as ``int32``.

    Each chunk of rows is sorted with the same algorithm as a full
    ``np.argsort(dist, axis=1)``, so the retained indices are identical to the
    leading columns of the full argsort.

    Args:
        dist: Distance matrix of shape ``(N, N)``.
        width: Number of leading neighbour indices to keep per row.

    Returns:
        An ``int32`` array of shape ``(N, width)``.
    """
    num = dist.shape[0]
    rank = np.empty((num, width), dtype=np.int32)
    for start in range(0, num, _RANK_CHUNK):
        stop = start + _RANK_CHUNK
        rank[start:stop] = np.argsort(dist[start:stop], axis=1)[:, :width]
    return rank


def re_ranking(
    qf: Tensor,
    gf: Tensor,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
    *,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Re-rank a query-gallery distance matrix with k-reciprocal encoding.

    Args:
        qf: Query feature tensor of shape ``(num_query, feat_dim)``.
        gf: Gallery feature tensor of shape ``(num_gallery, feat_dim)``.
        k1: Size of the k-reciprocal neighbourhood used to build the feature
            vectors ``V``.
        k2: Size of the local-query-expansion neighbourhood. ``k2 == 1``
            disables query expansion.
        lambda_value: Weight balancing the original distance against the
            Jaccard distance, so that ``final = lambda * original + (1 - lambda)
            * jaccard``. A value of ``0`` uses only the Jaccard distance.
        device: Optional device (for example ``"cuda"``) on which to compute the
            pairwise distance matrix. ``None`` keeps the features where they
            are. The Jaccard computation always runs on the CPU.

    Note:
        The distance used here is the squared Euclidean distance over the
        concatenated ``[query; gallery]`` features, and features are *not*
        normalized internally. For the result to be equivalent to a cosine
        re-ranking, and consistent with the base Euclidean metric used in
        :class:`~reid.evaluation.evaluator.Evaluator`, pass L2-normalized
        ``qf`` and ``gf`` (as ``Evaluator`` does when ``cfg.eval.feat_norm`` is
        ``True``).

    Returns:
        A float32 NumPy array of shape ``(num_query, num_gallery)`` containing
        the re-ranked distances (lower is more similar).

    Raises:
        ValueError: If ``k1`` or ``k2`` is smaller than 1, if ``lambda_value``
            is outside ``[0, 1]``, or if the feature shapes are incompatible.
    """
    if k1 < 1 or k2 < 1:
        raise ValueError(f"k1 and k2 must be >= 1, got k1={k1}, k2={k2}")
    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError(f"lambda_value must lie in [0, 1], got {lambda_value}")
    qf = torch.as_tensor(qf)
    gf = torch.as_tensor(gf)
    if qf.dim() != 2 or gf.dim() != 2 or qf.size(1) != gf.size(1):
        raise ValueError(
            f"qf and gf must be 2-D with the same feature dim, got {tuple(qf.shape)} "
            f"and {tuple(gf.shape)}"
        )

    query_num = qf.size(0)
    all_num = query_num + gf.size(0)
    feat = torch.cat([qf, gf], dim=0).float()
    if device is not None:
        feat = feat.to(device)

    logger.debug("Computing pairwise squared Euclidean distances for re-ranking...")
    original_dist = _squared_distance(feat)
    del feat

    # Normalise each column by its maximum, then transpose (paper convention).
    # An all-zero column only occurs on fully degenerate input; guarding it
    # avoids 0/0 producing NaN values that would spread through re-ranking.
    col_max = original_dist.max(axis=0)
    col_max[col_max == 0.0] = 1.0
    original_dist /= col_max
    original_dist = np.ascontiguousarray(original_dist.T)

    logger.debug("Computing k-reciprocal Jaccard distances...")
    initial_rank = _truncated_rank(original_dist, min(max(k1 + 1, k2), all_num))
    gallery_dist = np.zeros((all_num, all_num), dtype=np.float16)

    half_k1 = int(np.around(k1 / 2.0)) + 1
    for i in range(all_num):
        # k-reciprocal neighbours of probe i.
        forward_k_neigh = initial_rank[i, : k1 + 1]
        backward_k_neigh = initial_rank[forward_k_neigh, : k1 + 1]
        fi = np.where(backward_k_neigh == i)[0]
        k_reciprocal_index = forward_k_neigh[fi]

        # Expand the neighbour set using each neighbour's own reciprocal set.
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_forward = initial_rank[candidate, :half_k1]
            candidate_backward = initial_rank[candidate_forward, :half_k1]
            fi_candidate = np.where(candidate_backward == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward[fi_candidate]
            overlap = np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)
            if len(overlap) > 2.0 / 3.0 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        gallery_dist[i, k_reciprocal_expansion_index] = weight / np.sum(weight)

    # Only the query rows of the original distance are needed from here on.
    original_dist = original_dist[:query_num, query_num:].copy()

    # Local query expansion over the top-k2 neighbours.
    if k2 != 1:
        v_qe = np.zeros_like(gallery_dist)
        for i in range(all_num):
            v_qe[i, :] = np.mean(gallery_dist[initial_rank[i, :k2], :], axis=0)
        gallery_dist = v_qe
        del v_qe

    del initial_rank

    # Inverted index: for each column, which rows have a non-zero entry.
    inv_index = [np.where(gallery_dist[:, i] != 0)[0] for i in range(all_num)]

    jaccard_dist = np.zeros((query_num, all_num - query_num), dtype=np.float32)
    for i in range(query_num):
        temp_min = np.zeros(all_num, dtype=np.float32)
        ind_non_zero = np.where(gallery_dist[i, :] != 0)[0]
        for col in ind_non_zero:
            rows = inv_index[col]
            temp_min[rows] += np.minimum(gallery_dist[i, col], gallery_dist[rows, col])
        jaccard_dist[i] = (1.0 - temp_min / (2.0 - temp_min))[query_num:]

    final_dist = lambda_value * original_dist + (1.0 - lambda_value) * jaccard_dist
    return final_dist.astype(np.float32, copy=False)
