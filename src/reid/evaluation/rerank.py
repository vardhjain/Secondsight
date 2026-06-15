"""k-reciprocal encoding re-ranking for person re-identification.

This is a robust port of the re-ranking algorithm from Zhong et al.,
*"Re-ranking Person Re-identification with k-reciprocal Encoding"* (CVPR 2017),
cleaned up from the original research notebook. Re-ranking refines the initial
query-gallery distance matrix by combining the original Euclidean distance with
a Jaccard distance computed over k-reciprocal nearest-neighbour sets, which
typically yields a large mAP improvement.

The implementation operates on ``torch`` feature tensors for the distance
computation and falls back to NumPy for the (sparse, index-heavy) Jaccard
computation. Negative squared distances are clamped to zero before the square
root to avoid ``NaN`` values caused by floating-point error.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import Tensor

__all__ = ["re_ranking"]

logger = logging.getLogger(__name__)


def re_ranking(
    qf: Tensor,
    gf: Tensor,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
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
            Jaccard distance. ``final = lambda * original + (1 - lambda) *
            jaccard``. ``0`` uses only the Jaccard distance.

    Note:
        The distance used here is a plain (squared) Euclidean distance over the
        concatenated ``[query; gallery]`` features; features are *not* normalized
        internally. For the result to be equivalent to a cosine re-ranking -- and
        consistent with the base Euclidean metric used in
        :class:`~reid.evaluation.evaluator.Evaluator` -- pass L2-normalized
        ``qf`` / ``gf`` (as ``Evaluator`` does when ``cfg.eval.feat_norm`` is
        ``True``). The algorithm still runs on un-normalized inputs, but the
        distances are then ordinary Euclidean rather than cosine-equivalent.

    Returns:
        A NumPy array of shape ``(num_query, num_gallery)`` containing the
        re-ranked distances (lower is more similar).
    """
    query_num = qf.size(0)
    all_num = query_num + gf.size(0)
    feat = torch.cat([qf, gf], dim=0)

    logger.debug("Computing pairwise Euclidean distances for re-ranking...")
    # Squared Euclidean distance over the concatenated [query; gallery] set.
    sq_sum = torch.pow(feat, 2).sum(dim=1, keepdim=True)
    distmat = sq_sum.expand(all_num, all_num) + sq_sum.expand(all_num, all_num).t()
    distmat.addmm_(feat, feat.t(), beta=1, alpha=-2)

    # Clamp negative values (floating-point noise) to 0 before the sqrt.
    original_dist = distmat.cpu().numpy()
    original_dist = np.maximum(original_dist, 0.0)
    del distmat

    original_dist = np.power(original_dist, 0.5)
    # Normalise each column by its maximum, then transpose (paper convention).
    # Guard against an all-zero column (only possible on fully degenerate,
    # all-identical input) to avoid 0/0 -> NaN propagating through re-ranking.
    col_max = np.max(original_dist, axis=0)
    col_max[col_max == 0.0] = 1.0
    original_dist = original_dist / col_max
    original_dist = original_dist.T

    logger.debug("Computing k-reciprocal Jaccard distances...")
    gallery_dist = np.zeros_like(original_dist, dtype=np.float16)
    initial_rank = np.argsort(original_dist, axis=1)

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

    original_dist = original_dist[:query_num, :]

    # Local query expansion over the top-k2 neighbours.
    if k2 != 1:
        v_qe = np.zeros_like(gallery_dist, dtype=np.float16)
        for i in range(all_num):
            v_qe[i, :] = np.mean(gallery_dist[initial_rank[i, :k2], :], axis=0)
        gallery_dist = v_qe
        del v_qe

    del initial_rank

    # Inverted index: for each column, which rows have a non-zero entry.
    inv_index = [np.where(gallery_dist[:, i] != 0)[0] for i in range(all_num)]

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        ind_non_zero = np.where(gallery_dist[i, :] != 0)[0]
        ind_images = [inv_index[ind] for ind in ind_non_zero]
        for j in range(len(ind_non_zero)):
            temp_min[0, ind_images[j]] = temp_min[0, ind_images[j]] + np.minimum(
                gallery_dist[i, ind_non_zero[j]],
                gallery_dist[ind_images[j], ind_non_zero[j]],
            )
        jaccard_dist[i] = 1 - temp_min / (2.0 - temp_min)

    final_dist = lambda_value * original_dist + (1.0 - lambda_value) * jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist
