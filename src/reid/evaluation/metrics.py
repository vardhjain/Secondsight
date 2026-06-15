"""Market-1501 evaluation metrics (CMC and mean Average Precision).

This module implements the standard Market-1501 single-query evaluation
protocol used to score a person re-identification model. For every query
image we rank all gallery images by ascending distance, discard gallery
images that share *both* the query identity and the query camera (the
trivial same-camera matches), and then compute:

* the Cumulative Matching Characteristic (CMC) curve, and
* the Average Precision (AP), averaged across queries to give the mAP.

The implementation depends only on NumPy so that it can run in lightweight
environments without ``torch``/``torchvision`` installed. It is a cleaned-up
port of the evaluation cell from the original research notebook.
"""

from __future__ import annotations

import numpy as np

__all__ = ["compute_cmc_map", "compute_ap_per_query"]


def _build_match_matrix(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank the gallery for every query and flag the correct matches.

    Args:
        distmat: Distance matrix of shape ``(num_query, num_gallery)`` where a
            smaller value means a closer (more similar) pair.
        q_pids: Query person identities, shape ``(num_query,)``.
        g_pids: Gallery person identities, shape ``(num_gallery,)``.

    Returns:
        A tuple ``(indices, matches)`` where ``indices`` is the per-row
        ascending argsort of ``distmat`` (shape ``(num_query, num_gallery)``)
        and ``matches`` is a ``(num_query, num_gallery)`` int array that is 1
        where the ranked gallery identity equals the query identity.
    """
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    return indices, matches


def _raw_cmc_for_query(
    q_pid: int,
    q_camid: int,
    order: np.ndarray,
    row_matches: np.ndarray,
    g_pids: np.ndarray,
    g_camids: np.ndarray,
) -> np.ndarray:
    """Return the ranked match vector for one query after same-camera exclusion.

    Removes gallery samples that share both the query identity and the query
    camera, then returns the kept match flags in ranked order.
    """
    remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
    keep = np.invert(remove)
    return row_matches[keep]


def _average_precision(raw_cmc: np.ndarray) -> float:
    """Average Precision for a single query from its ranked match vector.

    ``raw_cmc`` must contain at least one positive (caller checks ``np.any``).
    """
    num_rel = raw_cmc.sum()
    tmp_cmc = raw_cmc.cumsum()
    precision = [x / (rank + 1.0) for rank, x in enumerate(tmp_cmc)]
    precision = np.asarray(precision) * raw_cmc
    return float(precision.sum() / num_rel)


def compute_cmc_map(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = 50,
) -> tuple[np.ndarray, float]:
    """Compute the CMC curve and mAP under the Market-1501 protocol.

    For each query the gallery samples that share the query's identity *and*
    camera are removed before scoring (same-camera same-identity exclusion).
    Queries that have no valid gallery match after this filtering are skipped,
    matching the reference Market-1501 evaluation code.

    Args:
        distmat: Distance matrix of shape ``(num_query, num_gallery)``; lower
            means more similar.
        q_pids: Query person identities, shape ``(num_query,)``.
        g_pids: Gallery person identities, shape ``(num_gallery,)``.
        q_camids: Query camera ids, shape ``(num_query,)``.
        g_camids: Gallery camera ids, shape ``(num_gallery,)``.
        max_rank: Length of the returned CMC curve. It is clamped to the
            number of gallery samples if the gallery is smaller.

    Returns:
        A tuple ``(cmc, mAP)`` where ``cmc`` is a float array of length
        ``max_rank`` (CMC accuracy at ranks ``1..max_rank``) and ``mAP`` is the
        mean Average Precision as a Python float.

    Raises:
        RuntimeError: If no query has a valid gallery match (e.g. all queries
            are filtered out by the same-camera exclusion).
    """
    distmat = np.asarray(distmat)
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)

    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g

    indices, matches = _build_match_matrix(distmat, q_pids, g_pids)

    all_cmc: list[np.ndarray] = []
    all_ap: list[float] = []
    num_valid_q = 0.0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        raw_cmc = _raw_cmc_for_query(q_pid, q_camid, order, matches[q_idx], g_pids, g_camids)
        if not np.any(raw_cmc):
            # This query has no true match in the gallery; skip it.
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        row = cmc[:max_rank]
        if row.shape[0] < max_rank:
            # A query whose post-exclusion gallery is shorter than max_rank still
            # counts as matched at every higher rank once it has hit, so pad with
            # the final value (forward-fill) rather than zeros.
            row = np.concatenate([row, np.full(max_rank - row.shape[0], row[-1], dtype=row.dtype)])
        all_cmc.append(row)
        num_valid_q += 1.0

        all_ap.append(_average_precision(raw_cmc))

    if num_valid_q == 0.0:
        raise RuntimeError(
            "No valid query found; check that gallery contains matches for "
            "the queries and that camera/identity ids are correct."
        )

    cmc_array = np.asarray(all_cmc).astype(np.float32)
    cmc_array = cmc_array.sum(axis=0) / num_valid_q
    mean_ap = float(np.mean(all_ap))
    return cmc_array, mean_ap


def compute_ap_per_query(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
) -> np.ndarray:
    """Compute the Average Precision for each individual query.

    This is a helper for analysis/visualisation (e.g. plotting the
    distribution of per-query AP values). It uses the same same-camera
    same-identity exclusion as :func:`compute_cmc_map`. Queries without a valid
    gallery match are skipped and therefore do not appear in the output array.

    Args:
        distmat: Distance matrix of shape ``(num_query, num_gallery)``; lower
            means more similar.
        q_pids: Query person identities, shape ``(num_query,)``.
        g_pids: Gallery person identities, shape ``(num_gallery,)``.
        q_camids: Query camera ids, shape ``(num_query,)``.
        g_camids: Gallery camera ids, shape ``(num_gallery,)``.

    Returns:
        A float array of per-query AP values; its length equals the number of
        valid queries (``<= num_query``).
    """
    distmat = np.asarray(distmat)
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    q_camids = np.asarray(q_camids)
    g_camids = np.asarray(g_camids)

    num_q = distmat.shape[0]
    indices, matches = _build_match_matrix(distmat, q_pids, g_pids)

    all_ap: list[float] = []
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        raw_cmc = _raw_cmc_for_query(q_pid, q_camid, order, matches[q_idx], g_pids, g_camids)
        if not np.any(raw_cmc):
            continue

        all_ap.append(_average_precision(raw_cmc))

    return np.asarray(all_ap, dtype=np.float32)
