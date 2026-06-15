"""Tests for the Market-1501 metrics (:mod:`reid.evaluation.metrics`).

Light tests: only ``numpy`` is required. The expected CMC and mAP values below
are computed by hand on tiny, fully tractable distance matrices, so any drift in
the metric implementation is caught exactly.

Hand-derivation reference for the primary case
(:func:`test_cmc_map_exact_values`)::

    queries: q0 -> pid 0, q1 -> pid 1 (both camera 2)
    gallery: g0,g1 -> pid 0 (cams 0,1); g2,g3 -> pid 1 (cams 0,1)
    No gallery item is excluded (query camera 2 is unique), so all 4 gallery
    items are kept for every query.

    q0 ranked relevance (by ascending distance): [1, 0, 1, 0]
        AP = (1/1 + 2/3) / 2 = 0.8333...
    q1 ranked relevance:                         [1, 0, 0, 1]
        AP = (1/1 + 2/4) / 2 = 0.75
    mAP = (0.8333... + 0.75) / 2 = 0.79166...
    Both queries hit at rank 1 -> CMC = [1, 1, 1, 1].
"""

from __future__ import annotations

import numpy as np
import pytest

from reid.evaluation.metrics import compute_ap_per_query, compute_cmc_map

# Shared "no exclusion" geometry used by several tests: distinct query camera so
# the same-camera same-identity filter removes nothing and all queries keep the
# full gallery (uniform CMC length).
_Q_PIDS = np.array([0, 1])
_Q_CAMIDS = np.array([2, 2])
_G_PIDS = np.array([0, 0, 1, 1])
_G_CAMIDS = np.array([0, 1, 0, 1])


def test_cmc_map_exact_values() -> None:
    """Exact CMC / mAP on a tiny matrix where both queries hit at rank 1."""
    distmat = np.array(
        [
            [0.10, 0.40, 0.20, 0.50],  # q0 (pid 0): relevance [1,0,1,0]
            [0.30, 0.20, 0.10, 0.60],  # q1 (pid 1): relevance [1,0,0,1]
        ],
        dtype=np.float32,
    )
    cmc, mean_ap = compute_cmc_map(distmat, _Q_PIDS, _G_PIDS, _Q_CAMIDS, _G_CAMIDS, max_rank=4)

    assert cmc.shape == (4,)
    np.testing.assert_allclose(cmc, [1.0, 1.0, 1.0, 1.0], atol=1e-6)
    assert mean_ap == pytest.approx((5.0 / 6.0 + 0.75) / 2.0, abs=1e-6)


def test_per_query_ap_exact_values() -> None:
    """Per-query AP values match the hand derivation."""
    distmat = np.array(
        [
            [0.10, 0.40, 0.20, 0.50],
            [0.30, 0.20, 0.10, 0.60],
        ],
        dtype=np.float32,
    )
    aps = compute_ap_per_query(distmat, _Q_PIDS, _G_PIDS, _Q_CAMIDS, _G_CAMIDS)
    assert aps.shape == (2,)
    np.testing.assert_allclose(aps, [5.0 / 6.0, 0.75], atol=1e-6)


def test_cmc_discriminates_ranks() -> None:
    """A rank-1 miss that recovers at rank 2 yields CMC ``[0.5, 1, ...]``."""
    distmat = np.array(
        [
            [0.10, 0.40, 0.20, 0.50],  # q0 (pid 0): rank-1 correct, relevance [1,0,1,0]
            [0.10, 0.30, 0.20, 0.60],  # q1 (pid 1): rank-1 WRONG, rank-2 correct [0,1,0,1]
        ],
        dtype=np.float32,
    )
    cmc, mean_ap = compute_cmc_map(distmat, _Q_PIDS, _G_PIDS, _Q_CAMIDS, _G_CAMIDS, max_rank=4)

    np.testing.assert_allclose(cmc, [0.5, 1.0, 1.0, 1.0], atol=1e-6)
    # q0 AP = (1/1 + 2/3)/2 = 0.8333..., q1 AP = (1/2 + 2/4)/2 = 0.5.
    assert mean_ap == pytest.approx((5.0 / 6.0 + 0.5) / 2.0, abs=1e-6)


def test_same_camera_same_id_exclusion() -> None:
    """The nearest gallery item sharing pid *and* camera is excluded.

    Each query's closest gallery item (distance ``0.05``) shares both the query
    identity and camera and must be filtered out before scoring, which drops the
    rank-1 accuracy below 1.
    """
    distmat = np.array(
        [
            [0.05, 0.40, 0.20, 0.50],
            [0.05, 0.40, 0.20, 0.50],
        ],
        dtype=np.float32,
    )
    q_pids = np.array([0, 1])
    q_camids = np.array([0, 0])
    g_pids = np.array([0, 1, 0, 1])
    g_camids = np.array([0, 0, 1, 1])

    cmc, _ = compute_cmc_map(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=3)
    # q0: after removing g0 (pid0/cam0), nearest kept is g2 (pid0) -> rank-1 hit.
    # q1: after removing g1 (pid1/cam0), nearest kept is g2 (pid0) -> rank-1 miss.
    np.testing.assert_allclose(cmc[0], 0.5, atol=1e-6)


def test_max_rank_clamped_to_gallery_size() -> None:
    """``max_rank`` larger than the gallery is clamped to the gallery size."""
    distmat = np.array(
        [
            [0.10, 0.40, 0.20, 0.50],
            [0.30, 0.20, 0.10, 0.60],
        ],
        dtype=np.float32,
    )
    cmc, _ = compute_cmc_map(distmat, _Q_PIDS, _G_PIDS, _Q_CAMIDS, _G_CAMIDS, max_rank=50)
    assert cmc.shape == (4,)  # clamped from 50 to num_gallery == 4.


def test_no_valid_query_raises_runtime_error() -> None:
    """If every query is filtered out, a :class:`RuntimeError` is raised."""
    distmat = np.array([[0.1, 0.2]], dtype=np.float32)
    q_pids = np.array([5])
    q_camids = np.array([1])
    g_pids = np.array([5, 9])  # the only same-id gallery item shares the camera.
    g_camids = np.array([1, 2])
    with pytest.raises(RuntimeError):
        compute_cmc_map(distmat, q_pids, g_pids, q_camids, g_camids)


def test_perfect_ranking_gives_unit_map() -> None:
    """A perfect ranking yields mAP == 1.0 and rank-1 == 1.0."""
    # Each query's single relevant gallery item is strictly the closest.
    distmat = np.array(
        [
            [0.10, 0.90],  # q0 (pid 0): g0 (pid 0) closest.
            [0.90, 0.10],  # q1 (pid 1): g1 (pid 1) closest.
        ],
        dtype=np.float32,
    )
    q_pids = np.array([0, 1])
    q_camids = np.array([5, 5])
    g_pids = np.array([0, 1])
    g_camids = np.array([0, 0])
    cmc, mean_ap = compute_cmc_map(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=2)
    assert mean_ap == pytest.approx(1.0, abs=1e-6)
    assert cmc[0] == pytest.approx(1.0, abs=1e-6)
