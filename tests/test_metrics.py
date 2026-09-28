"""Tests for the Market-1501 metrics (:mod:`reid.evaluation.metrics`).

Light tests: the metric code needs only ``numpy``, and a subprocess test below
checks that importing it does not pull in ``torch``. The expected CMC and mAP values below
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

import os
import subprocess
import sys
from pathlib import Path

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

    cmc, mean_ap = compute_cmc_map(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=3)
    # q0: after removing g0 (pid0/cam0) the kept order is g2, g1, g3 with
    # relevance [1, 0, 0], a rank-1 hit with AP 1.
    # q1: after removing g1 (pid1/cam0) the kept order is g0, g2, g3 with
    # relevance [0, 0, 1], a rank-3 hit with AP 1/3.
    np.testing.assert_allclose(cmc, [0.5, 0.5, 1.0], atol=1e-6)
    assert mean_ap == pytest.approx((1.0 + 1.0 / 3.0) / 2.0, abs=1e-6)


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


def test_cmc_forward_fills_when_exclusion_shortens_row() -> None:
    """A query whose kept gallery is shorter than ``max_rank`` is padded with its last value."""
    distmat = np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.4, 0.3]], dtype=np.float32)
    q_pids = np.array([0, 1])
    q_camids = np.array([0, 0])
    g_pids = np.array([0, 0, 0, 1])
    g_camids = np.array([0, 0, 1, 1])
    # q0 keeps only g2 (hit) and g3, so its CMC [1, 1] is forward-filled to length 4.
    # q1 keeps the whole gallery and first hits at rank 3 (AP 1/3).
    cmc, mean_ap = compute_cmc_map(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=4)
    np.testing.assert_allclose(cmc, [0.5, 0.5, 1.0, 1.0], atol=1e-6)
    assert mean_ap == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_forward_fill_single_query_after_exclusion() -> None:
    """After excluding the same-camera match, a rank-2 hit is forward-filled to rank 4."""
    distmat = np.array([[0.1, 0.3, 0.2, 0.4]], dtype=np.float32)
    cmc, mean_ap = compute_cmc_map(
        distmat,
        np.array([0]),
        np.array([0, 0, 1, 1]),
        np.array([0]),
        np.array([0, 1, 1, 2]),
        max_rank=4,
    )
    np.testing.assert_allclose(cmc, [0.0, 1.0, 1.0, 1.0], atol=1e-6)
    assert mean_ap == pytest.approx(0.5, abs=1e-6)


def test_unmatched_query_is_skipped() -> None:
    """A query without any valid match is skipped, leaving CMC and mAP unchanged."""
    base = np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.4, 0.3]], dtype=np.float32)
    g_pids = np.array([0, 0, 0, 1])
    g_camids = np.array([0, 0, 1, 1])
    cmc, mean_ap = compute_cmc_map(
        base, np.array([0, 1]), g_pids, np.array([0, 0]), g_camids, max_rank=4
    )
    extended = np.vstack([base, [[0.1, 0.2, 0.3, 0.4]]]).astype(np.float32)
    cmc_ext, mean_ap_ext = compute_cmc_map(
        extended, np.array([0, 1, 9]), g_pids, np.array([0, 0, 0]), g_camids, max_rank=4
    )
    np.testing.assert_allclose(cmc_ext, cmc, atol=1e-6)
    assert mean_ap_ext == pytest.approx(mean_ap, abs=1e-6)
    aps = compute_ap_per_query(extended, np.array([0, 1, 9]), g_pids, np.array([0, 0, 0]), g_camids)
    assert aps.shape == (2,)


def test_distractors_count_as_negatives() -> None:
    """Distractor gallery items (pid 0) ranked first count as misses."""
    distmat = np.array([[0.1, 0.2]], dtype=np.float32)
    cmc, mean_ap = compute_cmc_map(
        distmat, np.array([5]), np.array([0, 5]), np.array([1]), np.array([2, 2]), max_rank=2
    )
    np.testing.assert_allclose(cmc, [0.0, 1.0], atol=1e-6)
    assert mean_ap == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize("max_rank", [0, -3])
def test_invalid_max_rank_raises(max_rank: int) -> None:
    """A non-positive ``max_rank`` raises ``ValueError``."""
    distmat = np.array([[0.1, 0.2]], dtype=np.float32)
    with pytest.raises(ValueError):
        compute_cmc_map(
            distmat, np.array([0]), np.array([0, 1]), np.array([0]), np.array([1, 1]), max_rank
        )


def test_average_precision_matches_loop_definition() -> None:
    """The vectorized AP equals the textbook mean of precision at each hit."""
    rng = np.random.default_rng(0)
    distmat = rng.random((5, 30)).astype(np.float32)
    q_pids = np.arange(5)
    g_pids = rng.integers(0, 5, size=30)
    q_camids = np.zeros(5, dtype=np.int64)
    g_camids = np.ones(30, dtype=np.int64)
    aps = compute_ap_per_query(distmat, q_pids, g_pids, q_camids, g_camids)
    expected = []
    for q in range(5):
        rel = (g_pids[np.argsort(distmat[q])] == q_pids[q]).astype(float)
        if rel.any():
            precision = np.cumsum(rel) / np.arange(1, rel.size + 1)
            expected.append(float((precision * rel).sum() / rel.sum()))
    np.testing.assert_allclose(aps, expected, atol=1e-6)


def _run_isolated(code: str) -> str:
    """Run ``code`` in a fresh interpreter with ``src`` on the path and return stdout."""
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(repo / "src")}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    )
    return proc.stdout.strip()


def test_metrics_import_does_not_import_torch() -> None:
    """``import reid.evaluation.metrics`` needs only NumPy, thanks to the lazy package."""
    out = _run_isolated("import sys, reid.evaluation.metrics; print('torch' in sys.modules)")
    assert out == "False"


def test_reid_import_avoids_optional_dependencies() -> None:
    """Importing ``reid`` never loads torchvision or the optional extras."""
    code = (
        "import sys, reid; "
        "print([m for m in ('torchvision', 'cv2', 'gradio', 'kagglehub') if m in sys.modules])"
    )
    assert _run_isolated(code) == "[]"
