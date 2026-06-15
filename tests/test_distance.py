"""Tests for the distance utilities (:mod:`reid.utils.distance`).

Light tests: only ``torch`` is required. They check the Euclidean distance
against :func:`torch.cdist`, validate the cosine-distance semantics, exercise the
metric dispatcher, and confirm numerical safety (no ``NaN`` from the clamped
square root).
"""

from __future__ import annotations

import pytest
import torch

from reid.utils.distance import compute_distance_matrix, cosine_distance, euclidean_distance


def test_euclidean_matches_torch_cdist() -> None:
    """Euclidean distance equals :func:`torch.cdist` to float32 tolerance."""
    torch.manual_seed(0)
    x = torch.randn(5, 8)
    y = torch.randn(7, 8)
    got = euclidean_distance(x, y)
    ref = torch.cdist(x, y)
    assert got.shape == (5, 7)
    assert torch.allclose(got, ref, atol=1e-5)


def test_euclidean_self_diagonal_is_zero() -> None:
    """The distance of every point to itself is (numerically) zero."""
    torch.manual_seed(1)
    x = torch.randn(6, 4)
    d = euclidean_distance(x, x)
    # The clamp floor (sqrt(1e-12) == 1e-6) plus the expansion-form rounding for
    # large-norm rows means "zero" is only approximate; a loose tolerance is
    # appropriate here.
    assert torch.allclose(d.diagonal(), torch.zeros(6), atol=1e-3)


def test_euclidean_is_nonnegative_and_finite() -> None:
    """Euclidean distances are finite and non-negative (clamp before sqrt)."""
    torch.manual_seed(2)
    x = torch.randn(4, 3)
    y = torch.randn(9, 3)
    d = euclidean_distance(x, y)
    assert torch.isfinite(d).all()
    assert (d >= 0).all()


def test_cosine_distance_self_is_zero() -> None:
    """Cosine distance of identical vectors is zero."""
    torch.manual_seed(3)
    x = torch.randn(5, 10)
    d = cosine_distance(x, x)
    assert torch.allclose(d.diagonal(), torch.zeros(5), atol=1e-6)


def test_cosine_distance_range() -> None:
    """Cosine distance lies in ``[0, 2]``."""
    torch.manual_seed(4)
    x = torch.randn(6, 7)
    y = torch.randn(8, 7)
    d = cosine_distance(x, y)
    assert d.shape == (6, 8)  # (num_x, num_y); 7 is the shared feature dim.
    assert float(d.min()) >= -1e-6
    assert float(d.max()) <= 2.0 + 1e-6


def test_cosine_opposite_vectors_is_two() -> None:
    """Antiparallel vectors have cosine distance 2."""
    x = torch.tensor([[1.0, 0.0]])
    y = torch.tensor([[-1.0, 0.0]])
    d = cosine_distance(x, y)
    assert torch.allclose(d, torch.tensor([[2.0]]), atol=1e-6)


def test_compute_distance_matrix_dispatch() -> None:
    """The dispatcher routes to the requested metric (case-insensitive)."""
    torch.manual_seed(5)
    x = torch.randn(4, 5)
    y = torch.randn(6, 5)
    assert torch.allclose(compute_distance_matrix(x, y, "euclidean"), euclidean_distance(x, y))
    assert torch.allclose(compute_distance_matrix(x, y, "COSINE"), cosine_distance(x, y))


def test_compute_distance_matrix_invalid_metric_raises() -> None:
    """An unsupported metric name raises :class:`ValueError`."""
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    with pytest.raises(ValueError):
        compute_distance_matrix(x, y, "manhattan")
