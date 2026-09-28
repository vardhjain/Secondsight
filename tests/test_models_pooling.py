"""Tests for the global pooling layers (:mod:`reid.models.pooling`).

Only ``torch`` is required, so these run without torchvision.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 - conventional PyTorch alias

from reid.models.pooling import GeMPooling, GlobalAvgPool, GlobalMaxPool, build_pooling


def _feature_map(seed: int = 0) -> torch.Tensor:
    """Return a small strictly positive feature map.

    Args:
        seed: RNG seed.

    Returns:
        A ``(2, 8, 4, 4)`` tensor with values in ``[0.1, 1.1)``.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(2, 8, 4, 4, generator=generator) + 0.1


def test_gem_matches_formula() -> None:
    """GeM equals ``mean(x ** p) ** (1 / p)`` per channel."""
    x = _feature_map()
    expected = x.pow(3.0).mean(dim=(2, 3)).pow(1.0 / 3.0)
    assert torch.allclose(GeMPooling(p=3.0)(x), expected, atol=1e-6)


def test_gem_with_p_one_is_average_pooling() -> None:
    """With ``p=1`` GeM reduces to global average pooling."""
    x = _feature_map(1)
    assert torch.allclose(GeMPooling(p=1.0)(x), GlobalAvgPool()(x), atol=1e-6)


def test_gem_of_constant_map_is_the_constant() -> None:
    """Any power mean of a constant map is that constant."""
    x = torch.full((1, 2, 3, 3), 0.7)
    assert torch.allclose(GeMPooling()(x), torch.full((1, 2), 0.7), atol=1e-5)


def test_gem_clamps_non_positive_inputs() -> None:
    """Zero and negative activations are clamped to ``eps`` and stay finite."""
    x = torch.zeros(1, 2, 2, 2) - 1.0
    out = GeMPooling(eps=1e-6)(x)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, torch.full((1, 2), 1e-6), rtol=1e-3)


def test_gem_exponent_is_learnable() -> None:
    """The exponent ``p`` is a parameter that receives a gradient."""
    pool = GeMPooling()
    assert isinstance(pool.p, torch.nn.Parameter)
    pool(_feature_map()).sum().backward()
    assert pool.p.grad is not None
    assert torch.isfinite(pool.p.grad).all()


def test_max_pooling_matches_amax() -> None:
    """Global max pooling equals the spatial maximum."""
    x = _feature_map(2)
    assert torch.equal(GlobalMaxPool()(x), x.amax(dim=(2, 3)))


def test_avg_pooling_matches_adaptive_avg_pool() -> None:
    """Global average pooling equals the spatial mean."""
    x = _feature_map(3)
    expected = F.adaptive_avg_pool2d(x, 1).flatten(1)
    assert torch.allclose(GlobalAvgPool()(x), expected)


@pytest.mark.parametrize(
    ("name", "cls"),
    [("avg", GlobalAvgPool), (" MAX ", GlobalMaxPool), ("GeM", GeMPooling)],
)
def test_build_pooling_normalizes_names(name: str, cls: type) -> None:
    """Names are matched case-insensitively after stripping whitespace."""
    assert isinstance(build_pooling(name), cls)


def test_build_pooling_unknown_name_raises() -> None:
    """An unknown pooling name raises ``ValueError``."""
    with pytest.raises(ValueError, match="median"):
        build_pooling("median")
