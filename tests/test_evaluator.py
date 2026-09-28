"""Tests for feature extraction and the evaluation driver (:mod:`reid.evaluation.evaluator`).

Light tests with tiny stub models and synthetic in-memory loaders, so no dataset,
pretrained weights or network access is needed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, TensorDataset

from reid.config import Config
from reid.evaluation.evaluator import Evaluator, extract_features
from reid.evaluation.metrics import compute_cmc_map

_RERANK_KEYS = {"rerank_mAP", "rerank_rank1", "rerank_rank5", "rerank_rank10", "rerank_cmc"}


class ColMean(nn.Module):
    """Stub model returning the per-column mean, so a width flip changes the output."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, x: Tensor) -> Tensor:
        self.calls += 1
        return self.w * x.mean(dim=(1, 2))


def _loader(images: Tensor, pids: list[int], camids: list[int], batch_size: int = 2) -> DataLoader:
    """Wrap tensors in a deterministic ``(images, pids, camids)`` loader."""
    dataset = TensorDataset(images, torch.tensor(pids), torch.tensor(camids))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _point_loaders() -> tuple[DataLoader, DataLoader, Tensor, Tensor]:
    """Query and gallery loaders whose images are 3-D points of shape ``(3, 1, 1)``."""
    qf = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    gf = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    q_loader = _loader(qf.view(-1, 3, 1, 1), [1, 2], [1, 1])
    g_loader = _loader(gf.view(-1, 3, 1, 1), [1, 2, 3, 1], [2, 2, 2, 1])
    return q_loader, g_loader, qf, gf


def _config(rerank: bool = False) -> Config:
    cfg = Config()
    cfg.eval.rerank = rerank
    cfg.eval.flip_tta = False
    cfg.eval.feat_norm = True
    cfg.eval.rerank_k1 = 3
    cfg.eval.rerank_k2 = 2
    return cfg


def test_extract_features_flip_tta_averages_before_normalizing() -> None:
    """Flip TTA averages the original and width-flipped features, then L2-normalizes."""
    torch.manual_seed(0)
    images = torch.rand(5, 3, 4, 6)
    model = ColMean()
    feats, pids, camids = extract_features(
        model, _loader(images, [3, 1, 4, 1, 5], [0, 1, 0, 1, 0]), "cpu", flip_tta=True
    )
    with torch.no_grad():
        expected = normalize((model(images) + model(images.flip(3))) / 2, dim=1)
    torch.testing.assert_close(feats, expected)
    torch.testing.assert_close(feats.norm(dim=1), torch.ones(5))
    assert pids.dtype == np.int64 and camids.dtype == np.int64
    assert pids.tolist() == [3, 1, 4, 1, 5]
    assert camids.tolist() == [0, 1, 0, 1, 0]


def test_extract_features_without_norm_keeps_raw_features() -> None:
    """With ``feat_norm=False`` the raw model output is returned unchanged."""
    torch.manual_seed(1)
    images = torch.rand(3, 3, 4, 6)
    model = ColMean()
    feats, _, _ = extract_features(
        model, _loader(images, [0, 1, 2], [0, 0, 0]), "cpu", False, False
    )
    with torch.no_grad():
        torch.testing.assert_close(feats, model(images))


def test_extract_features_sets_eval_mode() -> None:
    """The model is switched to eval mode before extraction."""
    model = ColMean().train()
    extract_features(model, _loader(torch.rand(2, 3, 2, 2), [0, 1], [0, 0]), "cpu")
    assert not model.training


def test_extract_features_rejects_empty_loader() -> None:
    """An empty loader raises a clear ``ValueError``."""
    empty = _loader(torch.rand(0, 3, 2, 2), [], [])
    with pytest.raises(ValueError, match="no batches"):
        extract_features(ColMean(), empty, "cpu")


class _ExtractorOnly(nn.Module):
    """Model whose ``forward`` must not be used because ``extract_features`` exists."""

    def forward(self, x: Tensor) -> Tensor:
        raise AssertionError("forward should not be called")

    def extract_features(self, x: Tensor) -> Tensor:
        return x.flatten(1) * 2


class _TupleOutput(nn.Module):
    """Model returning a tuple whose last element is the feature."""

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return torch.zeros(x.size(0), 1), x.flatten(1)


def test_forward_prefers_extract_features_and_last_tuple_element() -> None:
    """``extract_features`` is preferred, and tuple outputs use their last element."""
    images = torch.rand(2, 3, 1, 1)
    loader = _loader(images, [0, 1], [0, 0])
    feats, _, _ = extract_features(_ExtractorOnly(), loader, "cpu", feat_norm=False)
    torch.testing.assert_close(feats, images.flatten(1) * 2)
    feats, _, _ = extract_features(_TupleOutput(), loader, "cpu", feat_norm=False)
    torch.testing.assert_close(feats, images.flatten(1))


def test_evaluator_end_to_end_exact() -> None:
    """A perfectly separable toy split scores 1.0 and matches ``compute_cmc_map``."""
    q_loader, g_loader, qf, gf = _point_loaders()
    evaluator = Evaluator(nn.Flatten(), q_loader, g_loader, "cpu", _config(rerank=True))
    results = evaluator.evaluate()

    for key in ("mAP", "rank1", "rank5", "rank10"):
        assert results[key] == pytest.approx(1.0)
    cmc = results["cmc"]
    assert isinstance(cmc, np.ndarray) and cmc.shape == (4,)
    assert set(results) >= _RERANK_KEYS
    assert np.isfinite(float(results["rerank_mAP"]))  # type: ignore[arg-type]

    distmat = torch.cdist(normalize(qf, dim=1), normalize(gf, dim=1)).numpy()
    _, expected_map = compute_cmc_map(
        distmat, np.array([1, 2]), np.array([1, 2, 3, 1]), np.array([1, 1]), np.array([2, 2, 2, 1])
    )
    assert results["mAP"] == pytest.approx(expected_map, abs=1e-6)

    no_rerank = evaluator.evaluate(rerank=False)
    assert not _RERANK_KEYS & set(no_rerank)


def test_evaluator_small_gallery_rank_fallback() -> None:
    """With a gallery smaller than 10, Rank-5 and Rank-10 equal the final CMC value."""
    q_loader, g_loader, _, _ = _point_loaders()
    results = Evaluator(nn.Flatten(), q_loader, g_loader, "cpu", _config()).evaluate()
    cmc = results["cmc"]
    assert isinstance(cmc, np.ndarray)
    assert results["rank5"] == pytest.approx(float(cmc[-1]))
    assert results["rank10"] == pytest.approx(float(cmc[-1]))


def test_evaluator_rejects_invalid_max_rank() -> None:
    """A non-positive ``cfg.eval.max_rank`` raises ``ValueError``."""
    q_loader, g_loader, _, _ = _point_loaders()
    cfg = _config()
    cfg.eval.max_rank = 0
    with pytest.raises(ValueError):
        Evaluator(nn.Flatten(), q_loader, g_loader, "cpu", cfg).evaluate()


def test_evaluate_caches_features_until_reset() -> None:
    """Features are extracted once and re-extracted only after ``reset_cache``."""
    torch.manual_seed(2)
    q_images = torch.rand(2, 3, 4, 6)
    g_images = torch.rand(4, 3, 4, 6)
    model = ColMean()
    evaluator = Evaluator(
        model,
        _loader(q_images, [0, 1], [0, 0]),
        _loader(g_images, [0, 1, 0, 1], [1, 1, 1, 1]),
        "cpu",
        _config(),
    )
    evaluator.evaluate()
    calls = model.calls
    first_qf = evaluator._qf
    assert first_qf is not None

    evaluator.evaluate()
    assert model.calls == calls

    with torch.no_grad():
        model.w.fill_(-1.0)  # flips every feature, which normalization preserves.
    evaluator.evaluate()
    assert model.calls == calls
    assert evaluator._qf is first_qf

    evaluator.reset_cache()
    evaluator.evaluate()
    assert model.calls > calls
    assert evaluator._qf is not None
    torch.testing.assert_close(evaluator._qf, -first_qf)
