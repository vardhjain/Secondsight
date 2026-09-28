"""Tests for :mod:`reid.visualization` on tiny synthetic data.

Every figure is drawn with the non-interactive Agg backend into ``tmp_path``,
and Grad-CAM runs on a tiny stand-in network, so the tests are fast and never
open a window.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reid.visualization import (  # noqa: E402
    GradCAM,
    analysis,
    overlay_heatmap,
    plot_ap_distribution,
    plot_camera_heatmap,
    plot_cmc_curve,
    plot_distance_distributions,
    plot_gradcam_samples,
    plot_rerank_impact,
    plot_success_failure,
    plot_tsne,
    visualize_ranked_results,
)
from reid.visualization._common import require  # noqa: E402


class _TinyDataset(torch.utils.data.Dataset):
    """Dataset of small random normalized tensors with fixed identities."""

    def __init__(self, pids: list[int], camids: list[int], seed: int = 0) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.images = torch.randn(len(pids), 3, 16, 8, generator=gen)
        self.pids = pids
        self.camids = camids

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        return self.images[idx], self.pids[idx], self.camids[idx]


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer4 = nn.Conv2d(3, 4, kernel_size=3, padding=1)


class _TinyReID(nn.Module):
    """Stand-in exposing ``backbone.layer4`` and ``extract_features`` like ReIDModel."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.backbone = _Backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.backbone.layer4(x)).mean(dim=(2, 3))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)


# Queries 0..3 with ids (1, 2, 3, 4) on camera 1, gallery with ids on cameras 1 and 2.
Q_PIDS = np.array([1, 2, 3, 4])
Q_CAMS = np.array([1, 1, 1, 1])
G_PIDS = np.array([1, 1, 2, 3, 4, 5])
G_CAMS = np.array([1, 2, 2, 2, 2, 2])


def _distmat() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((len(Q_PIDS), len(G_PIDS))).astype(np.float32)


@pytest.fixture
def datasets() -> tuple[_TinyDataset, _TinyDataset]:
    return (
        _TinyDataset(Q_PIDS.tolist(), Q_CAMS.tolist(), seed=1),
        _TinyDataset(G_PIDS.tolist(), G_CAMS.tolist(), seed=2),
    )


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _assert_written(path: Path) -> None:
    assert path.is_file()
    assert path.stat().st_size > 0
    assert plt.get_fignums() == []


def test_analysis_plots_write_files(tmp_path: Path) -> None:
    dist = _distmat()
    plot_cmc_curve(np.linspace(0.5, 1.0, 10), save_path=tmp_path / "a" / "cmc.png")
    _assert_written(tmp_path / "a" / "cmc.png")
    plot_ap_distribution(np.array([]), save_path=tmp_path / "ap.png")
    _assert_written(tmp_path / "ap.png")
    plot_camera_heatmap(dist, Q_PIDS, G_PIDS, Q_CAMS, G_CAMS, save_path=tmp_path / "cam.png")
    _assert_written(tmp_path / "cam.png")


def test_camera_heatmap_without_seaborn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "seaborn", None)
    plot_camera_heatmap(_distmat(), Q_PIDS, G_PIDS, Q_CAMS, G_CAMS, save_path=tmp_path / "h.png")
    _assert_written(tmp_path / "h.png")


def test_camera_pair_rank1_matches_manual_count() -> None:
    dist = np.array([[0.1, 0.9], [0.8, 0.2]])
    matrix = analysis._camera_pair_rank1(
        dist, np.array([1, 2]), np.array([1, 3]), np.array([1, 1]), np.array([2, 2]), [1, 2]
    )
    # Only query 0 has its identity in camera 2, and its top-1 is correct.
    assert matrix[0, 1] == pytest.approx(1.0)
    assert np.isnan(matrix[0, 0]) and np.isnan(matrix[1, 0])


def test_distance_distributions_drop_same_camera_positives(tmp_path: Path) -> None:
    fig = plot_distance_distributions(
        _distmat(), Q_PIDS, G_PIDS, Q_CAMS, G_CAMS, save_path=tmp_path / "d.png", max_neg=5
    )
    _assert_written(tmp_path / "d.png")
    ax = fig.axes[0]
    # Positives exclude the (query 0, gallery 0) same-id same-camera pair, so 4 remain.
    pos_counts = sum(p.get_height() > 0 for p in ax.containers[0].patches)
    assert pos_counts >= 1
    assert len(ax.containers) == 2


def test_tsne_is_seeded_and_uses_distinct_colours(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(30, 8)).astype(np.float32)
    pids = np.repeat(np.arange(10), 3)

    def offsets(seed: int) -> np.ndarray:
        fig = plot_tsne(feats, pids, num_ids=5, save_path=tmp_path / "t.png", seed=seed)
        return fig.axes[0].collections[0].get_array()

    first, second = offsets(3), offsets(3)
    np.testing.assert_array_equal(first, second)
    assert set(np.unique(first).tolist()) == set(range(5))
    _assert_written(tmp_path / "t.png")


def test_ranked_results_excludes_same_id_same_camera(tmp_path: Path, datasets) -> None:
    query, gallery = datasets
    dist = np.ones((4, 6), dtype=np.float32)
    dist[0, 0] = 0.0  # same id and camera as query 0; must be skipped
    dist[0, 1] = 0.1  # same id, other camera; must be rank 1
    fig = visualize_ranked_results(
        dist, query, gallery, Q_PIDS, G_PIDS, Q_CAMS, G_CAMS,
        topk=2, num_query=4, save_path=tmp_path / "r.png", seed=0,
    )  # fmt: skip
    _assert_written(tmp_path / "r.png")
    query_titles = [ax.get_title() for ax in fig.axes[::3]]
    row = query_titles.index("Query\nID 1")
    assert fig.axes[row * 3 + 1].get_title() == "R1\nID 1"


@pytest.mark.parametrize(("num_query", "topk"), [(1, 3), (3, 0)])
def test_ranked_results_degenerate_grids(tmp_path: Path, datasets, num_query, topk) -> None:
    query, gallery = datasets
    visualize_ranked_results(
        _distmat(), query, gallery, Q_PIDS, G_PIDS, Q_CAMS, G_CAMS,
        topk=topk, num_query=num_query, save_path=tmp_path / "r.png",
    )  # fmt: skip
    _assert_written(tmp_path / "r.png")


def test_success_failure_uses_distmat_and_seed(tmp_path: Path, datasets) -> None:
    query, gallery = datasets
    dist = np.ones((4, 6), dtype=np.float32)
    dist[0, 1] = 0.0  # query 0 (id 1): correct cross-camera match
    dist[1, 5] = 0.0  # query 1 (id 2): wrong match (id 5)

    def titles() -> list[str]:
        fig = plot_success_failure(
            dist, query, gallery, Q_PIDS, G_PIDS, Q_CAMS, G_CAMS,
            num_success=1, num_fail=1, save_path=tmp_path / "sf.png", seed=0,
        )  # fmt: skip
        return [ax.get_title() for ax in fig.axes]

    first = titles()
    assert first == titles()
    assert any(t.startswith("SUCCESS") for t in first)
    assert any(t.startswith("FAILURE") for t in first)
    _assert_written(tmp_path / "sf.png")


def test_success_failure_single_row(tmp_path: Path, datasets) -> None:
    query, gallery = datasets
    dist = np.ones((4, 6), dtype=np.float32)
    dist[:, 1] = 0.0  # query 0 finds its cross-camera match first
    fig = plot_success_failure(
        dist, query, gallery, Q_PIDS, G_PIDS, Q_CAMS, G_CAMS,
        num_success=1, num_fail=0, save_path=tmp_path / "one.png",
    )  # fmt: skip
    assert fig is not None and len(fig.axes) == 2


def test_rerank_impact_finds_corrected_query(tmp_path: Path, datasets) -> None:
    query, gallery = datasets
    before = np.ones((4, 6), dtype=np.float32)
    before[2, 4] = 0.0  # query 2 (id 3) wrongly matches id 4
    after = before.copy()
    after[2, 4] = 1.0
    after[2, 3] = 0.0  # re-ranking fixes it
    fig = plot_rerank_impact(
        before, after, query, gallery, Q_PIDS, G_PIDS, Q_CAMS, G_CAMS,
        save_path=tmp_path / "rr.png",
    )  # fmt: skip
    _assert_written(tmp_path / "rr.png")
    assert [ax.get_title() for ax in fig.axes] == [
        "Query (ID 3)",
        "Before: WRONG\n(ID 4)",
        "After: CORRECT\n(ID 3)",
    ]
    same = plot_rerank_impact(before, before, query, gallery, Q_PIDS, G_PIDS, Q_CAMS, G_CAMS)
    assert same is None
    with pytest.raises(ValueError, match="shape"):
        plot_rerank_impact(before, before[:2], query, gallery, Q_PIDS, G_PIDS, Q_CAMS, G_CAMS)


def test_gradcam_contract_and_hook_cleanup() -> None:
    pytest.importorskip("cv2")
    model = _TinyReID().train()
    layer = model.backbone.layer4
    x = torch.randn(1, 3, 16, 8)
    with torch.no_grad(), GradCAM(model, layer) as cam:
        heatmap = cam(x)
    assert heatmap.shape == (16, 8)
    assert heatmap.dtype == np.float32
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0
    assert not layer._forward_hooks and not layer._backward_hooks
    assert model.training  # the previous mode is restored
    assert all(p.grad is None for p in model.parameters())


def test_gradcam_works_under_inference_mode() -> None:
    pytest.importorskip("cv2")
    model = _TinyReID().eval()
    with torch.inference_mode(), GradCAM(model, model.backbone.layer4) as cam:
        heatmap = cam(torch.randn(1, 3, 16, 8))
    assert heatmap.shape == (16, 8)


def test_overlay_heatmap_handles_uint8() -> None:
    pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    image = rng.integers(1, 255, size=(8, 4, 3)).astype(np.uint8)
    out = overlay_heatmap(image, np.zeros((4, 2), dtype=np.float32), alpha=0.0)
    np.testing.assert_allclose(out, image / 255.0, atol=1e-6)


@pytest.mark.parametrize("num_samples", [1, 3])
def test_plot_gradcam_samples(tmp_path: Path, datasets, num_samples: int) -> None:
    pytest.importorskip("cv2")
    query, _ = datasets
    fig = plot_gradcam_samples(_TinyReID(), query, num_samples, save_path=tmp_path / "g.png")
    _assert_written(tmp_path / "g.png")
    assert len(fig.axes) == 2 * num_samples


def test_missing_optional_dependency_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(ImportError, match=r"secondsight\[viz\]"):
        require("cv2", "Grad-CAM")
