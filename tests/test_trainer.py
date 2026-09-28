"""Smoke and regression tests for :class:`reid.engine.trainer.Trainer`.

Everything runs on CPU in well under a second per test. A tiny linear stand-in
model replaces ResNet-50, and the data are small synthetic tensors, so the
tests exercise the real training loop (loss, center-loss optimizer, scheduler,
evaluation cadence, checkpointing and ``history.json``) without any dataset or
download.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from reid.config import Config
from reid.engine.scheduler import build_scheduler
from reid.engine.trainer import Trainer
from reid.evaluation.evaluator import Evaluator
from reid.losses.build import build_loss
from reid.utils.checkpoint import load_checkpoint
from reid.utils.meters import AverageMeter

NUM_IDS = 4
FEAT_DIM = 8
IMAGE_SHAPE = (3, 4, 2)  # (C, H, W), 24 input values per image.
LOGGER = logging.getLogger("test.trainer")


class TinyReIDModel(nn.Module):
    """Minimal stand-in for ``ReIDModel`` with the same train/eval contract."""

    def __init__(self, num_classes: int = NUM_IDS, feat_dim: int = FEAT_DIM) -> None:
        super().__init__()
        in_dim = IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * IMAGE_SHAPE[2]
        self.feat_dim = feat_dim
        self.embed = nn.Linear(in_dim, feat_dim, bias=False)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Return ``(logits, feat)`` in train mode and ``feat`` in eval mode."""
        feat = self.embed(x.flatten(1))
        if self.training:
            return self.classifier(feat), feat
        return feat


class StubEvaluator:
    """Evaluator double that records calls and returns scripted metrics."""

    def __init__(self, maps: list[float]) -> None:
        self.maps = list(maps)
        self.calls: list[str] = []

    def reset_cache(self) -> None:
        """Record a cache reset."""
        self.calls.append("reset")

    def evaluate(self, rerank: bool | None = None) -> dict[str, object]:
        """Record the call and return the next scripted mAP."""
        self.calls.append(f"eval({rerank})")
        value = self.maps.pop(0)
        return {"mAP": value, "rank1": value + 0.1, "rank5": value + 0.2}


def _cfg(tmp_path: Path, **train: Any) -> Config:
    """Build a tiny, valid CPU config writing into ``tmp_path``."""
    return Config.from_dict(
        {
            "data": {"batch_size": 8, "num_instances": 4},
            "loss": {"center_loss": True},
            "optim": {"lr": 0.01, "milestones": [1], "warmup_epochs": 0},
            "eval": {"feat_norm": False, "flip_tta": False},
            "train": {
                "device": "cpu",
                "amp": True,  # Ignored on CPU; exercises the disabled-AMP path.
                "log_period": 1,
                "output_dir": str(tmp_path),
                **train,
            },
        }
    )


def _train_batches(num_batches: int = 3) -> list[tuple[Tensor, Tensor, Tensor]]:
    """Make PK-style batches of two identities with four images each."""
    generator = torch.Generator().manual_seed(0)
    batches = []
    for i in range(num_batches):
        ids = torch.tensor([(2 * i) % NUM_IDS, (2 * i + 1) % NUM_IDS])
        labels = ids.repeat_interleave(4)
        images = torch.randn(8, *IMAGE_SHAPE, generator=generator)
        batches.append((images, labels, torch.zeros(8, dtype=torch.long)))
    return batches


def _make_trainer(cfg: Config, evaluator: Any = None) -> Trainer:
    """Wire a trainer around the tiny model and the real loss and scheduler."""
    model = TinyReIDModel()
    loss_fn = build_loss(cfg, num_classes=NUM_IDS, feat_dim=FEAT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr)
    scheduler = build_scheduler(cfg, optimizer)
    return Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        train_loader=_train_batches(),  # type: ignore[arg-type]
        evaluator=evaluator,
        logger=LOGGER,
    )


def test_training_loop_cadence_checkpoints_and_history(tmp_path: Path) -> None:
    """Periodic plus forced final evals, best/last checkpoints and history.json."""
    cfg = _cfg(tmp_path, max_epochs=3, eval_period=2)
    evaluator = StubEvaluator(maps=[0.3, 0.2])
    trainer = _make_trainer(cfg, evaluator)
    assert trainer.center_optimizer is not None
    assert trainer.loss_fn.center_loss is not None
    centers_before = trainer.loss_fn.center_loss.centers.detach().clone()

    summary = trainer.train()

    # Epoch 2 is periodic and epoch 3 is the forced final evaluation, and every
    # evaluation first drops the feature cache.
    assert evaluator.calls == ["reset", "eval(False)", "reset", "eval(False)"]
    assert summary["history"]["mAP"] == [None, 0.3, 0.2]
    assert summary["best_epoch"] == 2
    assert summary["best_mAP"] == pytest.approx(0.3)
    assert summary["best_rank1"] == pytest.approx(0.4)

    # The LR is recorded as used during each epoch: no warmup, decay at 1.
    lr = cfg.optim.lr
    assert summary["history"]["lr"] == pytest.approx([lr, 0.1 * lr, 0.1 * lr])
    assert all(loss > 0 for loss in summary["history"]["loss"])

    # The center-loss optimizer moved the centers.
    assert not torch.equal(trainer.loss_fn.center_loss.centers.detach(), centers_before)

    best = load_checkpoint(tmp_path / "best.pth")
    last = load_checkpoint(tmp_path / "last.pth")
    assert set(best) == {"epoch", "model", "best_mAP", "config"}
    assert best["epoch"] == 2
    assert last["epoch"] == 3
    assert last["config"] == cfg.to_portable_dict()
    assert last["config"]["data"]["root"] is None
    assert not Path(last["config"]["train"]["output_dir"]).is_absolute()

    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert history["best_epoch"] == 2
    assert history["history"]["epoch"] == [1, 2, 3]


def test_short_run_still_writes_best_checkpoint(tmp_path: Path) -> None:
    """With max_epochs below eval_period the final epoch is still evaluated."""
    cfg = _cfg(tmp_path, max_epochs=1, eval_period=10)
    evaluator = StubEvaluator(maps=[0.0])
    summary = _make_trainer(cfg, evaluator).train()

    assert evaluator.calls == ["reset", "eval(False)"]
    assert summary["best_epoch"] == 1
    # Even a zero mAP counts as the first (and so best) evaluation.
    assert load_checkpoint(tmp_path / "best.pth")["epoch"] == 1


def test_without_evaluator_only_last_checkpoint_is_written(tmp_path: Path) -> None:
    """No evaluator means no best.pth, but last.pth and history.json exist."""
    cfg = _cfg(tmp_path, max_epochs=1, eval_period=1)
    summary = _make_trainer(cfg).train()

    assert summary["best_epoch"] == -1
    assert (tmp_path / "last.pth").is_file()
    assert (tmp_path / "history.json").is_file()
    assert not (tmp_path / "best.pth").exists()


def test_center_gradients_are_unscaled_before_the_center_step(tmp_path: Path) -> None:
    """The center SGD step uses the raw center-loss gradient, not the weighted one."""
    cfg = _cfg(tmp_path, max_epochs=1)
    trainer = _make_trainer(cfg)
    center_loss = trainer.loss_fn.center_loss
    assert center_loss is not None
    centers = center_loss.centers

    feats = torch.randn(8, FEAT_DIM)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    (raw_grad,) = torch.autograd.grad(center_loss(feats, labels), centers)
    before = centers.detach().clone()

    (trainer.center_weight * center_loss(feats, labels)).backward()
    trainer._step_center_optimizer()

    expected = before - cfg.loss.center_lr * raw_grad
    torch.testing.assert_close(centers.detach(), expected)


def test_periodic_evaluation_sees_updated_weights(tmp_path: Path) -> None:
    """Regression test for the frozen feature cache: mAP follows the weights.

    One query (id 1, camera 1) and two gallery images (ids 1 and 2, camera 2)
    are arranged so that projecting onto input dimension 0 ranks the true match
    first (mAP 1.0) while projecting onto dimension 1 ranks it second (mAP 0.5).
    """
    cfg = _cfg(tmp_path, max_epochs=1)
    trainer = _make_trainer(cfg)
    model = trainer.model

    def image(d0: float, d1: float) -> Tensor:
        x = torch.zeros(1, *IMAGE_SHAPE)
        x.view(1, -1)[0, 0] = d0
        x.view(1, -1)[0, 1] = d1
        return x

    query = [(image(0.0, 0.0), torch.tensor([1]), torch.tensor([1]))]
    gallery = [
        (
            torch.cat([image(1.0, 10.0), image(3.0, 0.0)]),
            torch.tensor([1, 2]),
            torch.tensor([2, 2]),
        )
    ]
    evaluator = Evaluator(model, query, gallery, device="cpu", cfg=cfg)  # type: ignore[arg-type]
    trainer.evaluator = evaluator  # type: ignore[assignment]

    def project_onto(dim: int) -> None:
        with torch.no_grad():
            model.embed.weight.zero_()
            model.embed.weight[0, dim] = 1.0

    project_onto(0)
    map_first, _ = trainer._run_evaluation(epoch=0)
    project_onto(1)
    map_second, _ = trainer._run_evaluation(epoch=1)

    assert map_first == pytest.approx(1.0)
    assert map_second == pytest.approx(0.5)


def test_average_meter() -> None:
    """``AverageMeter`` keeps a count-weighted running average and resets."""
    meter = AverageMeter()
    meter.update(1.0, n=2)
    meter.update(4.0)
    assert meter.avg == pytest.approx(2.0)
    assert meter.count == 3
    assert meter.val == pytest.approx(4.0)
    meter.reset()
    assert (meter.val, meter.avg, meter.sum, meter.count) == (0.0, 0.0, 0.0, 0)
