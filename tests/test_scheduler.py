"""Tests for the warmup learning-rate schedulers (:mod:`reid.engine.scheduler`).

These are pure, CPU-only, deterministic checks needing only ``torch`` (no GPU,
dataset, or model). A single dummy parameter is wrapped in an SGD optimizer so
the schedulers are exercised through the public PyTorch API
(``optimizer.step()`` / ``scheduler.step()`` / ``scheduler.get_last_lr()``).
"""

from __future__ import annotations

import pytest
import torch

from reid.config import Config
from reid.engine.scheduler import (
    WarmupCosineLR,
    WarmupMultiStepLR,
    build_scheduler,
)

BASE_LR = 0.1


def _optimizer() -> torch.optim.Optimizer:
    """An SGD optimizer over a single dummy parameter at ``BASE_LR``."""
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([param], lr=BASE_LR)


def _collect_lrs(scheduler: object, num_epochs: int) -> list[float]:
    """Collect ``get_last_lr()[0]`` for epochs ``0 .. num_epochs - 1``."""
    lrs: list[float] = []
    for _ in range(num_epochs):
        scheduler.optimizer.step()  # silence the "step before optimizer" warning
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    return lrs


def test_multistep_linear_warmup_endpoints() -> None:
    """Linear warmup starts at ``base_lr * warmup_factor`` and reaches ``base_lr``."""
    sched = WarmupMultiStepLR(
        _optimizer(), milestones=[100], gamma=0.1, warmup_factor=0.01, warmup_epochs=5
    )
    lrs = _collect_lrs(sched, num_epochs=6)
    assert lrs[0] == pytest.approx(BASE_LR * 0.01)  # epoch 0
    assert lrs[5] == pytest.approx(BASE_LR)  # epoch == warmup_epochs


def test_multistep_decay_fires_at_milestones() -> None:
    """LR drops by ``gamma`` exactly at each milestone epoch (no warmup)."""
    sched = WarmupMultiStepLR(_optimizer(), milestones=[2, 4], gamma=0.1, warmup_epochs=0)
    lrs = _collect_lrs(sched, num_epochs=6)
    assert lrs[1] == pytest.approx(BASE_LR)  # before first milestone
    assert lrs[2] == pytest.approx(BASE_LR * 0.1)  # first decay
    assert lrs[3] == pytest.approx(BASE_LR * 0.1)
    assert lrs[4] == pytest.approx(BASE_LR * 0.01)  # second decay


def test_multistep_rejects_unknown_warmup_method() -> None:
    """An unknown warmup method is rejected at construction time."""
    with pytest.raises(ValueError, match="warmup method"):
        WarmupMultiStepLR(_optimizer(), milestones=[10], warmup_method="quadratic")


def test_cosine_warmup_then_anneals_to_eta_min() -> None:
    """Cosine schedule peaks at ``base_lr`` after warmup and decays to ``eta_min``."""
    sched = WarmupCosineLR(
        _optimizer(), max_epochs=10, warmup_epochs=2, warmup_factor=0.01, eta_min=0.0
    )
    lrs = _collect_lrs(sched, num_epochs=11)
    assert lrs[0] == pytest.approx(BASE_LR * 0.01)  # warmup start
    assert lrs[2] == pytest.approx(BASE_LR)  # peak at end of warmup
    assert lrs[10] == pytest.approx(0.0, abs=1e-6)  # annealed to eta_min


def test_cosine_is_monotonically_non_increasing_after_warmup() -> None:
    """After the warmup peak the cosine LR never increases."""
    sched = WarmupCosineLR(_optimizer(), max_epochs=12, warmup_epochs=3)
    lrs = _collect_lrs(sched, num_epochs=13)
    post_warmup = lrs[3:]
    assert all(a >= b - 1e-9 for a, b in zip(post_warmup, post_warmup[1:], strict=False))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("warmup_multistep", WarmupMultiStepLR),
        ("warmup_cosine", WarmupCosineLR),
        ("cosine", WarmupCosineLR),
    ],
)
def test_build_scheduler_dispatch(name: str, expected: type) -> None:
    """``build_scheduler`` returns the scheduler matching ``cfg.optim.scheduler``."""
    cfg = Config.from_dict({"optim": {"scheduler": name}})
    sched = build_scheduler(cfg, _optimizer())
    assert isinstance(sched, expected)


def test_build_scheduler_rejects_unknown() -> None:
    """An unrecognized scheduler name raises ``ValueError`` in the factory."""
    cfg = Config.from_dict({"optim": {"scheduler": "cosine"}})
    cfg.optim.scheduler = "does-not-exist"  # bypass load-time validation
    with pytest.raises(ValueError, match="scheduler"):
        build_scheduler(cfg, _optimizer())
