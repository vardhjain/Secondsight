"""Training engine for the Person Re-Identification strong baseline.

This module implements :class:`Trainer`, which orchestrates a full training run:

* a per-epoch loop with AMP mixed precision (CUDA-only, with a transparent CPU
  fallback),
* a combined identity + triplet (+ optional center) loss, always computed in
  float32 outside the autocast region because distance math is precision
  sensitive,
* an optional separate SGD optimizer for the center-loss parameters, using the
  standard "un-scale the center gradients by ``1 / center_weight``" trick so
  that the center term is optimized at its true magnitude,
* a warmup-aware learning-rate scheduler stepped once per epoch,
* periodic evaluation through an injected evaluator, every ``eval_period``
  epochs and always on the final epoch,
* checkpointing of the best-by-mAP model plus a rolling ``last.pth``, with
  machine-specific paths scrubbed from the embedded config, and
* a ``history.json`` record of the per-epoch loss, learning rate and metrics.

Heavy / optional dependencies (``tqdm``) are imported lazily so the module is
importable in minimal environments. Library logging is used throughout (never
``print``).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from reid.utils.checkpoint import save_checkpoint
from reid.utils.device import resolve_device
from reid.utils.meters import AverageMeter

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from torch.utils.data import DataLoader

    from reid.config import Config
    from reid.evaluation.evaluator import Evaluator
    from reid.losses.build import ReIDLoss


class Trainer:
    """Drive the training of a Re-ID model.

    Args:
        model: The :class:`reid.models.reid_model.ReIDModel` (or compatible
            module) to train. In ``train`` mode its forward returns
            ``(cls_score, global_feat)``.
        loss_fn: The combined :class:`reid.losses.build.ReIDLoss`. If it has
            center loss enabled, the trainer builds a dedicated optimizer for
            the center parameters.
        optimizer: The main optimizer over the model parameters.
        scheduler: The (epoch-stepped) learning-rate scheduler.
        cfg: The complete experiment configuration.
        train_loader: Iterable yielding ``(images, labels, camids)`` batches.
        evaluator: Optional evaluator run every ``cfg.train.eval_period``
            epochs and on the final epoch, so ``best.pth`` always exists when
            an evaluator is given. It is called with ``rerank=False``. The
            headline test-split results are the caller's responsibility (see
            :mod:`reid.cli.train`). Defaults to ``None``.
        logger: Optional logger. If ``None`` a module logger is used.

    Attributes:
        device: Resolved :class:`torch.device` (falls back to CPU if CUDA is
            requested but unavailable).
        use_amp: Whether AMP mixed precision is active (requires both
            ``cfg.train.amp`` and a CUDA device).
        best_map: Best mAP observed so far across periodic evaluations.
        history: Accumulated per-epoch metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: ReIDLoss,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        cfg: Config,
        train_loader: DataLoader,
        evaluator: Evaluator | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg = cfg
        self.logger = logger or logging.getLogger("reid.engine.trainer")

        # Resolve device with a CPU fallback so the trainer is runnable anywhere.
        self.device = resolve_device(cfg.train.device)

        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.evaluator = evaluator

        # AMP is CUDA-only; on CPU we silently run full precision.
        self.use_amp = bool(cfg.train.amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Optional dedicated optimizer for center-loss parameters.
        self.center_optimizer: Optimizer | None = None
        self.center_weight = float(cfg.loss.center_weight)
        if getattr(loss_fn, "use_center", False) and loss_fn.center_loss is not None:
            self.center_optimizer = torch.optim.SGD(
                loss_fn.center_loss.parameters(), lr=cfg.loss.center_lr
            )

        self.output_dir = Path(cfg.train.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_map: float = 0.0
        self.history: dict[str, list[Any]] = {
            "epoch": [],
            "loss": [],
            "lr": [],
            "mAP": [],
            "rank1": [],
        }

    def train_one_epoch(self, epoch: int) -> float:
        """Run a single training epoch.

        Args:
            epoch: Zero-based epoch index (used only for logging).

        Returns:
            The average total loss over the epoch.
        """
        self.model.train()
        loss_meter = AverageMeter()
        id_meter = AverageMeter()
        tri_meter = AverageMeter()
        center_meter = AverageMeter()

        progress = self._maybe_tqdm(self.train_loader, epoch)
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(progress, start=1):
            images, labels = batch[0], batch[1]
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            if self.center_optimizer is not None:
                self.center_optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                cls_score, global_feat = self.model(images)
            # The loss runs in float32 outside autocast. The triplet and center
            # distances expand ||a||^2 + ||b||^2 - 2ab, which cancels badly (and
            # can overflow) when the matmul is done in float16.
            loss, components = self.loss_fn(cls_score.float(), global_feat.float(), labels)

            # Backward + step through the grad scaler (a no-op scale of 1.0 when
            # AMP is disabled, so the same code path is CPU-safe).
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            if self.center_optimizer is not None:
                self._step_center_optimizer()

            self.scaler.update()

            batch_size = images.size(0)
            loss_meter.update(components["total"], batch_size)
            id_meter.update(components["id"], batch_size)
            tri_meter.update(components["triplet"], batch_size)
            center_meter.update(components["center"], batch_size)

            if batch_idx % self.cfg.train.log_period == 0:
                self._log_step(epoch, batch_idx, num_batches, loss_meter, id_meter, tri_meter)

        lr = self.optimizer.param_groups[0]["lr"]
        self.logger.info(
            "Epoch [%d/%d] done | loss=%.4f (id=%.4f, tri=%.4f, center=%.4f) | lr=%.2e",
            epoch + 1,
            self.cfg.train.max_epochs,
            loss_meter.avg,
            id_meter.avg,
            tri_meter.avg,
            center_meter.avg,
            lr,
        )
        return loss_meter.avg

    def _step_center_optimizer(self) -> None:
        """Apply the center-loss gradient correction and step its optimizer.

        The center loss is added to the total as ``center_weight * center``.
        To optimize the centers at their native scale we divide their gradients
        by ``center_weight`` (after un-scaling the AMP scale factor) before the
        SGD step. See Luo et al., 2019 ("Bag of Tricks").
        """
        assert self.center_optimizer is not None
        assert self.loss_fn.center_loss is not None

        # Un-scale AMP gradients on the center optimizer's params first so the
        # division below operates on true-magnitude gradients.
        self.scaler.unscale_(self.center_optimizer)
        if self.center_weight != 0.0:
            for param in self.loss_fn.center_loss.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(1.0 / self.center_weight)
        self.scaler.step(self.center_optimizer)

    def train(self) -> dict[str, Any]:
        """Run the full training loop with periodic evaluation and checkpointing.

        The evaluator runs every ``eval_period`` epochs and on the final epoch.
        The run summary is written to ``<output_dir>/history.json`` after every
        epoch, so a partial record survives an interrupted run.

        Returns:
            A dictionary with the per-epoch ``history``, the best metrics
            observed (``best_mAP``, the rank-1 at the best epoch, and the
            one-based ``best_epoch``, which is ``-1`` when no evaluation ran)
            and the ``elapsed_seconds`` of the run.
        """
        start = time.time()
        max_epochs = self.cfg.train.max_epochs
        self.logger.info(
            "Starting training: %d epochs on %s (AMP=%s)",
            max_epochs,
            self.device,
            self.use_amp,
        )

        best_rank1 = 0.0
        best_epoch = -1

        for epoch in range(max_epochs):
            # Capture the LR used DURING this epoch before stepping the scheduler,
            # so history["lr"] matches the per-step and end-of-epoch logs.
            lr_used = self.optimizer.param_groups[0]["lr"]
            avg_loss = self.train_one_epoch(epoch)
            self.scheduler.step()

            self.history["epoch"].append(epoch + 1)
            self.history["loss"].append(avg_loss)
            self.history["lr"].append(lr_used)

            map_value: float | None = None
            rank1_value: float | None = None

            # Always evaluate the final epoch as well, so short runs (max_epochs
            # below eval_period) and trailing epochs still compete for best.pth.
            is_last = epoch + 1 == max_epochs
            periodic = (epoch + 1) % self.cfg.train.eval_period == 0
            if self.evaluator is not None and (periodic or is_last):
                map_value, rank1_value = self._run_evaluation(epoch)

            self.history["mAP"].append(map_value)
            self.history["rank1"].append(rank1_value)

            # Track the best-by-mAP checkpoint. The first evaluation always
            # counts, so best.pth exists even when every mAP is exactly zero.
            is_best = False
            if map_value is not None and (best_epoch < 0 or map_value > self.best_map):
                is_best = True
                self.best_map = map_value
                best_rank1 = rank1_value if rank1_value is not None else 0.0
                best_epoch = epoch + 1

            self._save_checkpoints(epoch, is_best=is_best)
            self._write_history(best_rank1, best_epoch, time.time() - start)

        elapsed = time.time() - start
        self.logger.info(
            "Training complete in %.1fs | best mAP=%.4f (epoch %d)",
            elapsed,
            self.best_map,
            best_epoch,
        )
        return self._write_history(best_rank1, best_epoch, elapsed)

    def _write_history(self, best_rank1: float, best_epoch: int, elapsed: float) -> dict[str, Any]:
        """Build the run summary and write it atomically to ``history.json``.

        Args:
            best_rank1: Rank-1 accuracy at the best epoch so far.
            best_epoch: One-based best epoch so far, or ``-1`` before any eval.
            elapsed: Wall-clock seconds since training started.

        Returns:
            The summary dictionary that was written.
        """
        summary: dict[str, Any] = {
            "history": self.history,
            "best_mAP": self.best_map,
            "best_rank1": best_rank1,
            "best_epoch": best_epoch,
            "elapsed_seconds": elapsed,
        }
        path = self.output_dir / "history.json"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tmp.replace(path)
        return summary

    def _run_evaluation(self, epoch: int) -> tuple[float, float]:
        """Run the injected evaluator and log the headline metrics.

        Args:
            epoch: Zero-based epoch index (for logging).

        Returns:
            A ``(mAP, rank1)`` tuple of floats.
        """
        assert self.evaluator is not None
        self.model.eval()
        # Drop cached features so each periodic eval re-extracts from the CURRENT
        # weights. Without this the evaluator reuses the first eval's features
        # (Evaluator._ensure_features early-returns on the cache), so mAP stays
        # frozen and best-by-mAP locks best.pth to the earliest evaluated epoch.
        self.evaluator.reset_cache()
        metrics = self.evaluator.evaluate(rerank=False)
        self.model.train()

        map_value = _metric(metrics, "mAP")
        rank1_value = _metric(metrics, "rank1")
        self.logger.info(
            "Eval @ epoch %d | mAP=%.4f | Rank-1=%.4f | Rank-5=%.4f",
            epoch + 1,
            map_value,
            rank1_value,
            _metric(metrics, "rank5"),
        )
        return map_value, rank1_value

    def _save_checkpoints(self, epoch: int, is_best: bool) -> None:
        """Persist a rolling ``last.pth`` and, when improved, ``best.pth``.

        Args:
            epoch: Zero-based epoch index.
            is_best: Whether this epoch produced the best mAP so far.
        """
        # Persist only what a consumer can actually load: the model weights for
        # evaluation/inference, plus provenance metadata. Optimizer/scheduler
        # resume state is intentionally not saved (no resume path exists). The
        # portable config drops absolute local paths, because checkpoints may
        # be published (for example to the Hugging Face Space).
        state = {
            "epoch": epoch + 1,
            "model": self.model.state_dict(),
            "best_mAP": self.best_map,
            "config": self.cfg.to_portable_dict(),
        }

        save_checkpoint(state, self.output_dir / "last.pth")
        if is_best:
            save_checkpoint(state, self.output_dir / "best.pth")
            self.logger.info(
                "New best mAP=%.4f saved to %s",
                self.best_map,
                self.output_dir / "best.pth",
            )

    def _maybe_tqdm(self, loader: DataLoader, epoch: int) -> Iterable[Any]:
        """Wrap the loader in a tqdm bar if tqdm is installed, else pass through.

        Args:
            loader: The training data loader.
            epoch: Zero-based epoch index (for the bar description).

        Returns:
            Either a tqdm-wrapped iterator or the original loader.
        """
        try:
            from tqdm import tqdm
        except ImportError:
            return loader
        return tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{self.cfg.train.max_epochs}",
            leave=False,
        )

    def _log_step(
        self,
        epoch: int,
        batch_idx: int,
        num_batches: int,
        loss_meter: AverageMeter,
        id_meter: AverageMeter,
        tri_meter: AverageMeter,
    ) -> None:
        """Emit a periodic step log.

        Args:
            epoch: Zero-based epoch index.
            batch_idx: One-based index of the current batch in the epoch.
            num_batches: Total number of batches in the epoch.
            loss_meter: Running total-loss meter.
            id_meter: Running id-loss meter.
            tri_meter: Running triplet-loss meter.
        """
        lr = self.optimizer.param_groups[0]["lr"]
        self.logger.info(
            "Epoch [%d/%d] step [%d/%d] | loss=%.4f (id=%.4f, tri=%.4f) | lr=%.2e",
            epoch + 1,
            self.cfg.train.max_epochs,
            batch_idx,
            num_batches,
            loss_meter.avg,
            id_meter.avg,
            tri_meter.avg,
            lr,
        )


def _metric(metrics: dict[str, object], key: str) -> float:
    """Read one scalar metric from an evaluator result as a float.

    Args:
        metrics: The mapping returned by ``Evaluator.evaluate``.
        key: The metric name, for example ``"mAP"``.

    Returns:
        The metric as a Python float, or ``0.0`` when it is missing.
    """
    return float(cast(float, metrics.get(key, 0.0)))
