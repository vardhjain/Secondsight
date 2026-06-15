"""Typed configuration for the Re-ID pipeline.

The whole training / evaluation pipeline is driven by a single nested
:class:`Config` dataclass. Each logical area of the system has its own small
dataclass (data, model, loss, optimization, evaluation, training), and the
top-level :class:`Config` simply composes them.

Configs can be serialized to / from YAML and plain dictionaries. Construction
from external data is forgiving: unknown keys are ignored and missing keys fall
back to the dataclass defaults, so older or partially-specified YAML files keep
working as the schema evolves.

The default values defined here are the single source of truth and are mirrored
by ``configs/default.yaml``. The headline "strong baseline" recipe lives in
``configs/market1501_strong_baseline.yaml``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Dataset, augmentation and dataloader configuration.

    Attributes:
        root: Path to the Market-1501 data root (the directory that contains
            ``bounding_box_train``, ``query`` and ``bounding_box_test``). When
            ``None`` the path must be supplied via the CLI / build functions.
        height: Target image height in pixels.
        width: Target image width in pixels.
        batch_size: Number of images per training batch. With the identity
            sampler this equals ``P * K`` where ``P = batch_size //
            num_instances`` identities and ``K = num_instances`` images each.
        num_instances: Number of images sampled per identity (``K``).
        num_workers: Number of dataloader worker processes.
        pad: Padding (pixels) applied before the random crop during training.
        random_erasing: Whether to apply random erasing augmentation.
        re_prob: Probability of applying random erasing when enabled.
    """

    root: str | None = None
    height: int = 256
    width: int = 128
    batch_size: int = 64
    num_instances: int = 4
    num_workers: int = 4
    pad: int = 10
    random_erasing: bool = True
    re_prob: float = 0.5


@dataclass
class ModelConfig:
    """Backbone and Re-ID head configuration.

    Attributes:
        name: Backbone architecture name (e.g. ``"resnet50"``).
        pretrained: Whether to initialize the backbone from ImageNet weights.
        last_stride: Stride of the final ResNet stage (``layer4``). Setting this
            to ``1`` increases the spatial resolution of the feature map and is
            a key part of the strong baseline recipe.
        pooling: Global pooling type, one of ``{"avg", "gem", "max"}``.
        ibn: Whether to use an IBN-Net backbone variant when available.
        feat_dim: Dimensionality of the global feature vector. This must match
            the backbone's output channel width (``2048`` for ResNet-50); it is
            derived from the backbone at build time, so a value that differs
            from the backbone width is ignored rather than freely applied.
    """

    name: str = "resnet50"
    pretrained: bool = True
    last_stride: int = 1
    pooling: str = "gem"
    ibn: bool = False
    feat_dim: int = 2048


@dataclass
class LossConfig:
    """Loss-function configuration.

    Attributes:
        id_weight: Weight of the identity (cross-entropy) loss.
        triplet_weight: Weight of the batch-hard triplet loss.
        triplet_margin: Margin used by the triplet loss when not using the
            soft-margin variant.
        soft_margin: Whether to use the soft-margin (softplus) triplet loss
            instead of the hard margin.
        label_smoothing: Label-smoothing epsilon for the cross-entropy loss.
        center_loss: Whether to enable the center loss term.
        center_weight: Weight applied to the center loss term.
        center_lr: Learning rate for the dedicated center-loss SGD optimizer.
    """

    id_weight: float = 1.0
    triplet_weight: float = 1.0
    triplet_margin: float = 0.3
    soft_margin: bool = False
    label_smoothing: float = 0.1
    center_loss: bool = False
    center_weight: float = 0.0005
    center_lr: float = 0.5  # Bag-of-Tricks (Luo et al., 2019) center-loss SGD LR.


@dataclass
class OptimConfig:
    """Optimizer and learning-rate-schedule configuration.

    Attributes:
        name: Optimizer name (e.g. ``"adam"``).
        lr: Base learning rate.
        weight_decay: Weight decay (L2 regularization) coefficient.
        scheduler: Scheduler name, one of
            ``{"warmup_multistep", "warmup_cosine", "cosine"}``.
        milestones: Epoch indices at which the LR is decayed (multistep only).
        gamma: Multiplicative LR decay factor at each milestone.
        warmup_epochs: Number of warmup epochs at the start of training.
        warmup_factor: Initial LR multiplier at the start of warmup.
    """

    name: str = "adam"
    lr: float = 0.00035
    weight_decay: float = 0.0005
    scheduler: str = "warmup_multistep"
    milestones: list[int] = field(default_factory=lambda: [30, 50])
    gamma: float = 0.1
    warmup_epochs: int = 10
    warmup_factor: float = 0.01


@dataclass
class EvalConfig:
    """Evaluation-protocol configuration.

    Attributes:
        feat_norm: Whether to L2-normalize features (cosine metric) at eval.
        flip_tta: Whether to average features of the image and its horizontal
            flip at test time.
        rerank: Whether to apply k-reciprocal re-ranking.
        rerank_k1: ``k1`` parameter of k-reciprocal re-ranking.
        rerank_k2: ``k2`` parameter of k-reciprocal re-ranking.
        rerank_lambda: Mixing weight between original and Jaccard distance.
        max_rank: Maximum rank computed for the CMC curve.
    """

    feat_norm: bool = True
    flip_tta: bool = True
    rerank: bool = True
    rerank_k1: int = 20
    rerank_k2: int = 6
    rerank_lambda: float = 0.3
    max_rank: int = 50


@dataclass
class TrainConfig:
    """Top-level training-loop configuration.

    Attributes:
        max_epochs: Total number of training epochs.
        amp: Whether to use automatic mixed precision (CUDA only).
        seed: Global random seed for reproducibility.
        device: Compute device, ``"cuda"`` or ``"cpu"``.
        log_period: Log every ``log_period`` iterations.
        eval_period: Run evaluation every ``eval_period`` epochs.
        output_dir: Directory where checkpoints and logs are written.
    """

    max_epochs: int = 60
    amp: bool = True
    seed: int = 42
    device: str = "cuda"
    log_period: int = 50
    eval_period: int = 10
    output_dir: str = "outputs"


def _build_section(cls: type, value: Any) -> Any:
    """Construct a dataclass section from a mapping, ignoring unknown keys.

    Args:
        cls: The target dataclass type.
        value: A mapping of field values, or anything falsy to use defaults.

    Returns:
        An instance of ``cls`` populated from ``value`` and defaults.
    """
    if value is None:
        return cls()
    if is_dataclass(value) and isinstance(value, cls):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping for {cls.__name__}, got {type(value)!r}.")
    valid = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in value.items() if k in valid}
    return cls(**filtered)


@dataclass
class Config:
    """Top-level configuration composing all section configs.

    Attributes:
        data: Data / augmentation / dataloader settings.
        model: Backbone and head settings.
        loss: Loss-function settings.
        optim: Optimizer and scheduler settings.
        eval: Evaluation-protocol settings.
        train: Training-loop settings.
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Config:
        """Build a :class:`Config` from a (possibly partial) nested mapping.

        Unknown top-level sections and unknown keys within a section are
        ignored; missing sections / keys fall back to their defaults.

        Args:
            d: A nested mapping with optional ``data``/``model``/``loss``/
                ``optim``/``eval``/``train`` sub-mappings.

        Returns:
            A fully-populated :class:`Config` instance.
        """
        d = d or {}
        cfg = cls(
            data=_build_section(DataConfig, d.get("data")),
            model=_build_section(ModelConfig, d.get("model")),
            loss=_build_section(LossConfig, d.get("loss")),
            optim=_build_section(OptimConfig, d.get("optim")),
            eval=_build_section(EvalConfig, d.get("eval")),
            train=_build_section(TrainConfig, d.get("train")),
        )
        return cfg.validate()

    def validate(self) -> Config:
        """Validate constrained and coupled fields, failing fast on misconfig.

        Checks enum-like fields against the values the downstream builders
        actually accept (case-insensitively, matching their ``.lower()``
        dispatch) and the ``batch_size`` / ``num_instances`` coupling required
        by :class:`~reid.data.sampler.RandomIdentitySampler`.

        Returns:
            ``self``, to allow fluent use.

        Raises:
            ValueError: If any field holds an unsupported value.
        """
        if self.data.num_instances < 1:
            raise ValueError(f"data.num_instances ({self.data.num_instances}) must be >= 1.")
        if self.data.batch_size < self.data.num_instances:
            raise ValueError(
                f"data.batch_size ({self.data.batch_size}) must be "
                f">= data.num_instances ({self.data.num_instances})."
            )
        if self.data.batch_size % self.data.num_instances != 0:
            raise ValueError(
                f"data.batch_size ({self.data.batch_size}) must be divisible "
                f"by data.num_instances ({self.data.num_instances})."
            )

        pooling = self.model.pooling.lower()
        valid_pooling = {"avg", "gem", "max"}
        if pooling not in valid_pooling:
            raise ValueError(
                f"model.pooling ({self.model.pooling!r}) must be one of {sorted(valid_pooling)}."
            )

        scheduler = self.optim.scheduler.lower()
        valid_scheduler = {"warmup_multistep", "warmup_cosine", "cosine"}
        if scheduler not in valid_scheduler:
            raise ValueError(
                f"optim.scheduler ({self.optim.scheduler!r}) must be one of "
                f"{sorted(valid_scheduler)}."
            )

        optim_name = self.optim.name.lower()
        valid_optim = {"adam", "sgd"}
        if optim_name not in valid_optim:
            raise ValueError(
                f"optim.name ({self.optim.name!r}) must be one of {sorted(valid_optim)}."
            )

        # Only meaningful for the multistep scheduler; guard the empty
        # milestones list used by cosine scheduling.
        if self.optim.milestones and max(self.optim.milestones) > self.train.max_epochs:
            raise ValueError(
                f"max(optim.milestones)={max(self.optim.milestones)} exceeds "
                f"train.max_epochs ({self.train.max_epochs})."
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize this config to a plain nested dictionary.

        Returns:
            A nested ``dict`` mirroring the dataclass structure, suitable for
            YAML/JSON serialization.
        """
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a :class:`Config` from a YAML file.

        Args:
            path: Path to a YAML file with the nested config structure.

        Returns:
            A :class:`Config` instance constructed from the file contents.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Top-level YAML in {path} must be a mapping, got {type(data)!r}.")
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        """Write this config to a YAML file.

        Args:
            path: Destination path. Parent directories are created as needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, default_flow_style=False)


__all__ = [
    "DataConfig",
    "ModelConfig",
    "LossConfig",
    "OptimConfig",
    "EvalConfig",
    "TrainConfig",
    "Config",
]
