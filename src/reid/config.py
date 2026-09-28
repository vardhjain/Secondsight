"""Typed configuration for the Re-ID pipeline.

The whole training / evaluation pipeline is driven by a single nested
:class:`Config` dataclass. Each logical area of the system has its own small
dataclass (data, model, loss, optimization, evaluation, training), and the
top-level :class:`Config` simply composes them.

Configs can be serialized to / from YAML and plain dictionaries. Construction
from external data is forgiving: unknown keys are ignored (with a logged warning
so that typos do not go unnoticed) and missing keys fall back to the dataclass
defaults, so older or partially-specified YAML files keep working as the schema
evolves. Numeric strings such as ``"5e-4"``, which PyYAML reads as text, are
coerced to the annotated ``float`` or ``int`` type.

The default values defined here are the single source of truth and are mirrored
by ``configs/default.yaml``. The headline "strong baseline" recipe lives in
``configs/market1501_strong_baseline.yaml``.
"""

from __future__ import annotations

import logging
import typing
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_SECTION_NAMES = ("data", "model", "loss", "optim", "eval", "train")


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
        val_ids: Number of training identities held out as a validation
            query/gallery split for model selection. ``0`` disables the split,
            in which case periodic evaluation runs on the test split and serves
            as monitoring only.
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
    val_ids: int = 0


@dataclass
class ModelConfig:
    """Backbone and Re-ID head configuration.

    Attributes:
        name: Backbone architecture name (e.g. ``"resnet50"``).
        pretrained: Whether to initialize the backbone from ImageNet weights.
        last_stride: Stride of the final ResNet stage (``layer4``), either ``1``
            or ``2``. Setting this to ``1`` increases the spatial resolution of
            the feature map and is a key part of the strong baseline recipe.
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
        device: Compute device. One of ``"cuda"`` (or ``"cuda:N"``), ``"mps"``,
            ``"cpu"``, or ``"auto"``, which picks CUDA, then MPS, then CPU.
        log_period: Log every ``log_period`` iterations.
        eval_period: Run evaluation every ``eval_period`` epochs. The final
            epoch is always evaluated as well.
        output_dir: Directory where checkpoints and logs are written.
        deterministic: Whether to enable deterministic cuDNN and PyTorch
            kernels for run-to-run reproducibility. Turning it off enables
            ``cudnn.benchmark``, which is faster for the fixed input size.
    """

    max_epochs: int = 60
    amp: bool = True
    seed: int = 42
    device: str = "cuda"
    log_period: int = 50
    eval_period: int = 10
    output_dir: str = "outputs"
    deterministic: bool = True


def _coerce_value(cls: type, key: str, expected: Any, value: Any) -> Any:
    """Coerce a raw config value to its annotated scalar type when safe.

    PyYAML follows YAML 1.1, so literals such as ``5e-4`` or ``1e4`` load as
    strings. This converts such strings (and integers given for float fields)
    to the annotated type, so a bad value fails here with a clear message
    instead of deep inside the optimizer or the loss.

    Args:
        cls: The dataclass that owns the field (used in error messages).
        key: The field name.
        expected: The resolved type annotation of the field.
        value: The raw value from YAML or a mapping.

    Returns:
        The coerced value, or ``value`` unchanged when no coercion applies.

    Raises:
        TypeError: If a string cannot be parsed as the annotated number type.
    """
    if isinstance(value, bool) or expected not in (float, int):
        return value
    try:
        if expected is float and isinstance(value, (int, str)):
            return float(value)
        if expected is int and isinstance(value, str):
            return int(value)
    except ValueError as exc:
        raise TypeError(
            f"{cls.__name__}.{key} expects {expected.__name__}, got {value!r}."
        ) from exc
    return value


def _build_section(cls: type, value: Any) -> Any:
    """Construct a dataclass section from a mapping, ignoring unknown keys.

    Unknown keys are dropped with a logged warning, and numeric strings are
    coerced to the annotated field type (see :func:`_coerce_value`).

    Args:
        cls: The target dataclass type.
        value: A mapping of field values, or ``None`` to use defaults.

    Returns:
        An instance of ``cls`` populated from ``value`` and defaults.

    Raises:
        TypeError: If ``value`` is not a mapping, or a numeric field holds an
            unparseable string.
    """
    if value is None:
        return cls()
    if is_dataclass(value) and isinstance(value, cls):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping for {cls.__name__}, got {type(value)!r}.")
    valid = {f.name for f in fields(cls)}
    unknown = sorted(str(k) for k in set(value) - valid)
    if unknown:
        logger.warning("Ignoring unknown %s keys: %s", cls.__name__, unknown)
    hints = typing.get_type_hints(cls)
    filtered = {k: _coerce_value(cls, k, hints.get(k), v) for k, v in value.items() if k in valid}
    return cls(**filtered)


def _check_choice(name: str, value: str, valid: set[str]) -> str:
    """Check a case-insensitive, enum-like string field.

    Args:
        name: Dotted field name used in the error message.
        value: The configured value.
        valid: The accepted lower-case values.

    Returns:
        The lower-cased value.

    Raises:
        ValueError: If ``value`` is not one of ``valid``.
    """
    lowered = value.lower()
    if lowered not in valid:
        raise ValueError(f"{name} ({value!r}) must be one of {sorted(valid)}.")
    return lowered


def default_config_path(name: str = "market1501_strong_baseline.yaml") -> Path:
    """Locate a shipped YAML config file.

    Installed wheels carry the configs inside the package as ``reid/configs``.
    In a source checkout they live in the repository's top-level ``configs``
    directory instead, so that copy is the fallback.

    Args:
        name: File name of the config, for example ``"default.yaml"``.

    Returns:
        The packaged copy when it exists, otherwise the repository checkout copy
        (which may not exist outside a checkout).
    """
    packaged = resources.files("reid") / "configs" / name
    if packaged.is_file():
        return Path(str(packaged))
    return Path(__file__).resolve().parents[2] / "configs" / name


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
        ignored with a logged warning, and missing sections or keys fall back
        to their defaults.

        Args:
            d: A nested mapping with optional ``data``/``model``/``loss``/
                ``optim``/``eval``/``train`` sub-mappings.

        Returns:
            A fully-populated and validated :class:`Config` instance.
        """
        d = d or {}
        unknown = sorted(str(k) for k in set(d) - set(_SECTION_NAMES))
        if unknown:
            logger.warning("Ignoring unknown config sections: %s", unknown)
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
        dispatch), the PK sampler geometry that batch-hard triplet mining
        needs, and the numeric ranges of the training and evaluation settings.
        Schedule settings that are merely wasteful, such as a multistep
        milestone or the warmup reaching past ``train.max_epochs`` in a short
        smoke run, only log a warning, so such configs stay loadable.

        Returns:
            ``self``, to allow fluent use.

        Raises:
            ValueError: If any field holds an unsupported value.
        """
        data, optim, train, ev = self.data, self.optim, self.train, self.eval

        if data.num_instances < 2:
            raise ValueError(
                f"data.num_instances ({data.num_instances}) must be >= 2 so that every "
                "triplet anchor has a positive other than itself."
            )
        if data.batch_size % data.num_instances != 0:
            raise ValueError(
                f"data.batch_size ({data.batch_size}) must be divisible "
                f"by data.num_instances ({data.num_instances})."
            )
        if data.batch_size // data.num_instances < 2:
            raise ValueError(
                f"data.batch_size ({data.batch_size}) must hold at least two identities of "
                f"data.num_instances ({data.num_instances}) images each for batch-hard "
                "triplet mining."
            )
        if data.val_ids < 0:
            raise ValueError(f"data.val_ids ({data.val_ids}) must be >= 0.")

        _check_choice("model.pooling", self.model.pooling, {"avg", "gem", "max"})
        if self.model.last_stride not in (1, 2):
            raise ValueError(f"model.last_stride ({self.model.last_stride!r}) must be 1 or 2.")

        scheduler = _check_choice(
            "optim.scheduler", optim.scheduler, {"warmup_multistep", "warmup_cosine", "cosine"}
        )
        _check_choice("optim.name", optim.name, {"adam", "sgd"})
        if optim.warmup_epochs < 0:
            raise ValueError(f"optim.warmup_epochs ({optim.warmup_epochs}) must be >= 0.")

        for name, value in (
            ("train.max_epochs", train.max_epochs),
            ("train.log_period", train.log_period),
            ("train.eval_period", train.eval_period),
            ("eval.max_rank", ev.max_rank),
            ("eval.rerank_k1", ev.rerank_k1),
            ("eval.rerank_k2", ev.rerank_k2),
        ):
            if value < 1:
                raise ValueError(f"{name} ({value}) must be >= 1.")
        if not 0.0 <= ev.rerank_lambda <= 1.0:
            raise ValueError(f"eval.rerank_lambda ({ev.rerank_lambda}) must be in [0, 1].")

        # The last trained epoch runs with scheduler.last_epoch == max_epochs - 1,
        # so a milestone at or beyond max_epochs never fires. That is harmless
        # (every short smoke run hits it), so it is a warning, not an error.
        if scheduler == "warmup_multistep":
            late = [m for m in optim.milestones if m >= train.max_epochs]
            if late:
                logger.warning(
                    "optim.milestones %s are >= train.max_epochs (%d), so those LR decays "
                    "will never fire.",
                    late,
                    train.max_epochs,
                )
        if optim.warmup_epochs >= train.max_epochs:
            logger.warning(
                "optim.warmup_epochs (%d) >= train.max_epochs (%d), so training ends "
                "before the LR warmup completes.",
                optim.warmup_epochs,
                train.max_epochs,
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize this config to a plain nested dictionary.

        Returns:
            A nested ``dict`` mirroring the dataclass structure, suitable for
            YAML/JSON serialization.
        """
        return asdict(self)

    def to_portable_dict(self) -> dict[str, Any]:
        """Serialize this config without machine-specific absolute paths.

        This is the form embedded in checkpoints, which may be published.
        ``data.root`` is cleared and ``train.output_dir`` is reduced to its last
        path component, so no local user name or directory layout leaks.

        Returns:
            A nested ``dict`` like :meth:`to_dict` with the paths scrubbed.
        """
        d = self.to_dict()
        d["data"]["root"] = None
        d["train"]["output_dir"] = Path(self.train.output_dir).name or "outputs"
        return d

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a :class:`Config` from a YAML file.

        Args:
            path: Path to a YAML file with the nested config structure.

        Returns:
            A :class:`Config` instance constructed from the file contents.

        Raises:
            TypeError: If the top level of the YAML document is not a mapping.
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
    "default_config_path",
]
