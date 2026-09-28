"""Strong-baseline Re-ID model: ResNet-50 backbone + BNNeck.

This module assembles the full person Re-Identification network used throughout
the project. The architecture follows the well-known "Bag of Tricks" strong
baseline (Luo et al., CVPRW 2019):

#. A ResNet-50 backbone (optionally ``last_stride=1`` and/or IBN) produces a
   convolutional feature map.
#. A configurable global pooling layer (average, max or GeM) collapses the
   feature map into a global feature vector ``global_feat``.
#. A **BNNeck**, a 1D batch-norm layer whose bias is frozen, normalizes the
   feature into ``feat`` before classification.
#. A bias-free linear classifier maps ``feat`` to identity logits.

During training the model returns ``(cls_score, global_feat)`` so that the
classification loss operates on the post-BNNeck logits while the triplet (and
optional center) losses operate on the pre-BNNeck ``global_feat``. During
evaluation the model returns the post-BNNeck ``feat``, which empirically yields
better retrieval performance.

Weight initialization follows the original notebook: Kaiming initialization for
the BNNeck and a small-std normal initialization for the classifier.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from torch import Tensor, nn

from reid.models.backbone import build_backbone
from reid.models.pooling import build_pooling

if TYPE_CHECKING:
    from reid.config import Config

logger = logging.getLogger(__name__)


def weights_init_kaiming(m: nn.Module) -> None:
    """Applies Kaiming-style initialization to a module.

    Linear and convolutional layers receive Kaiming-normal weights, while affine
    batch-norm layers are initialized to unit scale and zero shift. This mirrors
    the initialization used for the BNNeck in the reference notebook.

    Args:
        m: The module to initialize in place.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.modules.conv._ConvNd):
        nn.init.kaiming_normal_(m.weight, a=0, mode="fan_in")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.modules.batchnorm._BatchNorm) and m.affine:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m: nn.Module) -> None:
    """Initializes a classifier layer with a small-std normal distribution.

    Args:
        m: The module to initialize in place. Only linear layers are affected.
    """
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class ReIDModel(nn.Module):
    """ResNet-50 + BNNeck Re-ID model.

    Attributes:
        backbone: Convolutional feature extractor returning ``[N, feat_dim, h, w]``.
        pool: Global pooling layer mapping ``[N, feat_dim, h, w]`` to
            ``[N, feat_dim]``.
        bottleneck: BNNeck batch-norm layer (bias frozen) applied to the pooled
            feature.
        classifier: Bias-free linear identity classifier.
        num_classes: Number of training identities.
        feat_dim: Dimensionality of the global feature vector.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        last_stride: int = 1,
        pooling: str = "gem",
        ibn: bool = False,
        feat_dim: int = 2048,
    ) -> None:
        """Builds the Re-ID model.

        Args:
            num_classes: Number of identity classes for the classifier head.
            backbone: Backbone identifier passed to :func:`build_backbone`.
            pretrained: Whether to initialize the backbone from ImageNet weights.
            last_stride: Stride of the first ``layer4`` block (``1`` removes the
                downsampling stride for higher resolution features).
            pooling: Global pooling type, one of ``"avg"``, ``"gem"`` or
                ``"max"``.
            ibn: Whether to use the ResNet-50-IBN-a backbone from ``torch.hub``.
                Loading failures raise instead of falling back.
            feat_dim: Expected feature dimensionality. Must match the backbone's
                output channel count (``2048`` for ResNet-50).
        """
        super().__init__()
        self.num_classes = num_classes

        self.backbone, backbone_dim = build_backbone(
            name=backbone,
            pretrained=pretrained,
            last_stride=last_stride,
            ibn=ibn,
        )
        # Honour the backbone's true output width so downstream layers stay
        # consistent even if a mismatched feat_dim was requested.
        if feat_dim != backbone_dim:
            logger.warning(
                "Requested feat_dim=%d does not match backbone output width %d; using %d.",
                feat_dim,
                backbone_dim,
                backbone_dim,
            )
        self.feat_dim = backbone_dim

        self.pool = build_pooling(pooling)

        # BNNeck: batch-norm before the classifier with a frozen bias.
        self.bottleneck = nn.BatchNorm1d(self.feat_dim)
        self.bottleneck.bias.requires_grad_(False)

        self.classifier = nn.Linear(self.feat_dim, num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Runs a forward pass.

        Args:
            x: Input image batch of shape ``[N, 3, H, W]``.

        Returns:
            During training: a tuple ``(cls_score, global_feat)`` where
            ``cls_score`` has shape ``[N, num_classes]`` and ``global_feat`` has
            shape ``[N, feat_dim]``.

            During evaluation: the post-BNNeck feature ``feat`` of shape
            ``[N, feat_dim]``.
        """
        feat_map = self.backbone(x)
        global_feat = self.pool(feat_map)
        feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat
        return feat

    def extract_features(self, x: Tensor) -> Tensor:
        """Extracts post-BNNeck evaluation features.

        This forces the module into evaluation mode for the duration of the
        forward pass so the returned features always correspond to the post-BN
        ``feat`` used at retrieval time, regardless of the module's current
        training flag.

        Args:
            x: Input image batch of shape ``[N, 3, H, W]``.

        Returns:
            The post-BNNeck feature tensor of shape ``[N, feat_dim]``.
        """
        was_training = self.training
        self.eval()
        try:
            feat_map = self.backbone(x)
            global_feat = self.pool(feat_map)
            feat = self.bottleneck(global_feat)
        finally:
            if was_training:
                self.train()
        return feat


def build_model(cfg: Config, num_classes: int, *, pretrained: bool | None = None) -> ReIDModel:
    """Builds a :class:`ReIDModel` from a configuration object.

    Args:
        cfg: The full project configuration. Only ``cfg.model`` fields are used.
        num_classes: Number of training identities for the classifier head.
        pretrained: Whether to initialize the backbone from ImageNet weights.
            ``None`` (the default) uses ``cfg.model.pretrained``. Inference
            paths that immediately load trained weights pass ``False`` so no
            ImageNet download happens.

    Returns:
        A configured :class:`ReIDModel`.
    """
    model_cfg = cfg.model
    return ReIDModel(
        num_classes=num_classes,
        backbone=model_cfg.name,
        pretrained=model_cfg.pretrained if pretrained is None else pretrained,
        last_stride=model_cfg.last_stride,
        pooling=model_cfg.pooling,
        ibn=model_cfg.ibn,
        feat_dim=model_cfg.feat_dim,
    )


def load_trained_model(
    weights: str | Path,
    cfg: Config | None = None,
    *,
    map_location: str = "cpu",
) -> tuple[ReIDModel, Config]:
    """Rebuilds a trained :class:`ReIDModel` from a checkpoint file.

    The architecture comes from the config embedded in the checkpoint when one
    is present, because rebuilding from a different YAML can silently produce
    wrong features (a ``last_stride`` mismatch, for example, changes no
    parameter names). The checkpoint supplies ``cfg.model`` and the input
    geometry ``cfg.data.height`` and ``cfg.data.width``; every other section
    comes from ``cfg`` (or the :class:`~reid.config.Config` defaults). The
    backbone is built without ImageNet weights, the classifier is sized from
    the checkpoint, and the weights are loaded strictly.

    Args:
        weights: Path to a checkpoint written by the trainer (``best.pth`` or
            ``last.pth``) or by :func:`reid.utils.checkpoint.save_model`
            (``model_final.pth``), or to a bare ``state_dict``.
        cfg: Base configuration for everything the checkpoint does not
            describe. It is not modified.
        map_location: Device mapping used when reading the checkpoint.

    Returns:
        A tuple ``(model, effective_cfg)`` with the model in eval mode on the
        CPU and the configuration it was built from.

    Raises:
        FileNotFoundError: If ``weights`` does not exist.
        RuntimeError: If the classifier size cannot be inferred or the weights
            do not match the rebuilt architecture exactly.
    """
    from reid.config import Config
    from reid.utils.checkpoint import (
        infer_num_classes_from_checkpoint,
        load_model,
        read_checkpoint_config,
    )

    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found: {path}")

    effective = copy.deepcopy(cfg) if cfg is not None else Config()
    embedded = read_checkpoint_config(path)
    if embedded is not None:
        model_section = embedded.get("model")
        if isinstance(model_section, dict):
            ckpt_model = Config.from_dict({"model": model_section}).model
            if cfg is not None:
                for field in dataclasses.fields(ckpt_model):
                    ours = getattr(cfg.model, field.name)
                    theirs = getattr(ckpt_model, field.name)
                    if field.name != "pretrained" and ours != theirs:
                        logger.warning(
                            "Using model.%s=%r from the checkpoint instead of %r from the config.",
                            field.name,
                            theirs,
                            ours,
                        )
            effective.model = ckpt_model
        data_section = embedded.get("data")
        if isinstance(data_section, dict):
            for key in ("height", "width"):
                if data_section.get(key) is not None:
                    setattr(effective.data, key, int(data_section[key]))
    else:
        logger.warning("Checkpoint %s has no embedded config; using the given config.", path)

    num_classes = infer_num_classes_from_checkpoint(path, fallback=0)
    if num_classes <= 0:
        raise RuntimeError(f"Could not infer the classifier size from {path}.")

    model = build_model(effective, num_classes, pretrained=False)
    load_model(model, path, map_location=map_location, strict=True)
    model.to("cpu")
    model.eval()
    return model, effective
