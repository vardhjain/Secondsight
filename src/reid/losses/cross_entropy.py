"""Label-smoothing cross-entropy loss for identity classification.

This module implements :class:`CrossEntropyLabelSmooth`, the soft-label
cross-entropy used by the BNNeck "strong baseline" to regularize the
identity classification head. Label smoothing prevents the model from
becoming over-confident on the training identities, which improves
generalization to the unseen query/gallery identities at test time.

Only :mod:`torch` is required, so this module stays importable in
lightweight environments without torchvision.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CrossEntropyLabelSmooth(nn.Module):
    r"""Cross-entropy loss with label smoothing.

    The target distribution for each sample assigns ``1 - epsilon`` probability
    mass to the ground-truth class and spreads ``epsilon`` uniformly across all
    classes:

    .. math::

        q_i = (1 - \epsilon)\,\mathbb{1}[i = y] + \frac{\epsilon}{C}

    where ``C`` is the number of classes. The loss is the cross-entropy between
    this smoothed target ``q`` and the log-softmax of the network logits.

    Args:
        num_classes: Number of identity classes ``C``.
        epsilon: Smoothing factor in ``[0, 1]``. ``0.0`` recovers standard
            cross-entropy. Defaults to ``0.1``.
    """

    def __init__(self, num_classes: int, epsilon: float = 0.1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute the smoothed cross-entropy loss.

        Args:
            inputs: Raw classifier logits of shape ``(N, C)``.
            targets: Ground-truth class indices of shape ``(N,)``.

        Returns:
            Scalar loss tensor (mean over the batch).
        """
        log_probs = self.log_softmax(inputs)
        # Build the one-hot target on the same device/dtype as the log-probs.
        targets_onehot = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1.0)
        targets_onehot = (1.0 - self.epsilon) * targets_onehot + self.epsilon / self.num_classes
        loss = (-targets_onehot * log_probs).sum(dim=1)
        return loss.mean()
