"""Segmentation loss: cross-entropy with optional class weights and foreground Dice.

Defect masks are mostly background. Plain pixel cross-entropy is dominated by
the background pixels, so a model can reach a low loss while missing small
defects. Two standard counter-measures are offered:

* class weights from training pixel frequencies (same "balanced"/"sqrt"
  formulas as classification, see class_weights.weights_from_counts);
* a soft Dice term over the foreground classes, which scores each class by
  overlap ratio and therefore does not shrink with the class's pixel count.

Pixels labelled 255 are training exclusions for every term.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

SEGMENTATION_LOSSES = ("ce", "ce_dice")
IGNORE_INDEX = 255


def pixel_class_weights(dataset, num_classes: int, mode: str, class_names):
    """Class weights from the training masks' pixel counts at training size.

    Returns (weights or None, counts or None). Absent classes keep their output
    channel with neutral weight; only present-class frequencies are normalized.
    """
    from class_weights import weights_from_counts

    if mode == "none":
        return None, None
    counts = dataset.class_pixel_counts(num_classes)
    present = [count for count in counts if count > 0]
    if not present:
        raise ValueError("학습 가능한 마스크 픽셀이 없습니다. 255는 학습 제외 픽셀입니다.")
    present_weights = iter(weights_from_counts(present, mode))
    # Keep every channel and penalize false positives for absent classes too.
    # Their target never occurs, so a neutral positive CE weight is sufficient.
    return [next(present_weights) if count else 1.0 for count in counts], counts



class SegmentationLoss(nn.Module):
    def __init__(self, num_classes: int, *, loss: str = "ce", weights=None,
                 label_smoothing: float = 0.0):
        super().__init__()
        if loss not in SEGMENTATION_LOSSES:
            raise ValueError(f"segmentation loss must be one of {SEGMENTATION_LOSSES}")
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        weight = None if weights is None else torch.as_tensor(weights, dtype=torch.float32)
        if weight is not None and (weight.shape != (num_classes,) or not torch.isfinite(weight).all()
                                   or not (weight > 0).all()):
            raise ValueError("segmentation class weights must be positive and match num_classes")
        self.register_buffer("weight", weight)
        self.num_classes, self.loss, self.label_smoothing = num_classes, loss, float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()  # AMP: keep the reductions in FP32
        if not bool((targets != IGNORE_INDEX).any()):
            # A batch whose every pixel is excluded has nothing to learn; plain CE
            # would return NaN (0/0) and poison the weights.
            return logits.sum() * 0.0
        weight = None if self.weight is None else self.weight.to(logits.device)
        ce = F.cross_entropy(logits, targets, weight=weight, ignore_index=IGNORE_INDEX,
                             label_smoothing=self.label_smoothing)
        if self.loss == "ce" or self.num_classes < 2:
            return ce
        valid = (targets != IGNORE_INDEX).unsqueeze(1)
        probabilities = logits.softmax(dim=1) * valid
        onehot = F.one_hot(targets.masked_fill(~valid[:, 0], 0), self.num_classes)
        onehot = onehot.permute(0, 3, 1, 2).to(probabilities.dtype) * valid
        dims = (0, 2, 3)
        intersection = (probabilities * onehot).sum(dims)
        denominator = probabilities.sum(dims) + onehot.sum(dims)
        dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
        return ce + (1.0 - dice[1:].mean())  # foreground classes; class 0 is background
