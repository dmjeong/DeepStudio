"""커스텀 모델과 기존 외부 엔진 분류 모델의 가중 손실."""

import torch
from torch import nn
from torch.nn import functional as F


class WeightedClassificationLoss(nn.Module):
    """가중치 합 대신 배치 크기로 평균하여 소수 클래스의 기여도를 유지."""

    def __init__(self, weights=None, label_smoothing=0.0):
        super().__init__()
        if not 0 <= label_smoothing <= 1:
            raise ValueError("Label smoothing은 0~1 범위 필요")
        weight = None if weights is None else torch.as_tensor(weights, dtype=torch.float32)
        if weight is not None and (weight.ndim != 1 or not torch.isfinite(weight).all()
                                   or not (weight > 0).all()):
            raise ValueError("클래스 가중치는 유한한 양수의 1차원 배열 필요")
        self.register_buffer("weight", weight)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets):
        # 큰 역빈도 가중치를 FP16으로 먼저 변환하면 AMP의 FP32 CE 전에
        # inf가 생긴다. 저정밀 logits도 손실 계산은 FP32로 수행한다.
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        weight = self.weight
        if weight is not None:
            weight = weight.to(device=logits.device, dtype=logits.dtype)
        # weight의 데이터 전체 평균이 1이므로 batch=1에서도 상대 가중치 유지.
        return F.cross_entropy(logits, targets, weight=weight,
                               label_smoothing=self.label_smoothing, reduction="none").mean()
