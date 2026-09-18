"""정상 특징 후보와 거리 행렬 크기를 제한하는 PatchCore 샘플링."""

import math
import torch


class FeatureReservoir:
    """무작위 우선순위로 전체 스트림에서 제한된 수의 후보만 보관한다."""
    def __init__(self, limit, seed=0):
        if int(limit) < 1:
            raise ValueError("최대 후보 패치 수는 1 이상 필요")
        self.limit = int(limit)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.features = self.keys = None
        self.seen = 0

    def add(self, features):
        if features.ndim != 2 or features.shape[0] < 1 or not torch.isfinite(features).all():
            raise ValueError("추출된 패치 특징이 비어 있거나 유한하지 않습니다")
        n = features.shape[0]
        self.seen += n
        keys = torch.rand(n, generator=self.generator)
        if n > self.limit:
            keep = keys.topk(self.limit, largest=False).indices
            keys, features = keys[keep], features[keep.to(features.device)]
        features = features.detach().float().cpu()
        if self.features is not None:
            features = torch.cat((self.features, features))
            keys = torch.cat((self.keys, keys))
        if len(keys) > self.limit:
            keep = keys.topk(self.limit, largest=False).indices
            keys, features = keys[keep], features[keep]
        self.keys, self.features = keys, features


def coreset(features, count, *, seed=0, projection_dim=128, cancel=None, progress=None):
    """투영 공간에서 k-center를 고르되 저장하는 특징은 원래 차원을 유지한다."""
    if features.ndim != 2 or features.shape[0] == 0 or not torch.isfinite(features).all():
        raise ValueError("코어셋 입력 특징 오류")
    n, dimension = features.shape
    count = min(max(int(count), 1), n)
    if count == n:
        return features.clone()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    projected = features.float().cpu()
    if dimension > projection_dim:
        projection = torch.randn(dimension, projection_dim, generator=generator) / math.sqrt(projection_dim)
        projected = projected @ projection
    norms = projected.square().sum(dim=1)
    selected = torch.empty(count, dtype=torch.long)
    current = int(torch.randint(n, (1,), generator=generator).item())
    minimum = torch.full((n,), float("inf"))
    for index in range(count):
        if cancel:
            cancel()
        selected[index] = current
        distance = (norms + norms[current] - 2 * (projected @ projected[current])).clamp_min_(0)
        minimum = torch.minimum(minimum, distance)
        minimum[selected[:index + 1]] = -1
        current = int(minimum.argmax().item())
        if progress and (index % 50 == 0 or index + 1 == count):
            progress(index + 1, count, f"대표 패치 선별 {index + 1:,}/{count:,}")
    return features[selected.to(features.device)]
