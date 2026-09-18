"""
PatchCore — 산업용 이상 탐지 알고리즘

논문: "Towards Total Recall in Industrial Anomaly Detection" (Roth et al., CVPR 2022)

핵심 아이디어:
┌───────────────────────────────────────────────────────────────────────┐
│                        PatchCore Pipeline                             │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─── 학습 (정상 이미지만) ───────────────────────────────────────┐   │
│  │                                                                │   │
│  │  1. 사전학습 백본(ResNet/WideResNet)으로 패치 특징 추출         │   │
│  │     - Layer 2 + Layer 3 의 중간 특징맵 사용                    │   │
│  │     - 각 위치의 특징을 이웃 패치와 집계 (AvgPool)              │   │
│  │                                                                │   │
│  │  2. 메모리 뱅크 구축                                           │   │
│  │     - 모든 정상 패치 특징을 수집                                │   │
│  │     - Coreset Subsampling으로 대표 패치만 선별                  │   │
│  │       (Greedy k-Center 알고리즘)                                │   │
│  │     - 저장: memory_bank.pt (재학습 불필요)                      │   │
│  │                                                                │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌─── 추론 ──────────────────────────────────────────────────────┐   │
│  │                                                                │   │
│  │  1. 테스트 이미지 → 동일한 백본으로 패치 특징 추출              │   │
│  │  2. 각 패치와 메모리 뱅크 간 최근접 거리 계산 (kNN)            │   │
│  │  3. 거리 맵 → 이상 히트맵 (원본 해상도로 업샘플)               │   │
│  │  4. 이미지 레벨 스코어 = max(거리 맵)                          │   │
│  │                                                                │   │
│  │  결과:                                                         │   │
│  │  ┌────────────┬───────────────────────────────────────────┐    │   │
│  │  │ anomaly_map│ (H, W) 픽셀별 이상 정도 히트맵             │    │   │
│  │  │ score      │ 이미지 레벨 이상 스코어 (0~∞)              │    │   │
│  │  └────────────┴───────────────────────────────────────────┘    │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  기존 Reconstruction 방식과의 비교:                                    │
│  ┌────────────────┬──────────────────┬──────────────────┐            │
│  │                │ Reconstruction   │ PatchCore        │            │
│  ├────────────────┼──────────────────┼──────────────────┤            │
│  │ 학습 방식      │ 디코더 역전파     │ 특징 추출만 (1회)│            │
│  │ 학습 시간      │ 수십 에폭        │ 1 에폭 (추출)    │            │
│  │ 정확도         │ 보통             │ SOTA (MVTec AD)  │            │
│  │ XAI            │ 재구성 오차      │ 패치 거리 히트맵  │            │
│  │ 사전학습 필요  │ 아니요           │ 예 (ImageNet)    │            │
│  └────────────────┴──────────────────┴──────────────────┘            │
└───────────────────────────────────────────────────────────────────────┘
"""

import os
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  특징 추출 백본 (WideResNet-50 기반)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PatchCoreBackbone(nn.Module):
    """
    PatchCore 특징 추출 백본

    torchvision의 Wide ResNet-50-2 사용 (ImageNet 사전학습)
    Layer 2, 3의 중간 특징맵을 추출하여 결합

    특징맵 추출 지점:
    ┌────────────────────────────────────────────────────┐
    │ Input (3, 224, 224)                                │
    │   └─ conv1 + bn1 + relu + maxpool                  │
    │       └─ layer1 (skip — 너무 저수준)               │
    │           └─ layer2 → feat2 (512, 28, 28)  ◄ 추출  │
    │               └─ layer3 → feat3 (1024, 14, 14) ◄   │
    │                   └─ layer4 (skip — 너무 고수준)    │
    └────────────────────────────────────────────────────┘

    Layer 2 + 3 조합 이유:
    - Layer 2: 중간 수준 텍스처/패턴 (결함 감지에 적합)
    - Layer 3: 더 넓은 수용 영역의 구조 정보
    - Layer 1: 너무 저수준 (에지, 색상)
    - Layer 4: 너무 추상적 (의미 수준 — 위치 정보 손실)
    """

    def __init__(self, backbone_name: str = "wide_resnet50_2", pretrained: bool = True):
        super().__init__()
        from patchcore_weights import BACKBONES, build_backbone
        base = build_backbone(backbone_name, pretrained)
        self.feat2_channels, self.feat3_channels = BACKBONES[backbone_name][:2]

        # 필요한 레이어만 보존
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        # layer4는 PatchCore에서 사용하지 않음

        # 모든 파라미터 동결 (학습하지 않음)
        for param in self.parameters():
            param.requires_grad = False

        self.eval()

    @property
    def feature_dim(self) -> int:
        """결합된 특징 벡터 차원"""
        return self.feat2_channels + self.feat3_channels

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        패치 특징 추출

        Args:
            x: (B, 3, H, W) 입력 이미지

        Returns:
            features: (B, feat2_ch + feat3_ch, h, w)
                      Layer2, 3의 특징을 동일 해상도로 결합
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feat2 = self.layer2(x)     # (B, 512, 28, 28) for WRN-50-2
        feat3 = self.layer3(feat2)  # (B, 1024, 14, 14)

        # feat3을 feat2 해상도로 업샘플하여 결합
        feat3_up = F.interpolate(
            feat3, size=feat2.shape[2:],
            mode="bilinear", align_corners=False,
        )
        # 채널 방향 결합 → (B, 1536, 28, 28)
        combined = torch.cat([feat2, feat3_up], dim=1)
        return combined


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Coreset Subsampling (Greedy k-Center)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PatchCoreCancelled(RuntimeError):
    """사용자의 취소 요청. 학습 실패와 구분하는 종료 상태."""


def _check_cancel(cancel_callback):
    if cancel_callback is not None and cancel_callback():
        raise PatchCoreCancelled("PatchCore 구축 취소")


def greedy_coreset_sampling(
    features: torch.Tensor,
    sampling_ratio: float = 0.01,
    device: torch.device = None,
    cancel_callback=None,
) -> torch.Tensor:
    """
    Greedy k-Center Coreset 서브샘플링

    전체 N개 패치 특징에서 대표적인 M=N×ratio 개를 선별
    메모리 뱅크 크기를 줄이면서 커버리지를 최대화

    알고리즘:
    ┌─────────────────────────────────────────────────────────┐
    │ 1. 랜덤 시드 포인트 하나 선택                            │
    │ 2. 각 포인트 → 이미 선택된 코어셋까지의 최소 거리 계산    │
    │ 3. 최소 거리가 가장 큰 포인트를 다음 코어셋으로 선택       │
    │ 4. M개가 될 때까지 2-3 반복                              │
    │                                                          │
    │ 직관: "기존 선택과 가장 먼 점"을 순차 추가하면             │
    │       전체 분포를 균일하게 커버하는 대표 집합이 됨          │
    └─────────────────────────────────────────────────────────┘

    Args:
        features: (N, D) 전체 패치 특징
        sampling_ratio: 선별 비율 (0.01 = 1%)
        device: 연산 디바이스

    Returns:
        coreset: (M, D) 선별된 대표 패치 특징
    """
    from patchcore_sampling import coreset
    _check_cancel(cancel_callback)
    if features.ndim != 2 or not 0 < sampling_ratio <= 1:
        raise ValueError("특징 형식 또는 코어셋 비율 오류")
    return coreset(features, max(int(features.shape[0] * sampling_ratio), 1),
                   cancel=lambda: _check_cancel(cancel_callback))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PatchCore 메인 클래스
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PatchCore:
    """
    PatchCore 이상 탐지 엔진

    사용 흐름:
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │  # 1. 학습 (정상 이미지의 DataLoader)                         │
    │  pc = PatchCore(device="cuda")                               │
    │  pc.fit(train_loader, progress_callback=...)                 │
    │                                                              │
    │  # 2. 저장                                                   │
    │  pc.save("memory_bank.pt")                                   │
    │                                                              │
    │  # 3. 로드 + 추론                                            │
    │  pc = PatchCore.load("memory_bank.pt", device="cuda")        │
    │  score, anomaly_map = pc.predict(image_tensor)               │
    │  # score: 이미지 레벨 이상 스코어                              │
    │  # anomaly_map: (H, W) 픽셀별 이상 히트맵                    │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        device: str | torch.device = "cpu",
        sampling_ratio: float = 0.01,
        n_neighbors: int = 9,
        input_size: int = 224,
        pretrained: bool = True,
        backbone_weights: str = "",
        max_candidates: int = 20000,
        max_memory_bank: int = 4096,
        seed: int = 0,
        preprocessing: str = "full_range_v1",
        center_crop: dict | None = None,
    ):
        """
        Args:
            backbone_name: 특징 추출 백본 ("wide_resnet50_2" 또는 "resnet18")
            device: 연산 디바이스
            sampling_ratio: 코어셋 비율 (0.01 = 전체 패치의 1%)
            n_neighbors: kNN의 k값 (거리 평균에 사용)
            input_size: 입력 이미지 크기 (정사각형)
        """
        if (not 0 < sampling_ratio <= 1 or n_neighbors < 1
                or int(n_neighbors) != n_neighbors or int(input_size) < 32):
            raise ValueError("PatchCore 설정 범위 오류")
        self.device = torch.device(device)
        self.backbone_name = backbone_name
        self.sampling_ratio = sampling_ratio
        self.n_neighbors = int(n_neighbors)
        self.input_size = int(input_size)
        self.anomaly_threshold = None
        self.score_normalization = None
        self.calibration = {"evaluable": False, "reason": "임계값 미보정"}

        if min(int(max_candidates), int(max_memory_bank)) < 1 or int(seed) < 0:
            raise ValueError("PatchCore 후보/대표 패치 수와 시드 범위 오류")
        if preprocessing not in ("full_range_v1", "legacy_pil_rgb"):
            raise ValueError("PatchCore 미지원 전처리")
        self.max_candidates, self.max_memory_bank = int(max_candidates), int(max_memory_bank)
        self.seed, self.preprocessing = int(seed), preprocessing
        from center_crop import validate_center_crop
        self.center_crop = validate_center_crop(center_crop)
        self.training_metadata = {}
        self.backbone = PatchCoreBackbone(backbone_name, pretrained=pretrained and not backbone_weights)
        self.weight_source = {"kind": "imagenet" if pretrained else "uninitialized",
                              "backbone": backbone_name, "weights": "IMAGENET1K_V1" if pretrained else "none"}
        if backbone_weights:
            from patchcore_weights import load_backbone_weights
            self.weight_source = load_backbone_weights(self.backbone, backbone_weights, backbone_name)
        self.backbone = self.backbone.to(self.device).eval()

        # 메모리 뱅크 (fit 후 채워짐)
        self.memory_bank: Optional[torch.Tensor] = None

        # 이웃 패치 집계용 풀링 (3×3)
        self._avg_pool = nn.AvgPool2d(3, stride=1, padding=1)

    @torch.no_grad()
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        이미지 배치에서 패치 특징 추출

        Args:
            images: (B, 3, H, W)

        Returns:
            patch_features: (B*h*w, D) 모든 패치 특징을 평탄화
        """
        # 백본으로 특징맵 추출
        feat = self.backbone(images)  # (B, D, h, w)

        # 이웃 패치 집계 (local awareness)
        feat = self._avg_pool(feat)

        B, D, h, w = feat.shape
        # (B, D, h, w) → (B*h*w, D)
        feat = feat.permute(0, 2, 3, 1).reshape(-1, D)
        return feat

    @torch.no_grad()
    def fit(self, train_loader, progress_callback=None, cancel_callback=None, *, append=False):
        """고정된 백본으로 정상 특징을 구축한다. 취소 시 기존 뱅크는 유지한다."""
        _check_cancel(cancel_callback)
        from patchcore_sampling import FeatureReservoir, coreset
        self.backbone.eval()
        existing = self.memory_bank if append else None
        old_count = 0 if existing is None else existing.shape[0]
        if append and existing is None:
            raise ValueError("추가 학습에 사용할 기존 메모리 뱅크 없음")
        if old_count >= self.max_memory_bank:
            raise ValueError(f"기존 대표 패치 {old_count:,}개로 저장 한도 도달. 최대 대표 패치 수를 늘리거나 새 뱅크 구축 선택 필요")
        pool = FeatureReservoir(self.max_candidates, self.seed)
        total_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            _check_cancel(cancel_callback)
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            if images.ndim != 4 or images.shape[1] != 3:
                raise ValueError("PatchCore 입력은 정규화된 (B,3,H,W) 이미지여야 합니다")
            # 고해상도에서 큰 전체 배치를 GPU로 올리지 않는다.
            micro_batch = max(1, min(images.shape[0], 1048576 // (images.shape[2] * images.shape[3])))
            for start in range(0, images.shape[0], micro_batch):
                _check_cancel(cancel_callback)
                part = images[start:start + micro_batch].to(self.device, dtype=torch.float32)
                pool.add(self._extract_features(part))
                del part
            if progress_callback:
                progress_callback(batch_idx + 1, total_batches,
                                  f"정상 특징 추출 {batch_idx + 1}/{total_batches} | 전체 {pool.seen:,} / 후보 {pool.features.shape[0]:,}")
        _check_cancel(cancel_callback)
        if pool.features is None:
            raise ValueError("특징을 추출할 정상 학습 배치 없음")
        target = min(max(int(pool.seen * self.sampling_ratio), 1), self.max_memory_bank - old_count)
        selected = coreset(pool.features, target, seed=self.seed,
                           cancel=lambda: _check_cancel(cancel_callback), progress=progress_callback)
        new_bank = selected.to(self.device)
        if existing is not None:
            new_bank = torch.cat((existing, new_bank))
        _check_cancel(cancel_callback)
        self.memory_bank = new_bank
        self.anomaly_threshold = None
        self.score_normalization = None
        self.calibration = {"evaluable": False, "reason": "메모리 뱅크 변경 후 임계값 재보정 필요"}
        self.training_metadata = {"seen_patches": pool.seen, "candidate_patches": pool.features.shape[0],
                                  "retained_patches": old_count, "new_patches": selected.shape[0],
                                  "candidate_limit_applied": pool.seen > self.max_candidates,
                                  "seed": self.seed, "append": append}
        if progress_callback:
            progress_callback(1, 1, f"메모리 뱅크 구축 완료: {new_bank.shape[0]:,}개 대표 패치")

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        이상 탐지 추론

        Args:
            images: (B, 3, H, W) 입력 이미지

        Returns:
            scores: (B,) 이미지 레벨 이상 스코어
            anomaly_maps: (B, H, W) 픽셀별 이상 히트맵

        추론 파이프라인:
        ┌─────────────────────────────────────────────────────────┐
        │ 입력 → 백본 → 패치 특징 → kNN 거리 → 이상 맵 → 스코어  │
        └─────────────────────────────────────────────────────────┘
        """
        if self.memory_bank is None:
            raise RuntimeError(
                "메모리 뱅크가 아직 구축되지 않았습니다. "
                "fit() 또는 load()를 먼저 실행해 주세요."
            )

        if images.ndim != 4 or images.shape[1] != 3 or images.shape[0] < 1:
            raise ValueError("PatchCore 입력은 (B,3,H,W) 이미지여야 합니다")
        self.backbone.eval()
        micro_batch = max(1, 1048576 // (images.shape[2] * images.shape[3]))
        if images.shape[0] > micro_batch:
            outputs = [self.predict(images[start:start + micro_batch])
                       for start in range(0, images.shape[0], micro_batch)]
            return np.concatenate([out[0] for out in outputs]), np.concatenate([out[1] for out in outputs])
        images = images.to(self.device, dtype=torch.float32)
        B = images.shape[0]

        # 특징 추출
        feat = self.backbone(images)  # (B, D, h, w)
        feat = self._avg_pool(feat)
        _, D, h, w = feat.shape

        # (B*h*w, D)로 평탄화
        patch_features = feat.permute(0, 2, 3, 1).reshape(-1, D)

        # kNN 거리 계산 (배치 단위로 처리하여 메모리 효율화)
        distances = self._compute_knn_distances(patch_features)

        # (B*h*w,) → (B, h, w)
        distance_map = distances.reshape(B, h, w)

        # 원본 해상도로 업샘플
        H, W = images.shape[2], images.shape[3]
        anomaly_maps = F.interpolate(
            distance_map.unsqueeze(1),  # (B, 1, h, w)
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (B, H, W)

        # 가우시안 스무딩 (노이즈 감소)
        anomaly_maps = self._gaussian_smooth(anomaly_maps, sigma=4.0)

        # 이미지 레벨 스코어 = 최대 이상 거리
        scores = anomaly_maps.reshape(B, -1).max(dim=1)[0]

        return scores.cpu().numpy(), anomaly_maps.cpu().numpy()

    def _compute_knn_distances(
        self,
        query: torch.Tensor,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        """
        쿼리 패치와 메모리 뱅크 간 kNN 거리 계산

        대규모 거리 행렬을 배치로 분할하여 GPU 메모리 초과 방지

        Args:
            query: (N, D) 쿼리 패치 특징
            batch_size: 한 번에 처리할 쿼리 수

        Returns:
            distances: (N,) 각 쿼리의 k-최근접 평균 거리
        """
        if self.memory_bank is None or self.memory_bank.shape[0] == 0:
            raise ValueError("비어 있는 메모리 뱅크")
        if query.ndim != 2 or query.shape[1] != self.memory_bank.shape[1]:
            raise ValueError("질의와 메모리 뱅크 특징 차원 불일치")
        k = min(self.n_neighbors, self.memory_bank.shape[0])
        all_distances = []
        # 질의와 뱅크 양쪽을 나누어 거리 행렬을 최대 256x1024로 제한한다.
        for start in range(0, query.shape[0], min(max(int(batch_size), 1), 256)):
            q = query[start:start + min(max(int(batch_size), 1), 256)].float()
            best = None
            for bank_start in range(0, self.memory_bank.shape[0], 1024):
                bank = self.memory_bank[bank_start:bank_start + 1024].to(q.device, dtype=torch.float32)
                distance = (q.square().sum(1, keepdim=True) + bank.square().sum(1).unsqueeze(0)
                            - 2 * (q @ bank.T)).clamp_min_(0).sqrt_()
                candidates = distance if best is None else torch.cat((best, distance), dim=1)
                best = candidates.topk(min(k, candidates.shape[1]), largest=False, dim=1).values
            all_distances.append(best.mean(dim=1))
        return torch.cat(all_distances) if all_distances else query.new_empty((0,))

    @staticmethod
    def _gaussian_smooth(
        maps: torch.Tensor,
        sigma: float = 4.0,
        kernel_size: int = 0,
    ) -> torch.Tensor:
        """
        가우시안 스무딩 (이상 맵 노이즈 제거)

        Args:
            maps: (B, H, W)
            sigma: 가우시안 표준편차
            kernel_size: 커널 크기 (0이면 자동 계산)

        Returns:
            smoothed: (B, H, W)
        """
        if kernel_size == 0:
            kernel_size = int(2 * math.ceil(3 * sigma) + 1)

        # 1D 가우시안 커널 생성
        x = torch.arange(kernel_size, dtype=maps.dtype, device=maps.device)
        x = x - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        # 2D 분리 가능 가우시안 → 가로 + 세로 순차 적용
        maps = maps.unsqueeze(1)  # (B, 1, H, W)
        padding = kernel_size // 2

        # 가로 방향
        k_h = kernel_1d.reshape(1, 1, 1, -1)
        maps = F.conv2d(maps, k_h, padding=(0, padding))

        # 세로 방향
        k_v = kernel_1d.reshape(1, 1, -1, 1)
        maps = F.conv2d(maps, k_v, padding=(padding, 0))

        return maps.squeeze(1)

    # ── 저장 / 로드 ──────────────────────────────────

    def get_score_normalization(self) -> dict:
        """Freeze the display scale independently of the current inference batch."""
        from patchcore_scores import score_normalization, validate_normalization
        if self.score_normalization is None:
            self.score_normalization = score_normalization(self.anomaly_threshold)
        return validate_normalization(self.score_normalization)

    @property
    def normalized_threshold(self):
        from patchcore_scores import normalize_threshold
        return normalize_threshold(self.anomaly_threshold, self.get_score_normalization())

    def save(self, path: str) -> None:
        """
        메모리 뱅크와 설정을 파일로 저장

        저장 내용:
        ┌──────────────────────────────────────────┐
        │ memory_bank: (M, D) 대표 패치 특징        │
        │ backbone_name: 사용된 백본 모델명          │
        │ sampling_ratio: 코어셋 비율               │
        │ n_neighbors: kNN의 k값                    │
        │ input_size: 입력 이미지 크기               │
        │ feature_dim: 특징 벡터 차원                │
        └──────────────────────────────────────────┘
        """
        if self.memory_bank is None:
            raise RuntimeError("저장할 메모리 뱅크가 없습니다.")

        import tempfile
        from pathlib import Path
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "patchcore",
            "schema_version": 4 if self.center_crop is not None else 3,
            "center_crop": self.center_crop,
            "preprocessing": self.preprocessing,
            "weight_source": self.weight_source,
            "training_metadata": self.training_metadata,
            "max_candidates": self.max_candidates, "max_memory_bank": self.max_memory_bank,
            "seed": self.seed,
            "backbone_state_dict": {k: v.detach().cpu() for k, v in self.backbone.state_dict().items()},
            "anomaly_threshold": self.anomaly_threshold,
            "threshold_comparator": ">=",
            "score_definition": "patchcore_smoothed_knn_max",
            "score_normalization": self.get_score_normalization(),
            "calibration": self.calibration,
            "memory_bank": self.memory_bank.cpu(),
            "backbone_name": self.backbone_name,
            "sampling_ratio": self.sampling_ratio,
            "n_neighbors": self.n_neighbors,
            "input_size": self.input_size,
            "feature_dim": self.backbone.feature_dim,
            "memory_bank_size": self.memory_bank.shape[0],
        }
        fd, temporary = tempfile.mkstemp(prefix=".patchcore-", dir=target.parent)
        os.close(fd)
        try:
            # 파일 스트림은 PyTorch ZIP writer의 임시 파일명 제한을 받지 않는다.
            with open(temporary, "wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(
        cls,
        path: str,
        device: str | torch.device = "cpu",
    ) -> "PatchCore":
        """
        저장된 메모리 뱅크에서 PatchCore 복원

        Args:
            path: 저장 파일 경로 (.pt)
            device: 연산 디바이스

        Returns:
            PatchCore 인스턴스 (추론 즉시 가능)
        """
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict) or data.get("type") != "patchcore":
            raise ValueError("PatchCore 체크포인트 형식 오류")
        if data.get("schema_version", 1) not in (1, 2, 3, 4):
            raise ValueError("미지원 PatchCore 체크포인트 버전")
        from center_crop import validate_center_crop
        crop = validate_center_crop(data.get("center_crop"))
        if data.get("schema_version", 1) == 4 and crop is None:
            raise ValueError("PatchCore 중앙 크롭 설정 누락")
        if data.get("schema_version", 1) >= 2 and "backbone_state_dict" not in data:
            raise ValueError("PatchCore 체크포인트 백본 가중치 누락")
        threshold = data.get("anomaly_threshold")
        if threshold is not None and not math.isfinite(float(threshold)):
            raise ValueError("PatchCore 임계값이 유한하지 않습니다")
        if data.get("threshold_comparator", ">=") != ">=":
            raise ValueError("미지원 PatchCore 임계값 비교 연산")
        if data.get("score_definition", "patchcore_smoothed_knn_max") != "patchcore_smoothed_knn_max":
            raise ValueError("PatchCore 점수 정의 불일치")
        from patchcore_scores import score_normalization, validate_normalization
        normalization = (validate_normalization(data["score_normalization"]) if "score_normalization" in data
                         else score_normalization(threshold))

        instance = cls(
            backbone_name=data["backbone_name"],
            device=device,
            sampling_ratio=data["sampling_ratio"],
            n_neighbors=data["n_neighbors"],
            input_size=data.get("input_size", 224),
            pretrained="backbone_state_dict" not in data,
            max_candidates=data.get("max_candidates", 20000),
            max_memory_bank=data.get("max_memory_bank", max(4096, data.get("memory_bank_size", 0))),
            seed=data.get("seed", 0), preprocessing=data.get("preprocessing", "legacy_pil_rgb"),
            center_crop=crop,
        )
        if "backbone_state_dict" in data:
            state = data["backbone_state_dict"]
            if not isinstance(state, dict) or any(
                not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                for value in state.values()
            ):
                raise ValueError("PatchCore 백본 가중치 형식 또는 값 오류")
            instance.backbone.load_state_dict(state, strict=True)
        bank = data["memory_bank"]
        if (not isinstance(bank, torch.Tensor) or bank.ndim != 2 or bank.shape[0] == 0
                or bank.shape[1] != instance.backbone.feature_dim or not torch.isfinite(bank).all()):
            raise ValueError("PatchCore 메모리 뱅크 크기 또는 특징 차원 오류")
        if not bank.is_floating_point():
            raise ValueError("PatchCore 메모리 뱅크는 실수 특징이어야 합니다")
        instance.memory_bank = bank.float().to(instance.device)
        instance.weight_source = data.get("weight_source", {"kind": "saved_patchcore", "backbone": instance.backbone_name})
        instance.training_metadata = data.get("training_metadata", {})
        instance.anomaly_threshold = None if threshold is None else float(threshold)
        instance.score_normalization = normalization
        instance.calibration = data.get("calibration", {"evaluable": False, "reason": "임계값 미보정"})

        return instance

    @torch.no_grad()
    def predict_from_path(
        self,
        image_path: str,
        *, center_crop=...,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        이미지 파일 경로에서 직접 추론

        GUI 추론 위젯에서 간편하게 호출할 수 있는 래퍼:
        ┌────────────────────────────────────────────────┐
        │ 파일 경로 → PIL 로드 → 전처리 → predict()      │
        │          → (이상 스코어, 이상 히트맵) 반환       │
        └────────────────────────────────────────────────┘

        Args:
            image_path: 이미지 파일 경로

        Returns:
            scores: (1,) 이미지 레벨 이상 스코어
            anomaly_maps: (1, H, W) 픽셀별 이상 히트맵
        """
        from patchcore_data import input_tensor
        # Ellipsis uses the saved region; explicit None selects the whole image.
        from center_crop import validate_center_crop
        crop = self.center_crop if center_crop is ... else validate_center_crop(center_crop)
        tensor = input_tensor(image_path, self.input_size, self.preprocessing, crop).unsqueeze(0)
        return self.predict(tensor)

    def get_info(self) -> dict:
        """모델 정보 반환"""
        return {
            "type": "PatchCore",
            "center_crop": self.center_crop,
            "score_normalization": self.get_score_normalization(),
            "normalized_threshold": self.normalized_threshold,
            "weight_source": self.weight_source,
            "preprocessing": self.preprocessing,
            "max_candidates": self.max_candidates, "max_memory_bank": self.max_memory_bank,
            "backbone": self.backbone_name,
            "feature_dim": self.backbone.feature_dim,
            "memory_bank_size": (
                self.memory_bank.shape[0] if self.memory_bank is not None else 0
            ),
            "n_neighbors": self.n_neighbors,
            "sampling_ratio": self.sampling_ratio,
            "input_size": self.input_size,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  이상 히트맵 시각화 유틸리티
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def anomaly_map_to_heatmap(
    anomaly_map: np.ndarray,
    original_image: np.ndarray,
    alpha: float = 0.5,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    이상 맵 → JET 히트맵 + 원본 오버레이

    시각화 파이프라인:
    ┌────────────┐   정규화    ┌──────────┐   JET    ┌──────────┐
    │ anomaly_map│ ────────── │ [0, 1]   │ ──────► │ heatmap  │
    └────────────┘             └──────────┘          └─────┬────┘
                                                          │
    ┌────────────┐                                   ┌────▼─────┐
    │ original   │ ──────── α-블렌딩 ───────────────►│ overlay  │
    └────────────┘                                   └──────────┘

    Args:
        anomaly_map: (H, W) 이상 맵
        original_image: (H, W, 3) RGB uint8 원본
        alpha: 히트맵 투명도
        normalize: 맵을 [0, 1]로 정규화할지

    Returns:
        heatmap: (H, W, 3) JET 히트맵 uint8
        overlay: (H, W, 3) 원본 + 히트맵 블렌딩 uint8
    """
    if normalize:
        vmin, vmax = anomaly_map.min(), anomaly_map.max()
        if vmax - vmin > 1e-8:
            anomaly_map = (anomaly_map - vmin) / (vmax - vmin)
        else:
            anomaly_map = np.zeros_like(anomaly_map)

    anomaly_map = np.clip(anomaly_map, 0, 1)

    # 원본 이미지 크기에 맞게 리사이즈 (필요 시)
    h, w = original_image.shape[:2]
    if anomaly_map.shape != (h, w):
        map_t = torch.from_numpy(anomaly_map).float().unsqueeze(0).unsqueeze(0)
        map_t = F.interpolate(map_t, size=(h, w), mode="bilinear", align_corners=False)
        anomaly_map = map_t.squeeze().numpy()

    # JET 컬러맵 적용 (순수 numpy)
    v = anomaly_map
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0, 1)
    heatmap = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    # 알파 블렌딩
    overlay = (
        alpha * heatmap.astype(np.float32)
        + (1 - alpha) * original_image.astype(np.float32)
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return heatmap, overlay


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  테스트 코드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 60)
    print("PatchCore Anomaly Detection — 구조 검증")
    print("=" * 60)

    # 백본 구조 확인
    backbone = PatchCoreBackbone("wide_resnet50_2")
    dummy = torch.randn(2, 3, 224, 224)
    feat = backbone(dummy)
    print(f"  백본 출력:  {feat.shape}")
    print(f"  특징 차원:  {backbone.feature_dim}")

    # PatchCore 인스턴스 (메모리 뱅크 없이)
    pc = PatchCore(backbone_name="wide_resnet50_2", device="cpu")
    info = pc.get_info()
    print(f"  PatchCore 정보: {info}")

    # 코어셋 샘플링 테스트
    dummy_features = torch.randn(1000, 1536)
    coreset = greedy_coreset_sampling(dummy_features, sampling_ratio=0.1)
    print(f"  코어셋:    {dummy_features.shape[0]} → {coreset.shape[0]}")

    print("\n  PatchCore 구조 검증 완료")
