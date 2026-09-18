"""
CustomCSP 모델 정의
- CSPDarknet 기반 백본 (경량화)
- Classification Head: GAP → FC
- Segmentation Head: FPN + Upsample → Pixel-wise Conv

아키텍처 개요:
┌───────────────────────────────────────────────────────────────────┐
│                        CustomCSP Architecture                        │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input(3,224,224)                                                 │
│       │                                                           │
│  ┌────▼────────────────────────────────────────────────────────┐  │
│  │  Stem: Conv3x3(3→32, s=2) → BN → SiLU                     │  │
│  └────┬────────────────────────────────────────────────────────┘  │
│       │ (32, 112, 112)                                            │
│  ┌────▼────────────────────────────────────────────────────────┐  │
│  │  Stage1: Conv3x3(32→64, s=2) → CSPBlock×1                  │  │
│  └────┬────────────────────────────────────────────────────────┘  │
│       │ (64, 56, 56)  ─── P2 (Seg용)                             │
│  ┌────▼────────────────────────────────────────────────────────┐  │
│  │  Stage2: Conv3x3(64→128, s=2) → CSPBlock×2                 │  │
│  └────┬────────────────────────────────────────────────────────┘  │
│       │ (128, 28, 28) ─── P3 (Seg용)                             │
│  ┌────▼────────────────────────────────────────────────────────┐  │
│  │  Stage3: Conv3x3(128→256, s=2) → CSPBlock×3                │  │
│  └────┬────────────────────────────────────────────────────────┘  │
│       │ (256, 14, 14) ─── P4 (Seg용)                             │
│  ┌────▼────────────────────────────────────────────────────────┐  │
│  │  Stage4: Conv3x3(256→512, s=2) → CSPBlock×2 → SPPF        │  │
│  └────┬────────────────────────────────────────────────────────┘  │
│       │ (512, 7, 7)   ─── P5                                     │
│       │                                                           │
│  ┌────▼─────────┐        ┌──────▼──────────────────────────┐     │
│  │ Classification│        │ Segmentation Head               │     │
│  │ GAP → FC(N)  │        │ FPN(P3,P4,P5) → Upsample → Conv│     │
│  └──────────────┘        └─────────────────────────────────┘     │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  기본 빌딩 블록
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ConvBNSiLU(nn.Module):
    """
    Conv2d → BatchNorm → SiLU 기본 블록
    자체 CSP 모델의 기본 컨볼루션 단위

    Parameters:
        in_ch: 입력 채널 수
        out_ch: 출력 채널 수
        kernel: 커널 크기
        stride: 스트라이드
        groups: 그룹 컨볼루션 수
    """
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, groups: int = 1):
        super().__init__()
        # 패딩 자동 계산: kernel=3이면 pad=1, kernel=1이면 pad=0
        padding = kernel // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel, stride, padding,
            groups=groups, bias=False  # BN 사용 시 bias 불필요
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)  # Swish 활성화

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """
    CSP Bottleneck 블록
    1×1 Conv로 채널 축소 → 3×3 Conv로 특징 추출 → 잔차 연결

    Structure:
        x ──► 1×1 Conv ──► 3×3 Conv ──┐
        │                               │
        └────── (shortcut) ────────────+──► output
    """
    def __init__(self, ch: int, shortcut: bool = True):
        super().__init__()
        hidden = ch // 2  # 채널 축소 비율 50%
        self.cv1 = ConvBNSiLU(ch, hidden, 1)   # 1×1: 채널 축소
        self.cv2 = ConvBNSiLU(hidden, ch, 3)    # 3×3: 특징 추출
        self.shortcut = shortcut  # 잔차 연결 활성화 여부

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        # 잔차 연결: 그래디언트 흐름 개선
        return x + out if self.shortcut else out


class CSPBlock(nn.Module):
    """
    Cross Stage Partial (CSP) 블록
    특징맵을 두 경로로 분할하여 처리 후 합침 → 연산량 절감

    Structure:
        x ────────┬──────────────────────────┐
                  │                          │
            ┌─────▼─────┐            ┌───────▼───────┐
            │ 1×1 Conv  │            │   1×1 Conv    │
            │ (split 1) │            │   (split 2)   │
            └─────┬─────┘            └───────┬───────┘
                  │                          │
            ┌─────▼─────┐                   │
            │ Bottleneck│×n                 │
            │ (연속적용) │                   │
            └─────┬─────┘                   │
                  │                          │
                  └──────────┬───────────────┘
                             │ Concat
                       ┌─────▼─────┐
                       │  1×1 Conv │ (채널 정리)
                       └───────────┘

    Parameters:
        in_ch: 입력 채널
        out_ch: 출력 채널
        n: Bottleneck 반복 횟수
    """
    def __init__(self, in_ch: int, out_ch: int, n: int = 1):
        super().__init__()
        hidden = out_ch // 2  # 분할 후 각 경로의 채널 수

        # 경로 1: Bottleneck 연속 적용
        self.cv1 = ConvBNSiLU(in_ch, hidden, 1)
        self.blocks = nn.Sequential(
            *[Bottleneck(hidden) for _ in range(n)]
        )

        # 경로 2: 직접 연결 (shortcut)
        self.cv2 = ConvBNSiLU(in_ch, hidden, 1)

        # 합침 후 채널 정리
        self.cv3 = ConvBNSiLU(hidden * 2, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 두 경로 처리 후 채널 방향 결합
        path1 = self.blocks(self.cv1(x))
        path2 = self.cv2(x)
        return self.cv3(torch.cat([path1, path2], dim=1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF)
    다양한 수용 영역의 특징을 효율적으로 결합

    기존 SPP 대비 3× 빠른 속도 (직렬 맥스풀링)

    Structure:
        x → 1×1 Conv → MaxPool → MaxPool → MaxPool
                  │          │          │          │
                  └──────────┴──────────┴──────────┘
                                  Concat
                              1×1 Conv → output
    """
    def __init__(self, in_ch: int, out_ch: int, pool_size: int = 5):
        super().__init__()
        hidden = in_ch // 2
        self.cv1 = ConvBNSiLU(in_ch, hidden, 1)
        # 직렬 맥스풀링: k=5 한 번 ≈ k=5, 두 번 ≈ k=9, 세 번 ≈ k=13
        self.pool = nn.MaxPool2d(pool_size, stride=1, padding=pool_size // 2)
        # concat 후 4배 채널 → 출력 채널로 축소
        self.cv2 = ConvBNSiLU(hidden * 4, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.cv1(x)
        y2 = self.pool(y1)   # 수용영역 5×5
        y3 = self.pool(y2)   # 수용영역 ~9×9
        y4 = self.pool(y3)   # 수용영역 ~13×13
        return self.cv2(torch.cat([y1, y2, y3, y4], dim=1))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CustomCSP 백본
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CustomCSPBackbone(nn.Module):
    """
    CustomCSP 백본 네트워크 (CSPDarknet 변형)
    - 26개 주요 컨볼루션 레이어로 구성
    - 각 스테이지에서 다중 스케일 특징맵 추출

    Parameters:
        in_channels: 입력 채널 (1: 그레이, 3: RGB)
        channels: 각 스테이지 채널 수 [32, 64, 128, 256, 512]
        depths: 각 스테이지 CSP 블록 깊이 [1, 2, 3, 2]
    """
    def __init__(self, in_channels: int = 3,
                 channels: List[int] = None,
                 depths: List[int] = None):
        super().__init__()
        # 기본값 설정
        if channels is None:
            channels = [32, 64, 128, 256, 512]
        if depths is None:
            depths = [1, 2, 3, 2]

        # Stem: 입력 → 초기 특징 추출
        self.stem = ConvBNSiLU(in_channels, channels[0], 3, stride=2)

        # Stage 1: 해상도 1/4 (stride=2 다운샘플링)
        self.stage1 = nn.Sequential(
            ConvBNSiLU(channels[0], channels[1], 3, stride=2),
            CSPBlock(channels[1], channels[1], n=depths[0])
        )

        # Stage 2: 해상도 1/8 → P3 특징맵 (세그멘테이션용)
        self.stage2 = nn.Sequential(
            ConvBNSiLU(channels[1], channels[2], 3, stride=2),
            CSPBlock(channels[2], channels[2], n=depths[1])
        )

        # Stage 3: 해상도 1/16 → P4 특징맵
        self.stage3 = nn.Sequential(
            ConvBNSiLU(channels[2], channels[3], 3, stride=2),
            CSPBlock(channels[3], channels[3], n=depths[2])
        )

        # Stage 4: 해상도 1/32 → P5 특징맵 + SPPF
        self.stage4 = nn.Sequential(
            ConvBNSiLU(channels[3], channels[4], 3, stride=2),
            CSPBlock(channels[4], channels[4], n=depths[3]),
            SPPF(channels[4], channels[4])
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        다중 스케일 특징맵 반환
        Returns: [P2(1/4), P3(1/8), P4(1/16), P5(1/32)]
        """
        x = self.stem(x)      # (B, 32, H/2, W/2)
        p2 = self.stage1(x)   # (B, 64,  H/4,  W/4)
        p3 = self.stage2(p2)  # (B, 128, H/8,  W/8)
        p4 = self.stage3(p3)  # (B, 256, H/16, W/16)
        p5 = self.stage4(p4)  # (B, 512, H/32, W/32)
        return [p2, p3, p4, p5]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Classification Head
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ClassificationHead(nn.Module):
    """
    분류 헤드: 백본의 최종 특징맵(P5) → 클래스 확률

    Structure:
        P5 (B, 512, 7, 7)
            │
        ┌───▼────────────────┐
        │ Global Avg Pool    │  → (B, 512, 1, 1)
        ├────────────────────┤
        │ Flatten            │  → (B, 512)
        ├────────────────────┤
        │ Dropout(p)         │
        ├────────────────────┤
        │ FC(512 → num_cls)  │  → (B, num_classes)
        └────────────────────┘
    """
    def __init__(self, in_channels: int = 512,
                 num_classes: int = 10,
                 dropout: float = 0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # 글로벌 평균 풀링
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 백본 출력 [P2, P3, P4, P5]
        Returns:
            logits: (B, num_classes) 클래스 점수
        """
        # P5 (최종 스테이지)만 사용
        x = features[-1]
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Segmentation Head (FPN 기반)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SegmentationHead(nn.Module):
    """
    세그멘테이션 헤드: FPN(Feature Pyramid Network) 기반 디코더

    Structure:
        P5 (512, H/32)          P4 (256, H/16)          P3 (128, H/8)
            │                       │                       │
        ┌───▼───┐                   │                       │
        │1×1 Conv│ →128             │                       │
        │Upsample│ ×2               │                       │
        └───┬───┘                   │                       │
            │                  ┌────▼────┐                  │
            └── Concat ──────►│1×1 Conv │→128              │
                              │Upsample │ ×2               │
                              └────┬────┘                   │
                                   │                   ┌────▼────┐
                                   └── Concat ────────►│1×1 Conv │→128
                                                       └────┬────┘
                                                            │
                                                       ┌────▼────┐
                                                       │3×3 Conv │→64
                                                       │Upsample │ ×8
                                                       │1×1 Conv │→num_cls
                                                       └─────────┘
                                                            │
                                                    Output (num_cls, H, W)
    """
    def __init__(self, backbone_channels: List[int] = None,
                 num_classes: int = 2,
                 fpn_channels: int = 128):
        super().__init__()
        if backbone_channels is None:
            backbone_channels = [64, 128, 256, 512]  # P2, P3, P4, P5

        # P5 → FPN 채널로 축소
        self.lateral5 = ConvBNSiLU(backbone_channels[3], fpn_channels, 1)
        # P4 → FPN 채널로 축소
        self.lateral4 = ConvBNSiLU(backbone_channels[2], fpn_channels, 1)
        # P3 → FPN 채널로 축소
        self.lateral3 = ConvBNSiLU(backbone_channels[1], fpn_channels, 1)

        # FPN 합침 후 정제 컨볼루션
        self.smooth4 = ConvBNSiLU(fpn_channels * 2, fpn_channels, 3)
        self.smooth3 = ConvBNSiLU(fpn_channels * 2, fpn_channels, 3)

        # 최종 디코더: 세그멘테이션 맵 생성
        self.decoder = nn.Sequential(
            ConvBNSiLU(fpn_channels, 64, 3),
            ConvBNSiLU(64, 64, 3),
            nn.Conv2d(64, num_classes, 1)  # 1×1 Conv로 클래스별 맵
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: 백본 출력 [P2, P3, P4, P5]
        Returns:
            seg_map: (B, num_classes, H, W) 원본 해상도 세그멘테이션 맵
        """
        _, p3, p4, p5 = features
        input_size = features[0].shape[2:]  # P2 기준 해상도

        # Top-down 경로: P5 → P4 → P3
        f5 = self.lateral5(p5)  # (B, 128, H/32, W/32)

        # P5 업샘플 + P4 결합
        f5_up = F.interpolate(f5, size=p4.shape[2:], mode='bilinear',
                              align_corners=False)
        f4 = self.smooth4(torch.cat([f5_up, self.lateral4(p4)], dim=1))

        # P4 업샘플 + P3 결합
        f4_up = F.interpolate(f4, size=p3.shape[2:], mode='bilinear',
                              align_corners=False)
        f3 = self.smooth3(torch.cat([f4_up, self.lateral3(p3)], dim=1))

        # 최종 업샘플: P3 해상도 → 원본의 1/4 → 원본
        # P3은 원본의 1/8이므로, 원본 크기로 4× 업샘플 (P2 기준 2×)
        out = F.interpolate(f3, size=(input_size[0] * 4, input_size[1] * 4),
                            mode='bilinear', align_corners=False)
        out = self.decoder(out)
        return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Detection Head (앵커 프리)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DetectionHead(nn.Module):
    """
    앵커 프리 객체 탐지 헤드 (FCOS 스타일)
    P3, P4, P5 다중 스케일에서 탐지 수행

    각 그리드 셀에서 예측:
    ┌──────────────────────────────────────────┐
    │ bbox  (4) : x_center, y_center, w, h    │
    │ obj   (1) : 객체 존재 확률 (sigmoid)      │
    │ cls   (C) : 클래스별 확률 (sigmoid)       │
    └──────────────────────────────────────────┘

    출력 텐서: (B, N, 5+C)  N = 모든 스케일 그리드 셀 합
    """
    def __init__(self, backbone_channels: List[int] = None,
                 num_classes: int = 20, fpn_channels: int = 128,
                 box_encoding: str = "grid_sigmoid_xywh"):
        super().__init__()
        if backbone_channels is None:
            backbone_channels = [64, 128, 256, 512]

        self.num_classes = num_classes
        if box_encoding not in ("grid_sigmoid_xywh", "legacy_raw"):
            raise ValueError("미지원 검출 좌표 형식")
        self.box_encoding = box_encoding

        # ── FPN lateral connections ──
        self.lateral5 = ConvBNSiLU(backbone_channels[3], fpn_channels, 1)
        self.lateral4 = ConvBNSiLU(backbone_channels[2], fpn_channels, 1)
        self.lateral3 = ConvBNSiLU(backbone_channels[1], fpn_channels, 1)

        # ── FPN smoothing ──
        self.smooth4 = ConvBNSiLU(fpn_channels * 2, fpn_channels, 3)
        self.smooth3 = ConvBNSiLU(fpn_channels * 2, fpn_channels, 3)

        # ── 공유 탐지 컨볼루션 ──
        self.det_convs = nn.Sequential(
            ConvBNSiLU(fpn_channels, fpn_channels, 3),
            ConvBNSiLU(fpn_channels, fpn_channels, 3),
        )

        # ── 예측 헤드 (각 스케일 공유) ──
        self.cls_pred = nn.Conv2d(fpn_channels, num_classes, 1)
        self.bbox_pred = nn.Conv2d(fpn_channels, 4, 1)
        self.obj_pred = nn.Conv2d(fpn_channels, 1, 1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: [P2, P3, P4, P5] 백본 출력
        Returns:
            preds: (B, N, 5+num_classes) 전체 스케일 예측
                   N = H3×W3 + H4×W4 + H5×W5
        """
        _, p3, p4, p5 = features

        # FPN top-down 경로
        f5 = self.lateral5(p5)
        f5_up = F.interpolate(f5, size=p4.shape[2:], mode='bilinear',
                              align_corners=False)
        f4 = self.smooth4(torch.cat([f5_up, self.lateral4(p4)], dim=1))
        f4_up = F.interpolate(f4, size=p3.shape[2:], mode='bilinear',
                              align_corners=False)
        f3 = self.smooth3(torch.cat([f4_up, self.lateral3(p3)], dim=1))

        # 각 스케일에서 탐지
        outputs = []
        for feat in [f3, f4, f5]:
            det = self.det_convs(feat)
            B, _, H, W = det.shape

            cls = self.cls_pred(det)    # (B, C, H, W)
            bbox = self.bbox_pred(det)  # (B, 4, H, W)
            obj = self.obj_pred(det)    # (B, 1, H, W)

            # (B, H*W, 5+C) 형태로 변환
            cls = cls.permute(0, 2, 3, 1).reshape(B, H * W, self.num_classes)
            bbox = bbox.permute(0, 2, 3, 1).reshape(B, H * W, 4)
            obj = obj.permute(0, 2, 3, 1).reshape(B, H * W, 1)

            if self.box_encoding == "grid_sigmoid_xywh":
                # 셀 위치를 포함한 정규화 중심과 양수 크기로 복원한다.
                yy, xx = torch.meshgrid(torch.arange(H, device=det.device),
                                        torch.arange(W, device=det.device), indexing="ij")
                cx = (bbox[..., 0].sigmoid() + xx.reshape(1, -1)) / W
                cy = (bbox[..., 1].sigmoid() + yy.reshape(1, -1)) / H
                bbox = torch.stack((cx, cy, bbox[..., 2].sigmoid(), bbox[..., 3].sigmoid()), dim=-1)

            pred = torch.cat([bbox, obj, cls], dim=-1)
            outputs.append(pred)

        return torch.cat(outputs, dim=1)  # (B, N, 5+C)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Anomaly Detection Head (재구성 기반)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AnomalyHead(nn.Module):
    """
    이상 탐지 헤드 — 재구성(Reconstruction) 기반

    원리:
    ┌─────────────────────────────────────────────────────┐
    │ 1. 정상 이미지로만 학습                               │
    │ 2. 백본 특징 → 디코더 → 원본 이미지 재구성            │
    │ 3. 재구성 오차 = 이상 영역 (정상: 낮음, 이상: 높음)    │
    └─────────────────────────────────────────────────────┘

    Structure:
        P5 ──► Upsample ──┐
        P4 ─── Skip ───── Concat ──► Conv ──► Upsample ──┐
        P3 ─── Skip ──────────────────────── Concat ──► Conv
        P2 ─── Skip ─────────────────────────────────── Concat ──► Reconstruct
    """
    def __init__(self, backbone_channels: List[int] = None,
                 in_channels: int = 3, fpn_channels: int = 128):
        super().__init__()
        if backbone_channels is None:
            backbone_channels = [64, 128, 256, 512]

        # 디코더 경로: P5 → 업샘플 → P4 스킵 → 업샘플 → P3 스킵 → ...
        self.up5 = ConvBNSiLU(backbone_channels[3], fpn_channels, 1)
        self.dec4 = nn.Sequential(
            ConvBNSiLU(fpn_channels + backbone_channels[2], fpn_channels, 3),
            ConvBNSiLU(fpn_channels, fpn_channels, 3),
        )
        self.dec3 = nn.Sequential(
            ConvBNSiLU(fpn_channels + backbone_channels[1], fpn_channels // 2, 3),
            ConvBNSiLU(fpn_channels // 2, fpn_channels // 2, 3),
        )
        self.dec2 = nn.Sequential(
            ConvBNSiLU(fpn_channels // 2 + backbone_channels[0], 64, 3),
            ConvBNSiLU(64, 32, 3),
        )

        # 최종 재구성 레이어 (활성화 없음 → 정규화된 입력 범위 매칭)
        self.reconstruct = nn.Sequential(
            nn.Conv2d(32, in_channels, 3, padding=1),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: [P2, P3, P4, P5] 백본 출력
        Returns:
            reconstructed: (B, C, H, W) 재구성된 이미지
                          학습 시 → MSE(input, reconstructed)
                          추론 시 → anomaly_map = |input - reconstructed|²
        """
        p2, p3, p4, p5 = features

        # P5 디코딩
        d5 = self.up5(p5)

        # P4 스킵 연결 + 업샘플
        d5_up = F.interpolate(d5, size=p4.shape[2:], mode='bilinear',
                              align_corners=False)
        d4 = self.dec4(torch.cat([d5_up, p4], dim=1))

        # P3 스킵 연결 + 업샘플
        d4_up = F.interpolate(d4, size=p3.shape[2:], mode='bilinear',
                              align_corners=False)
        d3 = self.dec3(torch.cat([d4_up, p3], dim=1))

        # P2 스킵 연결 + 업샘플
        d3_up = F.interpolate(d3, size=p2.shape[2:], mode='bilinear',
                              align_corners=False)
        d2 = self.dec2(torch.cat([d3_up, p2], dim=1))

        # 최종 업샘플 (P2는 입력의 1/4) → 원본 해상도
        d2_up = F.interpolate(d2, scale_factor=4, mode='bilinear',
                              align_corners=False)
        return self.reconstruct(d2_up)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CustomCSP 통합 모델
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CustomCSP(nn.Module):
    """
    CustomCSP 통합 모델 - 4가지 비전 태스크 지원

    지원 태스크:
    ┌──────────────┬───────────────────────────────────────┐
    │ classify     │ 이미지 분류 (B, num_classes)           │
    │ segment      │ 시맨틱 분할 (B, num_classes, H, W)     │
    │ detect       │ 객체 탐지   (B, N, 5+num_classes)      │
    │ anomaly      │ 이상 탐지   (B, C, H, W) 재구성 이미지 │
    └──────────────┴───────────────────────────────────────┘

    사용법:
        model = CustomCSP(task="classify", num_classes=10)
        model = CustomCSP(task="segment",  num_classes=3)
        model = CustomCSP(task="detect",   num_classes=20)
        model = CustomCSP(task="anomaly")  # 정상 이미지 재구성
    """
    SUPPORTED_TASKS = ("classify", "segment", "detect", "anomaly")

    def __init__(self, task: str = "classify",
                 in_channels: int = 3,
                 num_classes: int = 10,
                 backbone_channels: List[int] = None,
                 csp_depth: List[int] = None,
                 dropout: float = 0.2,
                 detection_box_encoding: str = "grid_sigmoid_xywh"):
        super().__init__()

        if backbone_channels is None:
            backbone_channels = [32, 64, 128, 256, 512]
        if csp_depth is None:
            csp_depth = [1, 2, 3, 2]

        self.task = task
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.detection_box_encoding = detection_box_encoding

        # 공유 백본 (모든 태스크가 동일한 백본 사용)
        self.backbone = CustomCSPBackbone(
            in_channels=in_channels,
            channels=backbone_channels,
            depths=csp_depth
        )

        # ── 태스크별 헤드 선택 ──
        if task == "classify":
            self.head = ClassificationHead(
                in_channels=backbone_channels[-1],
                num_classes=num_classes,
                dropout=dropout
            )
        elif task == "segment":
            self.head = SegmentationHead(
                backbone_channels=backbone_channels[1:],
                num_classes=num_classes,
                fpn_channels=128
            )
        elif task == "detect":
            self.head = DetectionHead(
                backbone_channels=backbone_channels[1:],
                num_classes=num_classes,
                fpn_channels=128,
                box_encoding=detection_box_encoding,
            )
        elif task == "anomaly":
            self.head = AnomalyHead(
                backbone_channels=backbone_channels[1:],
                in_channels=in_channels,
                fpn_channels=128
            )
        else:
            raise ValueError(
                f"지원하지 않는 task: {task}. "
                f"사용 가능: {self.SUPPORTED_TASKS}"
            )

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """Kaiming 초기화 (SiLU 활성화에 적합)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파

        Args:
            x: 입력 이미지 (B, C, H, W)
        Returns:
            classify: (B, num_classes) 클래스 로짓
            segment:  (B, num_classes, H, W) 세그멘테이션 맵
        """
        features = self.backbone(x)
        output = self.head(features)
        if self.task == "segment" and output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return output

    def get_param_count(self) -> dict:
        """모델 파라미터 수 요약"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "backbone": backbone_params,
            "head": head_params,
            "total_MB": total * 4 / 1024 / 1024  # float32 기준
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  테스트 코드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    tasks = [
        ("classify", 5,  224, "🔍 Classification"),
        ("segment",  3,  320, "🎨 Segmentation"),
        ("detect",   20, 416, "📦 Detection"),
        ("anomaly",  0,  224, "🔬 Anomaly Detection"),
    ]

    for task, nc, size, label in tasks:
        print("=" * 60)
        print(f"{label} 테스트")
        print("=" * 60)
        nc_arg = nc if nc > 0 else 3  # anomaly는 num_classes 미사용
        model = CustomCSP(task=task, num_classes=nc_arg, in_channels=3)
        dummy = torch.randn(1, 3, size, size)
        out = model(dummy)
        params = model.get_param_count()
        print(f"  입력  : {dummy.shape}")
        print(f"  출력  : {out.shape}")
        print(f"  파라미터: {params['total']:,} ({params['total_MB']:.1f} MB)")
        print()
