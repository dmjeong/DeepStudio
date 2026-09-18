"""
CustomCSP 학습 설정 파일
- Classification / Segmentation 공통 하이퍼파라미터 관리
- CPU 환경 최적화 설정 포함
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import os


@dataclass
class CustomCSPConfig:
    """
    CustomCSP 모델 및 학습 전체 설정

    ┌─────────────────────────────────────────────┐
    │              Config 구조                     │
    ├─────────────────────────────────────────────┤
    │ Model Config  │ 백본 + 헤드 아키텍처 설정    │
    │ Train Config  │ 학습률, 에폭, 배치 크기       │
    │ Data Config   │ 이미지 경로, 크기, 증강        │
    │ Export Config │ ONNX 변환 설정                │
    └─────────────────────────────────────────────┘
    """

    # ── 태스크 설정 ──────────────────────────────
    # "classify" 또는 "segment"
    task: str = "classify"

    # ── 모델 아키텍처 설정 ──────────────────────────
    # 입력 이미지 크기 (H, W)
    input_size: Tuple[int, int] = (224, 224)
    # 입력 채널 수 (1: 그레이스케일, 3: RGB)
    in_channels: int = 3
    # 분류 클래스 수 (학습 데이터에서 자동 결정 가능)
    num_classes: int = 10

    # 백본 채널 설정 [Stem, Stage1, Stage2, Stage3, Stage4]
    backbone_channels: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256, 512]
    )
    # CSP 블록 내 반복 횟수
    csp_depth: List[int] = field(
        default_factory=lambda: [1, 2, 3, 2]
    )
    # Dropout 비율 (Classification Head)
    dropout_rate: float = 0.2

    # ── 학습 설정 ─────────────────────────────────
    # 에폭 수
    epochs: int = 100
    # 배치 크기 (CPU 환경: 작게 유지)
    batch_size: int = 8
    # 학습률
    learning_rate: float = 1e-3
    # 가중치 감쇠 (L2 정규화)
    weight_decay: float = 5e-4
    # 학습률 스케줄러 (cosine, step, none)
    scheduler: str = "cosine"
    # 워밍업 에폭
    warmup_epochs: int = 3
    # 조기 종료 patience
    early_stop_patience: int = 15

    # ── 데이터 설정 ─────────────────────────────────
    # 학습 데이터 루트 디렉토리
    data_root: str = "./data"
    # 학습/검증 분할 비율
    val_split: float = 0.2
    # 데이터 로더 워커 수 (CPU: 0 권장)
    num_workers: int = 0

    # ── 데이터 증강 설정 ──────────────────────────────
    # 수평 뒤집기 확률
    flip_prob: float = 0.5
    # 회전 각도 범위 (도)
    rotation_range: float = 15.0
    # 밝기/대비 변화 범위
    color_jitter: float = 0.2
    # Mixup 알파 (0이면 비활성화)
    mixup_alpha: float = 0.0

    # ── 저장/로깅 설정 ──────────────────────────────
    # 체크포인트 저장 경로
    save_dir: str = "./runs"
    # 실험 이름
    exp_name: str = "custom_csp_exp"
    # 로그 간격 (배치 단위)
    log_interval: int = 10
    # 최적 모델만 저장
    save_best_only: bool = True

    # ── ONNX 변환 설정 ──────────────────────────────
    # ONNX opset 버전
    onnx_opset: int = 18
    # 동적 배치 크기 지원 여부
    onnx_dynamic_batch: bool = False
    # ONNX 최적화 수준 (0: 비활성, 1: 기본, 2: 확장)
    onnx_optimize_level: int = 1
    # 내보낼 ONNX 파일명
    onnx_output: str = "custom_csp_model.onnx"

    @property
    def save_path(self) -> str:
        """실험 결과 저장 전체 경로"""
        return os.path.join(self.save_dir, self.exp_name)

    def validate(self):
        """설정값 유효성 검사"""
        assert self.task in ("classify", "segment", "detect", "anomaly"), \
            f"task는 'classify', 'segment', 'detect', 'anomaly'만 가능: {self.task}"
        assert self.num_classes > 0, \
            f"num_classes는 양수여야 함: {self.num_classes}"
        assert 0.0 < self.val_split < 1.0, \
            f"val_split은 (0, 1) 범위여야 함: {self.val_split}"
        assert self.batch_size > 0, \
            f"batch_size는 양수여야 함: {self.batch_size}"
        print("✅ 설정 검증 완료")


# ── 프리셋 설정들 ─────────────────────────────────
def get_classification_config(**overrides) -> CustomCSPConfig:
    """Classification 기본 프리셋"""
    settings = {
        "task": "classify",
        "input_size": (224, 224),
        "dropout_rate": 0.3,
    }
    settings.update(overrides)
    return CustomCSPConfig(**settings)


def get_segmentation_config(**overrides) -> CustomCSPConfig:
    """Segmentation 기본 프리셋 (더 큰 입력, 더 작은 배치)"""
    settings = {
        "task": "segment",
        "input_size": (320, 320),
        "batch_size": 4,
        "dropout_rate": 0.1,
    }
    settings.update(overrides)
    return CustomCSPConfig(**settings)
