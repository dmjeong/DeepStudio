"""
커스텀 Qt 시그널 정의
- 학습 진행 상황, 에러, 완료 등의 이벤트를 GUI로 전달
- QThread와 메인 스레드 간 안전한 통신

시그널 흐름:
┌──────────────┐   signal   ┌──────────────┐
│ TrainThread  │ ─────────► │ GUI Widget   │
│ (워커 스레드)│            │ (메인 스레드)  │
└──────────────┘            └──────────────┘
"""

from PySide6.QtCore import QObject, Signal


class TrainingSignals(QObject):
    """
    학습 진행 시그널 — TrainWorker → GUI 통신

    시그널 흐름:
    ┌─────────────────────────────────────────────────────┐
    │  epoch_finished  →  Loss/Metric 차트 실시간 갱신     │
    │  batch_finished  →  프로그레스 바 업데이트             │
    │  lr_updated      →  LR 스케줄 차트 업데이트           │
    │  eval_finished   →  평가 결과 테이블 + Confusion Mat. │
    │  training_finished → 완료 알림 + 모델 비교            │
    └─────────────────────────────────────────────────────┘
    """

    # 에폭 완료: (epoch, train_loss, val_loss, metrics_dict)
    epoch_finished = Signal(int, float, float, dict)

    # Best 체크포인트 저장 완료: 해당 에폭의 손실과 지표를 함께 전달
    best_epoch_updated = Signal(int, float, float, dict)

    # 배치 완료: (epoch, batch_idx, total_batches, loss)
    batch_finished = Signal(int, int, int, float)

    # 학습 완료: (best_metric, best_epoch, checkpoint_path)
    training_finished = Signal(float, int, str)

    # 학습 에러: (에러 메시지)
    training_error = Signal(str)

    # 학습 로그: (로그 메시지)
    log_message = Signal(str)

    # 전체 진행률: (현재 에폭, 전체 에폭)
    progress_updated = Signal(int, int)

    # 조기 종료 발생
    early_stopped = Signal(int, float)  # (에폭, 최적 메트릭)

    # Learning Rate 업데이트: (epoch, lr)
    lr_updated = Signal(int, float)

    # 평가 완료: (eval_results_dict)
    # 학습 후 전체 검증/테스트 세트에 대한 종합 평가 결과
    eval_finished = Signal(dict)
    layer_debug = Signal(dict)


class InferenceSignals(QObject):
    """추론 시그널"""

    # 추론 완료: (이미지 경로, 결과 dict)
    inference_done = Signal(str, dict)

    # 배치 추론 진행: (완료 수, 전체 수)
    batch_progress = Signal(int, int)

    # 에러
    inference_error = Signal(str)


class DatasetSignals(QObject):
    """데이터셋 로딩 시그널"""

    # 로딩 진행: (완료 수, 전체 수, 메시지)
    loading_progress = Signal(int, int, str)

    # 로딩 완료: (이미지 수, 클래스 수)
    loading_finished = Signal(int, int)

    # 에러
    loading_error = Signal(str)
