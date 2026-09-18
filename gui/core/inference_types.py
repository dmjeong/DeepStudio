"""화면과 계산 워커가 교환하는 불변 결과 값."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class InferenceResult:
    """한 이미지의 계산 결과. UI 텍스트를 다시 읽어 판정을 만들지 않는다."""
    image_path: str
    status: str
    task: str
    summary: str
    color: str = "#5590F0"
    score: float | None = None
    threshold: float | None = None
    error: str = ""
    elapsed_sec: float = 0.0
    details: dict | None = None
    inference_sec: float | None = None
    gradcam_sec: float | None = None
    inference_status: str = "not_run"
    gradcam_status: str = "not_run"
    source_class: str = ""


def make_anomaly_result(image_path, score, threshold):
    score = float(score)
    if not np.isfinite(score):
        raise ValueError("이상 점수가 유한하지 않습니다")
    if threshold is None:
        return InferenceResult(image_path, "uncalibrated", "anomaly",
                               f"미보정 {score:.4f}", "#E5A832", score)
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("이상 판정 임계값이 유한하지 않습니다")
    abnormal = score >= threshold
    return InferenceResult(image_path, "ok", "anomaly",
                           f"{'NG' if abnormal else 'OK'} {score:.4f}",
                           "#E05555" if abnormal else "#34C759", score, threshold)

