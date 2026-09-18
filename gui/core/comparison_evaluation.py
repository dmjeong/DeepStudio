"""공식 모델과 Custom CSP의 예측을 같은 평가기에 전달한다."""

import numpy as np
from core.metrics import ClassificationMetrics, DetectionMetrics

EVALUATOR_VERSION = "studio-comparison-101-v2"


def evaluate_predictions(task, class_names, records):
    if not records or not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("중복 없는 클래스 목록과 평가 이미지 필요")
    if task not in {"classify", "detect"}:
        raise ValueError("공통 비교는 분류 또는 박스 검출 지원: 의미론적 분할과 인스턴스 분할은 별도 평가 필요")
    ids = [record["image_id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("평가 이미지 ID 중복")
    meter = (ClassificationMetrics(len(class_names), class_names) if task == "classify"
             else DetectionMetrics(len(class_names), class_names, ap_method="interp101"))
    for record in records:
        if task == "classify":
            meter.update(np.asarray([record["prediction"]]), np.asarray([record["target"]]))
        else:
            for box in record["prediction"] + record["target"]:
                xyxy = np.asarray(box["bbox"], dtype=float)
                if (xyxy.shape != (4,) or not np.isfinite(xyxy).all()
                        or np.any(xyxy < 0) or np.any(xyxy > 1)
                        or xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]):
                    raise ValueError("검출 비교 좌표는 0~1 정규화 xyxy 필요")
            meter.update(record["prediction"], record["target"])
    metrics = meter.compute()
    return {"evaluator": EVALUATOR_VERSION, "task": task, "class_names": class_names,
            "images": len(records), "metrics": metrics,
            "ap_method": "101-point precision envelope integration" if task == "detect" else None,
            "limitations": "COCO crowd/area/maxDets 평가 전체를 구현한 지표는 아님. 학습 중 엔진 지표와 별도."}
