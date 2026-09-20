"""UI와 학습 엔진이 공유하는 Best 선정 규칙. 검증 데이터만 사용한다."""

from dataclasses import dataclass
import math


LABELS = {
    "val_loss": "검증 손실 (Val Loss)",
    "engine_default": "엔진 기본 기준",
    "accuracy": "정확도 (Top-1)",
    "f1_macro": "Macro F1 (클래스별 F1 평균)",
    "recall_macro": "Macro Recall (클래스별 검출률 평균)",
    "mAP_50": "박스 mAP50",
    "mAP_50_95": "박스 mAP50-95",
    "mIoU": "mIoU (클래스별 영역 일치도)",
    "dice_score": "Dice (영역 겹침 점수)",
    "recon_loss": "재구성 손실",
}


def available_metrics(engine, task):
    if task == "classify":
        return ["engine_default", "accuracy", "f1_macro", "recall_macro", "val_loss"]
    if task in ("detect", "obb"):
        return ["engine_default", "mAP_50_95", "mAP_50", "val_loss"]
    if task == "segment":
        return ["engine_default", "mIoU", "dice_score", "val_loss"]
    return ["engine_default"]


@dataclass(frozen=True)
class SelectionPolicy:
    metric: str
    direction: str
    formula: str
    tie: str

    def value(self, metrics):
        value = metrics.get(self.metric)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Best 선정 지표 계산 불가: {self.metric}")
        return float(value)

    def describe(self):
        direction = "최소" if self.direction == "min" else "최대"
        return f"검증 {self.formula} {direction} | 동점: {self.tie} 에폭"


def selection_policy(config, engine, task):
    if task == "obb":
        raise ValueError("OBB 학습 엔진은 제공하지 않습니다")
    requested = getattr(config, "selection_metric", "engine_default")
    if requested not in available_metrics(engine, task):
        raise ValueError(f"이 모델에서 지원하지 않는 Best 기준: {requested}")
    metric = requested
    if requested == "engine_default":
        metric = ({"classify": "accuracy", "detect": "mAP_50",
                   "segment": "mIoU", "anomaly": "recon_loss"}[task])
    formula = LABELS[metric]
    return SelectionPolicy(metric, "min" if metric in {"recon_loss", "val_loss"} else "max",
                           formula, "최초")
