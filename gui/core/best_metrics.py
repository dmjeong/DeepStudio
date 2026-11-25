"""Display the metrics saved for one best checkpoint, never the latest epoch."""
import math
from numbers import Real


NON_METRICS = {"epoch", "best_epoch", "is_best", "learning_rate", "lr",
               "selected_value", "selection_value", "sample_count", "num_classes",
               "num_samples", "threshold", "optimal_threshold", "anomaly_threshold", "total_seconds"}

_DETECTION_ALIASES = {
    "mAP_50": ("metrics/mAP50", "metrics/mAP50(B)", "mAP50"),
    "mAP_50_95": ("metrics/mAP50-95", "metrics/mAP50-95(B)", "mAP50_95"),
    "AR100": ("metrics/AR100", "metrics/ar_100", "metrics/recall", "metrics/recall(B)"),
}
_DETECTION_LABELS = {
    "mAP_50": "mAP@0.50",
    "mAP_50_95": "mAP@0.50–0.95",
    "AR100": "AR@100",
    "precision": "Precision",
    "recall": "Recall",
    "detection_tp": "정탐 박스 (TP)",
    "detection_fp": "오검출 박스 (FP)",
    "detection_fn": "누락 박스 (FN)",
    "detection_support": "정답 박스 수",
}


def scalar_metrics(values):
    return {key: (float(value) if value is not None and math.isfinite(value) else None)
            for key, value in values.items()
            if key not in NON_METRICS and not key.endswith(("_sec", "_seconds", "_hms"))
            and (value is None or isinstance(value, Real) and not isinstance(value, bool))}


def display_metrics(values, task=None):
    """Translate native metric keys at the UI boundary, including older runs."""
    metrics = scalar_metrics(values)
    if task != "detect":
        return metrics
    aliases_present = any(key in metrics for key in ("metrics/precision", "metrics/precision(B)",
                                                      "metrics/recall", "metrics/recall(B)"))
    for key, aliases in _DETECTION_ALIASES.items():
        if key not in metrics:
            for alias in aliases:
                if alias in metrics:
                    metrics[key] = metrics[alias]
                    break
    for key in ("mAP_50", "mAP_50_95", "AR100"):
        if key in metrics and metrics[key] is not None and metrics[key] < 0:
            metrics[key] = None  # COCO reports -1 for a missing evaluation slice.
    if aliases_present:
        metrics.setdefault("precision", None)
        metrics.setdefault("recall", None)
    return {key: metrics[key] for key in (*_DETECTION_LABELS, "train_loss", "val_loss")
            if key in metrics}


def best_epoch_metrics(run, task=None):
    if run.best_epoch < 1:
        return {}
    config = run.config_snapshot or {}
    saved = config.get("best_selection", {})
    task = task or (run.eval_results or {}).get("task") or config.get("task") or saved.get("task")
    values = scalar_metrics(run.eval_results or {})
    history = run.metrics_history or {}
    epochs = history.get("epoch", list(range(1, max((len(v) for v in history.values()), default=0) + 1)))
    if "epoch" not in history and config.get("training", {}).get("training_mode") == "upstream_resume":
        epochs = []  # Legacy upstream records omitted the resume offset. Do not guess.
    if run.best_epoch in epochs:
        index = epochs.index(run.best_epoch)
        values.update(scalar_metrics({key: sequence[index] for key, sequence in history.items()
                                      if index < len(sequence)}))
    metric_name = run.best_metric_name
    legacy_detection = task == "detect" and metric_name in _DETECTION_ALIASES["mAP_50_95"]
    map_keys = ("mAP_50_95", *_DETECTION_ALIASES["mAP_50_95"])
    if saved.get("epoch") == run.best_epoch:
        saved_values = scalar_metrics(saved.get("metrics", {}))
        if legacy_detection:
            # The old worker wrote the last epoch's selection score into this
            # saved Best field. Prefer the actual Best epoch's validation data.
            for key in map_keys:
                saved_values.pop(key, None)
        values.update(saved_values)
    if legacy_detection:
        result = config.get("upstream_result", {})
        score = next((values[key] for key in map_keys if values.get(key) is not None and 0 <= values[key] <= 1), None)
        if score is None and result.get("best_epoch") == run.best_epoch:
            score = scalar_metrics({"mAP_50_95": result.get("best_mAP50_95")}).get("mAP_50_95")
        values["mAP_50_95"] = score if score is not None and 0 <= score <= 1 else None
    elif metric_name and metric_name != "unavailable":
        values.update(scalar_metrics({metric_name: run.best_metric}))
    return display_metrics(values, task)


def selection_value_for_display(run, task):
    if task == "detect" and run.best_metric_name in _DETECTION_ALIASES["mAP_50_95"]:
        return best_epoch_metrics(run, task).get("mAP_50_95")
    return run.best_metric


def saved_detection_evaluation(run):
    details = (run.eval_results or {}).get("detection_evaluation")
    if details:
        return details
    config = run.config_snapshot or {}
    mode = str(config.get("training", {}).get("training_mode", ""))
    if config.get("engine") == "upstream" or mode.startswith("upstream_"):
        training = config.get("training", {})
        return {"unit": "patch" if training.get("patch_training") else "image"
                if "patch_training" in training else None, "protocol": "coco"}
    return {}


def detection_evaluation_text(details):
    unit = details.get("unit")
    area = {"patch": "패치별", "image": "이미지별"}.get(unit, "검증 단위 미기록")
    text = f"검증 데이터(Val) · {area} 평가"
    protocol = {"coco": "COCO AP", "voc11": "VOC 11점 AP"}.get(details.get("protocol"))
    if protocol:
        text += f" · {protocol}"
    confidence, iou = details.get("confidence"), details.get("iou")
    if all(isinstance(value, Real) and not isinstance(value, bool) for value in (confidence, iou)):
        text += f" · Precision/Recall 기준: confidence ≥ {confidence:.2f}, IoU ≥ {iou:.2f}"
    if unit == "patch":
        text += " · 추론 화면에서는 패치 결과를 원본 영상에 합칩니다."
    return text


def metric_label(key, task="classify"):
    from core.metrics import TASK_METRIC_NAMES
    labels = {"train_loss": "Train Loss", "val_loss": "Val Loss", "prompt_dice": "Prompt Dice",
              "prompt_iou": "Prompt IoU", "accuracy_top5": "Top-5 Accuracy", "recon_loss": "Reconstruction Loss"}
    labels.update(TASK_METRIC_NAMES.get(task, {}).get("labels", {}))
    if task == "detect":
        for canonical, aliases in _DETECTION_ALIASES.items():
            labels.update(dict.fromkeys(aliases, _DETECTION_LABELS[canonical]))
        labels.update(_DETECTION_LABELS)
    return labels.get(key, key)
