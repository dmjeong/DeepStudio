"""Pinned native adapters for the shipped LibreYOLO model families.

The product must never train a catalog model with CustomCSP as a fallback.
This module is deliberately small: it maps a product model ID to the public,
versioned LibreYOLO API and writes the data contract that API consumes.  No
upstream source is copied into Deep Vision Studio; the installer resolves the
pinned wheel declared in ``requirements-upstream-models.txt`` and ships its
license notices separately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class UpstreamModelSpec:
    model_id: str
    family: str
    task: str
    size: str
    input_size: int
    pretrained_name: str


UPSTREAM_MODELS: dict[str, UpstreamModelSpec] = {
    "libreyolo_classify_mobilenetv4_small": UpstreamModelSpec(
        "libreyolo_classify_mobilenetv4_small", "mobilenetv4", "classify", "s", 224,
        "LibreMobileNetV4s-cls.pt"),
    "libreyolo_detect_9t": UpstreamModelSpec(
        "libreyolo_detect_9t", "yolo9", "detect", "t", 640, "LibreYOLO9t.pt"),
    "re_detr_v4_small": UpstreamModelSpec(
        "re_detr_v4_small", "rtdetrv4", "detect", "s", 640, "LibreRTDETRv4s.pt"),
    "re_detr_v4_medium": UpstreamModelSpec(
        "re_detr_v4_medium", "rtdetrv4", "detect", "m", 640, "LibreRTDETRv4m.pt"),
    "re_detr_v4_large": UpstreamModelSpec(
        "re_detr_v4_large", "rtdetrv4", "detect", "l", 640, "LibreRTDETRv4l.pt"),
}


def get_upstream_spec(model_id: str) -> UpstreamModelSpec:
    try:
        return UPSTREAM_MODELS[model_id]
    except KeyError as exc:
        raise ValueError(f"native upstream adapter is unavailable for {model_id}") from exc


def upstream_model_ids() -> frozenset[str]:
    return frozenset(UPSTREAM_MODELS)


def _api():
    """Import only when the selected model needs the optional native runtime."""
    try:
        from libreyolo import LibreMobileNetV4, LibreRTDETRv4, LibreYOLO9
    except ImportError as exc:
        raise RuntimeError(
            "기본 제공 모델 런타임이 없습니다. 설치본을 다시 설치하거나 "
            "python -m pip install -r gui/requirements-upstream-models.txt 를 실행하세요."
        ) from exc
    return LibreMobileNetV4, LibreYOLO9, LibreRTDETRv4


def build_upstream_model(model_id: str, *, num_classes: int, device: str,
                         weights: str | None = None):
    """Create the exact public upstream wrapper for training or export.

    ``weights=None`` means random initialization.  Passing a local path is a
    transfer/resume request; the caller owns validation that the path belongs
    to the same selected family.
    """
    spec = get_upstream_spec(model_id)
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    LibreMobileNetV4, LibreYOLO9, LibreRTDETRv4 = _api()
    if spec.family == "mobilenetv4":
        return LibreMobileNetV4(model_path=weights, size=spec.size,
                                nb_classes=num_classes, device=device)
    if spec.family == "yolo9":
        return LibreYOLO9(model_path=weights, size=spec.size,
                          nb_classes=num_classes, device=device, task="detect")
    if spec.family == "rtdetrv4":
        return LibreRTDETRv4(model_path=weights, size=spec.size,
                              nb_classes=num_classes, device=device, task="detect")
    raise AssertionError(f"unhandled upstream family: {spec.family}")


def write_detection_dataset_yaml(root: str | Path, class_names: list[str], output_path: str | Path) -> Path:
    """Write an offline YOLO dataset descriptor for the app's existing layout.

    Deep Vision Studio already writes ``images/{train,val}`` and
    ``labels/{train,val}`` in standard normalized YOLO format.  JSON is valid
    YAML, so this avoids a second parser dependency while preserving Windows
    paths and non-ASCII class labels correctly.
    """
    dataset_root = Path(root).resolve()
    if not class_names or any(not isinstance(name, str) or not name for name in class_names):
        raise ValueError("detection dataset requires one or more class names")
    train = dataset_root / "images" / "train"
    val = dataset_root / "images" / "val"
    if not train.is_dir():
        raise ValueError(f"검출 학습 이미지 폴더가 없습니다: {train}")
    descriptor = {
        "path": str(dataset_root), "train": "images/train",
        "val": "images/val" if val.is_dir() else "images/train",
        "nc": len(class_names), "names": list(class_names),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(descriptor, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


class _ProgressCallback:
    def __init__(self, emit: Callable[[dict], None]):
        self.emit = emit

    def on_train_epoch_end(self, event) -> None:
        metrics = dict(getattr(event, "val_metrics", {}) or {})
        metric_name = getattr(event, "current_metric_name", None) or "metric"
        metric = getattr(event, "current_metric", None)
        if metric is not None:
            metrics[metric_name] = float(metric)
        # LibreYOLO reports validation values with names such as
        # ``metrics/val_loss``.  Do not substitute the training loss here:
        # that made the graph look complete while hiding a broken validator.
        val_loss = next((metrics[key] for key in
                         ("metrics/val_loss", "val_loss", "metrics/loss")
                         if key in metrics), None)
        if val_loss is None:
            raise RuntimeError("upstream validator가 val_loss를 보고하지 않았습니다")
        if metric is None:
            raise RuntimeError("upstream validator가 선택 검증 지표를 보고하지 않았습니다")
        self.emit({"event": "epoch_finished", "epoch": int(event.epoch),
                   "total_epochs": int(event.total_epochs),
                   "train_loss": float(event.train_loss),
                   "val_loss": float(val_loss),
                   "metric": float(metric),
                   "metric_name": metric_name, "metrics": metrics,
                   "epoch_time_sec": float(event.epoch_seconds)})


def train_upstream_model(model_id: str, *, data_root: str | Path, class_names: list[str],
                         output_dir: str | Path, epochs: int, batch_size: int,
                         learning_rate: float, device: str, weights: str | None,
                         resume: bool, use_amp: bool, patience: int,
                         emit: Callable[[dict], None]) -> dict:
    """Run an upstream native trainer and return its documented result map."""
    spec = get_upstream_spec(model_id)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if resume and not weights:
        raise ValueError("학습 재개에는 같은 모델의 last.pt 또는 best.pt가 필요합니다")
    model = build_upstream_model(model_id, num_classes=len(class_names), device=device,
                                 weights=weights)
    source = str(data_root)
    if spec.task == "detect":
        source = str(write_detection_dataset_yaml(data_root, class_names, output / "dataset.yaml"))
    callback = _ProgressCallback(emit)
    return model.train(data=source, epochs=int(epochs), batch=int(batch_size), imgsz=spec.input_size,
                       lr0=float(learning_rate), device=device, project=str(output.parent),
                       name=output.name, exist_ok=True, resume=bool(resume), amp=bool(use_amp),
                       patience=int(patience), callbacks=callback, val_loss=True,
                       eval_interval=1)
