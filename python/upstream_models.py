"""Pinned native adapters for the shipped LibreYOLO model families.

The product must never train a catalog model with CustomCSP as a fallback.
This module is deliberately small: it maps a product model ID to the public,
versioned LibreYOLO API and writes the data contract that API consumes.  No
upstream source is copied into Deep Vision Studio; the installer resolves the
pinned wheel declared in ``requirements-upstream-models.txt`` and ships its
license notices separately.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


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
    "libreyolo_classify_mobilenetv4_medium": UpstreamModelSpec(
        "libreyolo_classify_mobilenetv4_medium", "mobilenetv4", "classify", "m", 224,
        "LibreMobileNetV4m-cls.pt"),
    "libreyolo_detect_dfine_nano": UpstreamModelSpec(
        "libreyolo_detect_dfine_nano", "dfine", "detect", "n", 640, "LibreDFINEn.pt"),
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


def is_upstream_checkpoint(checkpoint: object) -> bool:
    """Return whether a trusted checkpoint uses one of the shipped APIs."""
    return (isinstance(checkpoint, dict) and
            checkpoint.get("model_family") in {"mobilenetv4", "yolo9", "rtdetrv4", "dfine"} and
            checkpoint.get("task") in {"classify", "detect"})


def load_upstream_checkpoint(path: str | Path, *, device: str):
    """Load an app-produced LibreYOLO checkpoint through its public factory."""
    try:
        from libreyolo import LibreYOLO
    except ImportError as exc:
        raise RuntimeError(
            "기본 제공 모델 런타임이 없습니다. 설치본을 다시 설치하거나 "
            "python -m pip install -r gui/requirements-upstream-models.txt 를 실행하세요."
        ) from exc
    model = LibreYOLO(str(path), device=device)
    family = str(getattr(model, "FAMILY", getattr(model, "family", ""))).lower()
    if family not in {"mobilenetv4", "yolo9", "rtdetrv4", "dfine"}:
        raise ValueError(f"Deep Vision Studio 기본 LibreYOLO 모델이 아닙니다: {family or 'unknown'}")
    if family == "mobilenetv4":
        from libreyolo_classification import configure_classifier
        configure_classifier(model)
    else:
        from native_detection import configure_detector
        configure_detector(model)
        # The factory can return an inner module still in training mode.
        # Detection inference must use the saved BatchNorm statistics and
        # the evaluation branch of the decoder before the first frame.
        model.model.eval()
    import torch
    from native_patches import validate_patches
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model.studio_patches = validate_patches(saved.get("studio_patches"), saved.get("task"))
    return model


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
                         weights: str | None = None, pretrained: bool = False):
    """Create the exact public upstream wrapper for training or export.

    ``weights=None`` means random initialization.  Passing a local path is a
    transfer/resume request; the caller owns validation that the path belongs
    to the same selected family.
    """
    spec = get_upstream_spec(model_id)
    if pretrained:
        if weights is not None:
            raise ValueError("기본 제공 사전학습 가중치와 로컬 가중치를 동시에 사용할 수 없습니다")
        from builtin_assets import builtin_asset_path
        weights = str(builtin_asset_path(model_id))
    if num_classes < 1:
        raise ValueError("num_classes must be positive")
    LibreMobileNetV4, LibreYOLO9, LibreRTDETRv4 = _api()
    def construct(cls, **kwargs):
        if weights is None and hasattr(cls, "_from_scratch"):
            model = cls._from_scratch(size=spec.size, nb_classes=num_classes, device=device, **kwargs)
        else:
            model = cls(model_path=weights, size=spec.size, nb_classes=num_classes, device=device, **kwargs)
        if spec.task == "detect":
            from native_detection import configure_detector
            model = configure_detector(model)
        return model
    if spec.family == "mobilenetv4":
        from libreyolo_classification import configure_classifier
        return configure_classifier(construct(LibreMobileNetV4))
    if spec.family == "yolo9":
        return construct(LibreYOLO9, task="detect")
    if spec.family == "dfine":
        from libreyolo import LibreDFINE
        return construct(LibreDFINE, task="detect")
    if spec.family == "rtdetrv4":
        return construct(LibreRTDETRv4, task="detect")
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
    if not val.is_dir():
        raise ValueError(f"검출 검증 이미지 폴더가 없습니다: {val}")
    descriptor = {
        "path": str(dataset_root), "train": "images/train",
        "val": "images/val",
        "nc": len(class_names), "names": list(class_names),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(descriptor, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


_CLASSIFICATION_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff",
}


def _classification_images(root: Path, class_name: str) -> list[Path]:
    class_dir = root / class_name
    if not class_dir.is_dir():
        return []
    return sorted(
        path for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _CLASSIFICATION_IMAGE_EXTENSIONS
    )


def _link_classification_images(files: list[Path], target: Path) -> None:
    """Materialize a LibreYOLO ImageFolder split without changing user data."""
    target.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(files):
        destination = target / f"{index:08d}{source.suffix.lower()}"
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


@contextmanager
def prepare_classification_dataset(
    data_root: str | Path,
    class_names: list[str],
    workspace: str | Path,
    *,
    val_split: float = 0.2,
    seed: int = 0,
):
    """Yield a complete ImageFolder dataset accepted by LibreYOLO.

    New projects contain empty ``val/<class>`` folders. Deep Studio's native
    loaders split ``train`` automatically, while LibreYOLO's ImageFolder rejects
    every empty class. Build a temporary class-stratified view so both engines
    apply the same project behaviour without modifying the user's images.
    """
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise ValueError(f"분류 데이터 폴더가 없습니다: {root}")
    if (not class_names or any(not isinstance(name, str) or not name for name in class_names)
            or len({name.casefold() for name in class_names}) != len(class_names)):
        raise ValueError("분류 학습에는 중복되지 않은 클래스 이름이 필요합니다")
    if not 0.0 < float(val_split) < 1.0:
        raise ValueError("검증 데이터 비율은 0과 1 사이여야 합니다")

    train_root = root / "train" if (root / "train").is_dir() else root
    val_root = root / "val"
    prepared: dict[str, tuple[list[Path], list[Path]]] = {}
    automatic_classes: list[str] = []
    for class_name in class_names:
        train_files = _classification_images(train_root, class_name)
        val_files = _classification_images(val_root, class_name)
        if not train_files:
            extensions = ", ".join(sorted(_CLASSIFICATION_IMAGE_EXTENSIONS))
            raise ValueError(
                f"클래스 '{class_name}'의 학습 이미지를 찾지 못했습니다: "
                f"{train_root / class_name}\n지원 확장자: {extensions}"
            )
        if not val_files:
            if len(train_files) < 2:
                raise ValueError(
                    f"클래스 '{class_name}'는 검증 폴더가 비어 있어 자동 분할해야 하지만 "
                    f"학습 이미지가 {len(train_files)}장뿐입니다. 클래스마다 최소 2장이 필요합니다."
                )
            class_seed = int.from_bytes(
                hashlib.sha256(class_name.encode("utf-8")).digest()[:8], "big"
            )
            shuffled = list(train_files)
            random.Random(int(seed) ^ class_seed).shuffle(shuffled)
            val_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * float(val_split))))
            val_files = sorted(shuffled[:val_count])
            train_files = sorted(shuffled[val_count:])
            automatic_classes.append(class_name)
        prepared[class_name] = (train_files, val_files)

    Path(workspace).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=str(Path(workspace)), prefix=".libreyolo-classification-"
    ) as temporary:
        staged_root = Path(temporary)
        for class_name, (train_files, val_files) in prepared.items():
            _link_classification_images(train_files, staged_root / "train" / class_name)
            _link_classification_images(val_files, staged_root / "val" / class_name)
        yield staged_root, {
            "automatic_classes": automatic_classes,
            "train_images": sum(len(files[0]) for files in prepared.values()),
            "val_images": sum(len(files[1]) for files in prepared.values()),
        }


_DETECTION_METRIC_NAMES = {
    "mAP_50": ("mAP_50", "metrics/mAP50", "metrics/mAP50(B)"),
    "mAP_50_95": ("mAP_50_95", "metrics/mAP50-95", "metrics/mAP50-95(B)"),
    "AR100": ("AR100", "metrics/AR100", "metrics/ar_100"),
}
_DETECTION_POINT_METRICS = (
    "precision", "recall", "detection_tp", "detection_fp", "detection_fn",
    "detection_support", "detection_confidence", "detection_iou",
)


def detection_validation_metrics(raw_metrics):
    """Map pinned LibreYOLO metrics without its deprecated P/R aliases."""
    metrics = {}
    for name, keys in _DETECTION_METRIC_NAMES.items():
        value = next((raw_metrics[key] for key in keys if key in raw_metrics), None)
        if value is not None and math.isfinite(value) and 0 <= value <= 1:
            metrics[name] = float(value)
    for name in _DETECTION_POINT_METRICS:
        value = raw_metrics.get(name)
        if value is not None and math.isfinite(value):
            metrics[name] = value
    return metrics


class _ProgressCallback:
    def __init__(self, emit: Callable[[dict], None]):
        self.emit = emit
        self.elapsed_seconds = 0.0
        self.detection_validation = None

    def on_detection_validation(self, result):
        self.detection_validation = result

    def on_missing_detection_best(self):
        self.emit({"event": "preprocessing_configured", "message":
                   "이전 Best 체크포인트를 찾지 못했습니다. 재개 후 검증 결과에서 Best를 다시 고릅니다."})

    def on_train_start(self, event) -> None:
        self.emit({"event": "training_started", "epoch": int(event.start_epoch),
                   "total_epochs": int(event.total_epochs), "family": str(event.model_family),
                   "size": event.model_size, "task": str(event.task),
                   "save_dir": str(event.save_dir)})

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
        is_detect = getattr(event, "task", None) == "detect"
        details = {}
        if is_detect:
            metric_name = next((name for name, aliases in _DETECTION_METRIC_NAMES.items()
                                if metric_name in aliases), metric_name)
            metrics = detection_validation_metrics(metrics)
            metrics[metric_name] = float(metric)
            validation = self.detection_validation
            if validation and validation["epoch"] == event.epoch:
                details["detection_per_class"] = validation["per_class"]
        epoch_seconds = float(event.epoch_seconds)
        self.elapsed_seconds += epoch_seconds
        metrics["epoch_time_sec"] = epoch_seconds
        metrics["elapsed_time_sec"] = self.elapsed_seconds
        self.emit({"event": "epoch_finished", "epoch": int(event.epoch),
                   "total_epochs": int(event.total_epochs),
                   "train_loss": float(event.train_loss),
                   "val_loss": float(val_loss),
                   "metric": float(metric),
                   "metric_name": metric_name, "metrics": metrics,
                   "epoch_time_sec": epoch_seconds, "elapsed_time_sec": self.elapsed_seconds,
                   "learning_rate": min((float(value) for value in event.lr.values()), default=0.0),
                   "is_best": bool(event.is_best),
                   "best_metric": float(event.best_metric) if event.best_metric is not None else None,
                   "best_epoch": int(event.best_epoch) if event.best_epoch is not None else None,
                   **details})

    def on_train_exception(self, event) -> None:
        self.emit({"event": "training_exception", "message": str(event.exception_message)})


def train_upstream_model(model_id: str, *, data_root: str | Path, class_names: list[str],
                         output_dir: str | Path, epochs: int, batch_size: int,
                         learning_rate: float, device: str, weights: str | None, pretrained: bool = False,
                         resume: bool, use_amp: bool, patience: int,
                         val_split: float = 0.2, seed: int = 0, center_crop=None,
                         augmentation=None, patch_training=False, patch_scale=1., patches_per_image=0,
                         small_object_coverage=False,
                         emit: Callable[[dict], None]) -> dict:
    """Run an upstream native trainer and return its documented result map."""
    spec = get_upstream_spec(model_id)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if resume and not weights:
        raise ValueError("학습 재개에는 같은 모델의 last.pt 또는 best.pt가 필요합니다")
    from native_patches import patch_contract
    from studio_augmentation import augmentation_contract
    augmentation = augmentation_contract(augmentation)
    patches = patch_contract(spec.task, spec.input_size, patch_scale, batch_size) if patch_training else None
    if type(small_object_coverage) is not bool or (
        small_object_coverage and (spec.task != "detect" or not patches or not patches_per_image)
    ):
        raise ValueError("작은 객체 우선에는 탐지 패치 학습과 이미지당 패치 수 1 이상이 필요합니다")
    if center_crop is not None:
        from crop_dataset import prepare_crop_dataset
        data_root = prepare_crop_dataset(data_root, output.parent / "generated", spec.task,
                                         center_crop, num_classes=len(class_names))
    common = {"epochs": int(epochs), "batch": int(batch_size), "imgsz": spec.input_size,
                  "lr0": float(learning_rate), "device": device, "project": str(output.parent), "name": output.name,
                  "exist_ok": True, "amp": bool(use_amp), "patience": int(patience),
                  "callbacks": _ProgressCallback(emit), "val_loss": True, "eval_interval": 1, "seed": int(seed)}
    emit({"event": "preprocessing_configured", "policy": "native_patches_v1" if patches else ("libreyolo_full_frame_v1" if spec.task == "classify" else "studio_detector_v1"),
          "message": f"증강 적용: {augmentation} | " +
          (f"{spec.input_size}px 패치 / 배율 {patch_scale:g} / " +
           ("원본 한 장 단위 분류" if spec.task == "classify" else "검증 mAP는 패치 단위")
           if patches else "전체 입력 영역 사용")})
    if spec.task == "detect":
        from libreyolo.data.utils import IMG_FORMATS
        from native_detection_training import train_detector
        from spatial_data import validate_spatial_dataset
        validation = validate_spatial_dataset(data_root, "detect", len(class_names), extensions=IMG_FORMATS)["val"]
        with tempfile.TemporaryDirectory(prefix="studio-detect-", dir=output.parent) as temp:
            if patches:
                from detection_patch_dataset import prepare_detection_patches
                data_root = prepare_detection_patches(data_root, Path(temp) / "patches", patches,
                    num_classes=len(class_names), patches_per_image=patches_per_image, seed=seed,
                    small_object_coverage=small_object_coverage)
                validation = validate_spatial_dataset(data_root, "detect", len(class_names))["val"]
                if small_object_coverage:
                    coverage = json.loads((data_root / "patches.json").read_text(encoding="utf-8"))["small_object_coverage"]
                    emit({"event": "preprocessing_configured", "policy": "small_object_patch_coverage_v1",
                          "message": "완전히 포함된 작은 객체 (패치 수가 제한된 학습 이미지) "
                                     f"{coverage['randomly_covered']} → {coverage['selected_covered']} / {coverage['small_targets']}",
                          **coverage})
            unit = "패치" if patches else "이미지"
            emit({"event": "preprocessing_configured", "message":
                  f"탐지 검증 {validation['images']}개 {unit} · 정답 박스 {validation['objects']}개"})
            source = str(write_detection_dataset_yaml(data_root, class_names, output / "dataset.yaml"))
            model = build_upstream_model(model_id, num_classes=len(class_names), device=device,
                                         weights=weights, pretrained=pretrained)
            return train_detector(model, data=source, resume=resume, center_crop=center_crop,
                                  augmentation=augmentation, patches=patches,
                                  small_object_coverage=small_object_coverage, train_patch_limit=patches_per_image,
                                  **common)
    with prepare_classification_dataset(
        data_root, class_names, output.parent, val_split=val_split, seed=seed,
    ) as (prepared_root, summary):
        emit({"event": "dataset_prepared", **summary})
        model = build_upstream_model(model_id, num_classes=len(class_names), device=device,
                                     weights=weights, pretrained=pretrained)
        from libreyolo_training import train_classifier
        return train_classifier(model, data=str(prepared_root), resume=resume,
                                center_crop=center_crop, augmentation=augmentation, patches=patches, **common)
