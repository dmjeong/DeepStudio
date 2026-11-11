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
    # MobileNetV4 Medium/Large: Small보다 깊고 넓다(정확도 ↑, 시간 ↑). Large의 기본 입력은 256
    "libreyolo_classify_mobilenetv4_medium": UpstreamModelSpec(
        "libreyolo_classify_mobilenetv4_medium", "mobilenetv4", "classify", "m", 224,
        "LibreMobileNetV4m-cls.pt"),
    "libreyolo_classify_mobilenetv4_large": UpstreamModelSpec(
        "libreyolo_classify_mobilenetv4_large", "mobilenetv4", "classify", "l", 256,
        "LibreMobileNetV4l-cls.pt"),
    "libreyolo_detect_9t": UpstreamModelSpec(
        "libreyolo_detect_9t", "yolo9", "detect", "t", 640, "LibreYOLO9t.pt"),
    # LibreYOLO9 S/M/C: Tiny와 같은 구조의 큰 변형 (같은 학습·내보내기·C++ 경로)
    "libreyolo_detect_9s": UpstreamModelSpec(
        "libreyolo_detect_9s", "yolo9", "detect", "s", 640, "LibreYOLO9s.pt"),
    "libreyolo_detect_9m": UpstreamModelSpec(
        "libreyolo_detect_9m", "yolo9", "detect", "m", 640, "LibreYOLO9m.pt"),
    "libreyolo_detect_9c": UpstreamModelSpec(
        "libreyolo_detect_9c", "yolo9", "detect", "c", 640, "LibreYOLO9c.pt"),
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
            checkpoint.get("model_family") in {"mobilenetv4", "yolo9", "rtdetrv4"} and
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
    if family not in {"mobilenetv4", "yolo9", "rtdetrv4"}:
        raise ValueError(f"Deep Vision Studio 기본 LibreYOLO 모델이 아닙니다: {family or 'unknown'}")
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


class _ProgressCallback:
    def __init__(self, emit: Callable[[dict], None]):
        self.emit = emit
        self.elapsed_seconds = 0.0

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
                   "best_epoch": int(event.best_epoch) if event.best_epoch is not None else None})

    def on_train_exception(self, event) -> None:
        self.emit({"event": "training_exception", "message": str(event.exception_message)})


TILING_KEY = "dvs_tiling"   # 패치 학습한 검출 체크포인트에 덧붙이는 타일 규칙 (LibreYOLO는 모르는 키를 그대로 둔다)
PATCH_TRAINING_FAMILIES = frozenset({"yolo9"})


def checkpoint_tiling(checkpoint) -> dict | None:
    """패치 학습한 검출 체크포인트의 타일 규칙. 전체 축소로 학습한 모델이면 None.

    {"tile": 640, "overlap": 128, "scale": 1.0, "interpolation": "INTER_AREA", "region": [가로, 세로] 또는 None}
    """
    value = checkpoint.get(TILING_KEY) if isinstance(checkpoint, dict) else None
    if value is None:
        return None
    try:
        tile, overlap, scale = int(value["tile"]), int(value["overlap"]), float(value["scale"])
        region = value.get("region")
        region = None if region is None else [int(region[0]), int(region[1])]
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("체크포인트의 타일 규칙(dvs_tiling) 형식 오류") from exc
    if tile < 32 or not 0 <= overlap < tile or not 0 < scale <= 1 or (region and min(region) < 1):
        raise ValueError("체크포인트의 타일 규칙(dvs_tiling) 값 범위 오류")
    return {"tile": tile, "overlap": overlap, "scale": scale,
            "interpolation": str(value.get("interpolation", "INTER_AREA")), "region": region}


def stamp_tiling(path: str | Path, contract: dict) -> None:
    """학습이 끝난 체크포인트에 타일 규칙을 기록한다 (추론 · 내보내기가 같은 배율과 타일 크기를 쓰도록)."""
    import torch
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint[TILING_KEY] = dict(contract)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)          # 쓰다 만 파일이 체크포인트를 대신하지 않게


def detection_augmentation_arguments(augmentation: dict | None) -> dict:
    """화면의 증강 설정 → LibreYOLO 검출 학습 인자. None이면 빈 dict(모델 기본 증강을 그대로 쓴다).

        수평 뒤집기 확률 ─▶ flip_prob
        회전 각도        ─▶ degrees      (LibreYOLO는 모자이크로 합친 영상에만 회전을 건다)
        색상 변화        ─▶ hsv_prob     (0이면 끔, 0보다 크면 켬 — LibreYOLO의 세기는 고정)
        모자이크         ─▶ mosaic_prob  (켬 = 1.0, 끔 = 0.0)
    """
    if augmentation is None:
        return {}
    flip = float(augmentation.get("horizontal_flip", 0.5))
    rotation = float(augmentation.get("rotation", 0.0))
    color = float(augmentation.get("color_jitter", 0.0))
    if not 0 <= flip <= 1 or not 0 <= rotation <= 180 or not 0 <= color <= 1:
        raise ValueError("증강 설정 범위 오류: 뒤집기·색상은 0~1, 회전은 0~180")
    return {"flip_prob": flip, "degrees": rotation, "hsv_prob": 1.0 if color > 0 else 0.0,
            "mosaic_prob": 1.0 if augmentation.get("mosaic") else 0.0}


@contextmanager
def classification_augmentation(augmentation: dict | None):
    """LibreYOLO 분류 학습의 고정 증강을 화면 설정으로 바꾼다. None이면 아무것도 바꾸지 않는다.

        LibreYOLO 기본   RandomResizedCrop(면적 50~100%) + 좌우반전 0.5   ← 인자로 바꿀 수 없다
        직접 설정        검증과 같은 구도(Resize + CenterCrop) + 화면의 뒤집기 · 회전 · 색상

    무작위 잘라내기는 가장자리 결함을 영상 밖으로 밀어낼 수 있어 직접 설정에서는 쓰지 않는다.
    고정된 libreyolo 버전의 변환 생성 함수 하나를 학습 동안만 바꿔 끼우고, 끝나면 되돌린다.
    """
    if augmentation is None:
        yield
        return
    flip = float(augmentation.get("horizontal_flip", 0.5))
    rotation = float(augmentation.get("rotation", 0.0))
    color = float(augmentation.get("color_jitter", 0.0))
    if not 0 <= flip <= 1 or not 0 <= rotation <= 180 or not 0 <= color <= 1:
        raise ValueError("증강 설정 범위 오류: 뒤집기·색상은 0~1, 회전은 0~180")
    import libreyolo.data.classify_dataset as module
    from torchvision import transforms
    original = module.build_classify_transforms

    def build(imgsz, augment, **options):
        if not augment:
            return original(imgsz, augment, **options)
        # 검증 변환(Resize → CenterCrop → ToTensor → Normalize)을 바탕으로 텐서 변환 앞에 증강을 끼운다
        base = {key: value for key, value in options.items() if key not in ("auto_augment", "erasing")}
        steps = list(original(imgsz, False, **base).transforms)
        position = next(index for index, step in enumerate(steps) if isinstance(step, transforms.ToTensor))
        added = []
        if flip > 0:
            added.append(transforms.RandomHorizontalFlip(flip))
        if rotation > 0:
            added.append(transforms.RandomRotation(rotation))
        if color > 0:
            added.append(transforms.ColorJitter(brightness=color, contrast=color))
        return transforms.Compose(steps[:position] + added + steps[position:])

    module.build_classify_transforms = build
    try:
        yield
    finally:
        module.build_classify_transforms = original


def train_upstream_model(model_id: str, *, data_root: str | Path, class_names: list[str],
                         output_dir: str | Path, epochs: int, batch_size: int,
                         learning_rate: float, device: str, weights: str | None, pretrained: bool = False,
                         resume: bool, use_amp: bool, patience: int,
                         val_split: float = 0.2, seed: int = 0, augmentation: dict | None = None,
                         tiling: dict | None = None,
                         emit: Callable[[dict], None]) -> dict:
    """Run an upstream native trainer and return its documented result map.

    augmentation: None이면 모델 기본 증강. dict면 화면 설정(horizontal_flip · rotation · color_jitter · mosaic)을 적용한다.
    tiling: 검출 패치 학습. None이면 영상 전체를 입력 크기로 줄여 학습한다.
            {"scale": 배율, "max_tiles": 영상당 타일 상한(0 = 제한 없음)} 이면 타일로 잘라 학습하고,
            결과 체크포인트에 타일 규칙을 기록한다 (LibreYOLO9만: 판정 영역 크기 그대로 C++로 내보낼 수 있는 구조).
    """
    spec = get_upstream_spec(model_id)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if resume and not weights:
        raise ValueError("학습 재개에는 같은 모델의 last.pt 또는 best.pt가 필요합니다")
    if tiling is not None and (spec.task != "detect" or spec.family not in PATCH_TRAINING_FAMILIES):
        raise ValueError("검출 패치 학습은 LibreYOLO9 모델만 지원합니다. RT-DETR는 입력 크기가 고정된 구조라 "
                         "판정 영역 전체를 한 번에 추론하는 C++ 배포를 만들 수 없습니다.")
    if spec.task == "detect":
        def run(dataset_root):
            source = str(write_detection_dataset_yaml(dataset_root, class_names, output / "dataset.yaml"))
            model = build_upstream_model(model_id, num_classes=len(class_names), device=device,
                                         weights=weights, pretrained=pretrained)
            callback = _ProgressCallback(emit)
            return model.train(data=source, epochs=int(epochs), batch=int(batch_size), imgsz=spec.input_size,
                               lr0=float(learning_rate), device=device, project=str(output.parent),
                               name=output.name, exist_ok=True, resume=bool(resume), amp=bool(use_amp),
                               patience=int(patience), callbacks=callback, val_loss=True,
                               eval_interval=1, **detection_augmentation_arguments(augmentation))

        if tiling is None:
            return run(data_root)
        from detection_tiles import prepare_tiled_detection_dataset
        # 타일은 학습 동안만 쓰는 임시 데이터다 (원본은 바꾸지 않는다). 시드가 같으면 재개 때도 같은 타일이 나온다.
        with tempfile.TemporaryDirectory(dir=str(output.parent), prefix=".libreyolo-tiles-") as temporary:
            summary = prepare_tiled_detection_dataset(
                data_root, temporary, tile=spec.input_size, scale=float(tiling.get("scale", 1.0)),
                max_tiles=int(tiling.get("max_tiles", 0)), seed=seed)
            emit({"event": "tiles_prepared", **summary})
            result = run(temporary)
        contract = {key: summary[key] for key in ("tile", "overlap", "scale", "interpolation", "region")}
        for key in ("best_checkpoint", "last_checkpoint"):
            if result.get(key) and Path(result[key]).is_file():
                stamp_tiling(result[key], contract)
        result["tiling"] = contract
        return result

    with prepare_classification_dataset(
        data_root, class_names, output.parent, val_split=val_split, seed=seed,
    ) as (prepared_root, summary):
        emit({"event": "dataset_prepared", **summary})
        model = build_upstream_model(model_id, num_classes=len(class_names), device=device,
                                     weights=weights, pretrained=pretrained)
        callback = _ProgressCallback(emit)
        with classification_augmentation(augmentation):
            return model.train(data=str(prepared_root), epochs=int(epochs), batch=int(batch_size),
                               imgsz=spec.input_size, lr0=float(learning_rate), device=device,
                               project=str(output.parent), name=output.name, exist_ok=True,
                               resume=bool(resume), amp=bool(use_amp), patience=int(patience),
                               callbacks=callback, val_loss=True, eval_interval=1)
