"""PatchCore 전용 정상/평가 데이터 선택과 학습-추론 공통 전처리."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from center_crop import center_crop_box, validate_center_crop


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NORMAL_NAMES = {"good", "normal", "ok", "정상", "양품"}
IMAGENET_MEAN = [.485, .456, .406]
IMAGENET_STD = [.229, .224, .225]


def image_paths(root):
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(str(p) for p in root.rglob("*") if p.is_file()
                  and p.suffix.lower() in IMAGE_EXTENSIONS
                  and not any(part.startswith(".") for part in p.relative_to(root).parts))


def discover_data(root):
    """빈 val/test는 건너뛴다. 학습 폴더의 알려진 정상 클래스만 사용한다."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"PatchCore 데이터 폴더 없음: {root}")
    train_root = root / "train" if (root / "train").is_dir() else root
    normal_dirs = [p for p in sorted(train_root.iterdir()) if p.is_dir() and p.name.casefold() in NORMAL_NAMES]
    if normal_dirs:
        training = sorted(path for folder in normal_dirs for path in image_paths(folder))
    elif train_root.name.casefold() in NORMAL_NAMES or not any(p.is_dir() for p in train_root.iterdir()):
        training = image_paths(train_root)
    else:
        raise ValueError(f"정상 학습 폴더 구분 불가: {train_root}\ntrain/good, train/normal, train/ok 또는 정상 이미지 폴더를 지정하세요")
    if not training:
        raise ValueError(f"정상 학습 이미지 없음: {train_root}")
    validation, labels, split = [], [], ""
    # 정상 폴더를 직접 지정한 경우 인접 폴더를 추측해서 평가하지 않는다.
    if train_root != root:
        for name in ("val", "test"):
            candidate = root / name
            paths = image_paths(candidate)
            if not paths:
                continue
            for filename in paths:
                parts = Path(filename).relative_to(candidate).parts
                if len(parts) < 2:
                    raise ValueError(f"평가 이미지의 정상/불량 폴더 라벨 없음: {filename}")
                labels.append(0 if parts[0].casefold() in NORMAL_NAMES else 1)
            validation, split = paths, name
            break
    overlap = set(map(lambda p: str(Path(p).resolve()), training)) & set(
        map(lambda p: str(Path(p).resolve()), validation))
    if overlap:
        raise ValueError(f"학습/평가에 같은 이미지 경로 사용: {next(iter(overlap))}")
    return training, validation, labels, split


def input_image(path, preprocessing="full_range_v1"):
    """Load a detached image in the same orientation for inference and preview."""
    with Image.open(path) as source:
        if preprocessing == "full_range_v1":
            return ImageOps.exif_transpose(source)
        if preprocessing == "legacy_pil_rgb":
            return source.copy()
        raise ValueError(f"미지원 전처리: {preprocessing}")


def validate_crop_images(paths, crop, preprocessing="full_range_v1", cancel_callback=None):
    """Validate every ROI using image headers before feature extraction."""
    crop = validate_center_crop(crop)
    if crop is None:
        return
    for path in paths:
        if cancel_callback:
            cancel_callback()
        try:
            with Image.open(path) as image:
                width, height = image.size
                if preprocessing == "full_range_v1" and image.getexif().get(274) in (5, 6, 7, 8):
                    width, height = height, width
                center_crop_box((width, height), crop)
        except (OSError, ValueError) as exc:
            raise ValueError(f"중앙 크롭 이미지 확인 실패: {path}\n{exc}") from exc


def input_tensor(path, size, preprocessing="full_range_v1", center_crop=None):
    """16비트 회색조의 전체 범위를 보존하고 RGB ImageNet 정규화를 적용한다."""
    import torch
    try:
        with input_image(path, preprocessing) as source:
            image = source.crop(center_crop_box(source.size, center_crop)) if center_crop is not None else source
            if preprocessing == "legacy_pil_rgb":
                pixels = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
            elif preprocessing == "full_range_v1":
                array = np.asarray(image)
                if array.ndim == 2 and (array.dtype.itemsize == 2 and array.dtype.kind == "u"
                                       or image.mode == "I" and array.min() >= 0 and array.max() <= 65535):
                    height = np.asarray(Image.fromarray(array.astype(np.float32) / 65535).resize(
                        (size, size), Image.Resampling.BILINEAR), dtype=np.float32)
                    pixels = np.repeat(height[..., None], 3, axis=2)
                elif array.dtype.kind == "f" or array.dtype.kind == "i":
                    raise ValueError("실수/범위 밖 정수 높이맵은 먼저 명시적인 범위로 uint16 또는 uint8 변환 필요")
                else:
                    pixels = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
            else:
                raise ValueError(f"미지원 전처리: {preprocessing}")
        tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (tensor - mean) / std
    except Exception as exc:
        raise ValueError(f"PatchCore 이미지 읽기/전처리 오류: {path}\n{exc}") from exc


class PatchCoreDataset:
    def __init__(self, samples, input_size, labels=None, preprocessing="full_range_v1", center_crop=None):
        self.samples = list(samples)
        self.labels = list(labels) if labels is not None else [0] * len(samples)
        self.input_size, self.preprocessing = int(input_size), preprocessing
        self.center_crop = validate_center_crop(center_crop)
        if len(self.samples) != len(self.labels):
            raise ValueError("이미지와 평가 라벨 수 불일치")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return input_tensor(self.samples[index], self.input_size, self.preprocessing, self.center_crop), self.labels[index]
