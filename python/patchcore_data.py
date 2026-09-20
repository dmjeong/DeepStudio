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
    """정상 학습과 보정 split을 발견한다.

    val/test 중 정상과 불량이 모두 있는 split을 우선한다. 비어 있지 않은 val이
    정상만 포함해도 test의 불량 평가를 가리는 기존 동작을 피한다.
    """
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
        candidates = []
        for name in ("val", "test"):
            candidate = root / name
            paths = image_paths(candidate)
            if not paths:
                continue
            candidate_labels = []
            for filename in paths:
                parts = Path(filename).relative_to(candidate).parts
                if len(parts) < 2:
                    raise ValueError(f"평가 이미지의 정상/불량 폴더 라벨 없음: {filename}")
                candidate_labels.append(0 if parts[0].casefold() in NORMAL_NAMES else 1)
            candidates.append((paths, candidate_labels, name))
        if candidates:
            # F1/AUROC 및 자동 임계값에는 두 클래스가 반드시 필요하다.
            validation, labels, split = next(
                (item for item in candidates if set(item[1]) == {0, 1}), candidates[0])
    overlap = set(map(lambda p: str(Path(p).resolve()), training)) & set(
        map(lambda p: str(Path(p).resolve()), validation))
    if overlap:
        raise ValueError(f"학습/평가에 같은 이미지 경로 사용: {next(iter(overlap))}")
    return training, validation, labels, split


def input_image(path, preprocessing="opencv_full_range_v2"):
    """Load a detached image in the same orientation for inference and preview."""
    with Image.open(path) as source:
        if preprocessing in ("full_range_v1", "opencv_full_range_v2"):
            return ImageOps.exif_transpose(source)
        if preprocessing == "legacy_pil_rgb":
            return source.copy()
        raise ValueError(f"미지원 전처리: {preprocessing}")


def validate_crop_images(paths, crop, preprocessing="opencv_full_range_v2", cancel_callback=None):
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
                if preprocessing in ("full_range_v1", "opencv_full_range_v2") and image.getexif().get(274) in (5, 6, 7, 8):
                    width, height = height, width
                center_crop_box((width, height), crop)
        except (OSError, ValueError) as exc:
            raise ValueError(f"중앙 크롭 이미지 확인 실패: {path}\n{exc}") from exc


def input_tensor(path, size, preprocessing="opencv_full_range_v2", center_crop=None):
    """16비트 전체 범위와 OpenCV C++ SDK 리사이즈 계약을 보존한다."""
    import torch
    try:
        with input_image(path, preprocessing) as source:
            image = source.crop(center_crop_box(source.size, center_crop)) if center_crop is not None else source
            if preprocessing == "legacy_pil_rgb":
                pixels = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255
            elif preprocessing in ("full_range_v1", "opencv_full_range_v2"):
                array = np.asarray(image)
                if array.dtype.itemsize == 2 and array.dtype.kind == "u" and array.ndim in (2, 3):
                    height = array
                    if preprocessing == "opencv_full_range_v2":
                        import cv2
                        height = cv2.resize(height, (size, size), interpolation=cv2.INTER_LINEAR_EXACT)
                    else:
                        if height.ndim != 2:
                            raise ValueError("이전 PIL 전처리는 16비트 RGB 이미지를 지원하지 않습니다")
                        height = np.asarray(Image.fromarray(height.astype(np.float32) / 65535).resize(
                            (size, size), Image.Resampling.BILINEAR), dtype=np.float32)
                    if height.ndim == 2:
                        pixels = np.repeat((height.astype(np.float32) / (65535 if preprocessing == "opencv_full_range_v2" else 1))[..., None], 3, axis=2)
                    else:
                        pixels = height.astype(np.float32) / 65535
                elif array.ndim == 2 and image.mode == "I" and array.min() >= 0 and array.max() <= 65535:
                    height = array.astype(np.float32) / 65535
                    if preprocessing == "opencv_full_range_v2":
                        import cv2
                        height = cv2.resize(height, (size, size), interpolation=cv2.INTER_LINEAR_EXACT)
                    else:
                        height = np.asarray(Image.fromarray(height).resize(
                            (size, size), Image.Resampling.BILINEAR), dtype=np.float32)
                    pixels = np.repeat(height[..., None], 3, axis=2)
                elif array.dtype.kind == "f" or array.dtype.kind == "i":
                    raise ValueError("실수/범위 밖 정수 높이맵은 먼저 명시적인 범위로 uint16 또는 uint8 변환 필요")
                else:
                    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                    if preprocessing == "opencv_full_range_v2":
                        import cv2
                        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR_EXACT)
                    else:
                        rgb = np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)
                    pixels = rgb.astype(np.float32) / 255
            else:
                raise ValueError(f"미지원 전처리: {preprocessing}")
        tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (tensor - mean) / std
    except Exception as exc:
        raise ValueError(f"PatchCore 이미지 읽기/전처리 오류: {path}\n{exc}") from exc


class PatchCoreDataset:
    def __init__(self, samples, input_size, labels=None, preprocessing="opencv_full_range_v2", center_crop=None):
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
