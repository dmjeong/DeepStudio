"""Studio MobileNetV4 input contract: full RGB field of view, no implicit crop."""
from types import MethodType

import numpy as np
import torch

from upstream_original_size import input_hw
from opencv_preprocess import IMPLEMENTATION, resize, resize_contract

POLICY = "libreyolo_full_frame_v1"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def preprocessing_contract(size, saved=None, *, original_size_region=None):
    contract = resize_contract({
        "policy": POLICY, "resize_implementation": IMPLEMENTATION,
        "input_size": list(input_hw(size)), "in_channels": 3, "color_order": "RGB",
        "normalize_mean": list(MEAN), "normalize_std": list(STD), "center_crop": None,
    })
    if original_size_region is not None:
        contract["original_size_region"] = list(original_size_region)
    if saved is not None and saved != contract:
        raise ValueError("LibreYOLO 전체 영상 전처리 계약이 현재 모델 설정과 다릅니다")
    return contract


class FullFrameTransform:
    """Pickle-safe train/val transform. No hidden crop or edge removal."""
    def __init__(self, size, augment=False, augmentation=None, *, original_size=False):
        self.original_size = bool(original_size)
        self.size = input_hw(size)
        self.augment = bool(augment)
        from studio_augmentation import augmentation_contract
        self.augmentation = augmentation_contract(augmentation)

    def __call__(self, image):
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        pixels = np.asarray(image)
        if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
            raise ValueError("LibreYOLO 분류 입력은 RGB uint8 영상이어야 합니다")
        if pixels.shape[:2] != self.size:
            if self.original_size:
                raise ValueError(f"원본 학습 영역 {self.size[1]}×{self.size[0]} px가 필요합니다.")
            pixels = resize(pixels, self.size)
        if self.augment:
            from studio_augmentation import augment_rgb
            pixels, _ = augment_rgb(pixels, self.augmentation)
        chw = pixels.transpose(2, 0, 1).astype(np.float32) / np.float32(255)
        chw = (chw - np.asarray(MEAN, np.float32)[:, None, None]) / np.asarray(STD, np.float32)[:, None, None]
        return torch.from_numpy(np.ascontiguousarray(chw))


def _preprocess(wrapper, image, color_format="auto", input_size=None):
    from libreyolo.utils.image_loader import ImageLoader
    source = ImageLoader.load(image, color_format=color_format)
    size = input_size if input_size is not None else wrapper.input_size
    native = getattr(wrapper, "studio_original_size_region", None)
    if native:
        if source.size != tuple(native):
            raise ValueError(f"원본 학습 영역 {native[0]}×{native[1]} px가 필요합니다.")
        size = native[::-1]
    return FullFrameTransform(size)(source).unsqueeze(0), source, source.size, 1.0


def preprocess_numpy(image, input_size):
    # Export calibration consumes (CHW ndarray, ratio), unlike native predict.
    return FullFrameTransform(input_size)(image).numpy(), 1.0


def _get_preprocess_numpy(wrapper):
    return preprocess_numpy


def configure_classifier(wrapper):
    """Adapt only this wrapper; never monkeypatch LibreYOLO's global transforms."""
    wrapper._preprocess = MethodType(_preprocess, wrapper)
    wrapper._get_preprocess_numpy = MethodType(_get_preprocess_numpy, wrapper)
    wrapper.studio_input_policy = POLICY
    return wrapper
