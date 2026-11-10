"""Studio MobileNetV4 input contract: full RGB field of view, no implicit crop."""
from types import MethodType

import numpy as np
import torch

from opencv_preprocess import IMPLEMENTATION, resize, resize_contract

POLICY = "libreyolo_full_frame_v1"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def preprocessing_contract(size, saved=None):
    contract = resize_contract({
        "policy": POLICY, "resize_implementation": IMPLEMENTATION,
        "input_size": [int(size), int(size)], "in_channels": 3, "color_order": "RGB",
        "normalize_mean": list(MEAN), "normalize_std": list(STD), "center_crop": None,
    })
    if saved is not None and saved != contract:
        raise ValueError("LibreYOLO 전체 영상 전처리 계약이 현재 모델 설정과 다릅니다")
    return contract


class FullFrameTransform:
    """Pickle-safe train/val transform. No hidden crop or edge removal."""
    def __init__(self, size, augment=False):
        self.size = int(size)
        if self.size < 1:
            raise ValueError("양의 입력 크기가 필요합니다")
        self.augment = bool(augment)

    def __call__(self, image):
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        pixels = np.asarray(image)
        if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
            raise ValueError("LibreYOLO 분류 입력은 RGB uint8 영상이어야 합니다")
        pixels = resize(pixels, (self.size, self.size))
        if self.augment and torch.rand(()).item() < .5:
            pixels = pixels[:, ::-1]
        chw = pixels.transpose(2, 0, 1).astype(np.float32) / np.float32(255)
        chw = (chw - np.asarray(MEAN, np.float32)[:, None, None]) / np.asarray(STD, np.float32)[:, None, None]
        return torch.from_numpy(np.ascontiguousarray(chw))


def _preprocess(wrapper, image, color_format="auto", input_size=None):
    from libreyolo.utils.image_loader import ImageLoader
    source = ImageLoader.load(image, color_format=color_format)
    size = input_size if input_size is not None else wrapper.input_size
    return FullFrameTransform(size)(source).unsqueeze(0), source, source.size, 1.0


def preprocess_numpy(image, input_size):
    # Export calibration consumes (CHW ndarray, ratio), unlike native predict.
    if isinstance(input_size, (list, tuple)):
        if len(input_size) != 2 or input_size[0] != input_size[1]:
            raise ValueError("LibreYOLO 분류 입력 크기는 정사각형이어야 합니다")
        input_size = input_size[0]
    return FullFrameTransform(input_size)(image).numpy(), 1.0


def _get_preprocess_numpy(wrapper):
    return preprocess_numpy


def configure_classifier(wrapper):
    """Adapt only this wrapper; never monkeypatch LibreYOLO's global transforms."""
    wrapper._preprocess = MethodType(_preprocess, wrapper)
    wrapper._get_preprocess_numpy = MethodType(_get_preprocess_numpy, wrapper)
    wrapper.studio_input_policy = POLICY
    return wrapper
