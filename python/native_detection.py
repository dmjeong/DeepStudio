"""Native detector image contract shared by train, app and ONNX SDK."""
from types import MethodType

import cv2
import numpy as np
import torch
from opencv_preprocess import IMPLEMENTATION


def preprocessing_contract(family, size):
    return {"policy": "studio_detector_v1", "resize_implementation": IMPLEMENTATION,
            "resize_mode": "letterbox_top_left" if family == "yolo9" else "stretch",
            "resize_interpolation": "linear_exact", "input_size": [size, size],
            "in_channels": 3, "color_order": "RGB", "normalize_mean": [0., 0., 0.],
            "normalize_std": [1., 1., 1.], "center_crop": None}


def detector_pixels(rgb, size, family):
    h, w = rgb.shape[:2]
    if family == "yolo9":
        ratio = min(size / h, size / w)
        nw, nh = max(1, int(w * ratio)), max(1, int(h * ratio))
        pixels = np.full((size, size, 3), 114, np.uint8)
        pixels[:nh, :nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR_EXACT)
        return pixels, (ratio, ratio)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR_EXACT), (size / w, size / h)


def _preprocess(wrapper, image, color_format="auto", input_size=None):
    from libreyolo.utils.image_loader import ImageLoader
    source = ImageLoader.load(image, color_format=color_format)
    pixels, _ = detector_pixels(np.asarray(source.convert("RGB")), input_size or wrapper.input_size, wrapper.FAMILY)
    tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))).float().div_(255)
    return tensor.unsqueeze(0), source, source.size, 1.


class DetectorValidationTransform:
    """Original pixels and boxes enter once; validation uses deployment resizing."""
    wants_unresized_image = True
    custom_normalization = True
    normalize = True

    def __init__(self, family, size):
        self.family, self.img_size, self.max_labels = family, (size, size), 300
        self.uses_letterbox = family == "yolo9"

    def letterbox_scale(self, orig_h, orig_w, imgsz):
        return min(imgsz / orig_h, imgsz / orig_w), 0., 0.

    def __call__(self, image, targets, input_size):
        if input_size[0] != input_size[1]:
            raise ValueError("탐지 입력 크기는 정사각형이어야 합니다")
        pixels, (sx, sy) = detector_pixels(image[:, :, ::-1], input_size[0], self.family)
        if len(targets) > self.max_labels:
            raise ValueError("검증 이미지의 박스 개수가 설정 한도를 넘습니다")
        labels = np.zeros((self.max_labels, 5), np.float32)
        if len(targets):
            labels[:len(targets)] = targets
            labels[:len(targets), :4] *= np.asarray([sx, sy, sx, sy], np.float32)
        return np.ascontiguousarray(pixels.transpose(2, 0, 1), dtype=np.float32) / 255, labels


def _get_val_preprocessor(wrapper, img_size=None):
    return DetectorValidationTransform(wrapper.FAMILY, img_size or wrapper.input_size)


def configure_detector(wrapper):
    wrapper._preprocess = MethodType(_preprocess, wrapper)
    wrapper._get_val_preprocessor = MethodType(_get_val_preprocessor, wrapper)
    return wrapper
