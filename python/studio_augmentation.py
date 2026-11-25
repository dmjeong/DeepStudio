"""Editable augmentation shared by Studio's native classification/detection adapters."""
import math

import cv2
import numpy as np
import torch


def augmentation_contract(values=None):
    defaults = {"horizontal_flip": .5, "rotation": 0., "color_jitter": 0.}
    values = defaults if values is None else values
    if not isinstance(values, dict):
        raise TypeError("증강 설정은 객체여야 합니다")
    result = {}
    for key, limit in (("horizontal_flip", 1.), ("rotation", 180.), ("color_jitter", 1.)):
        value = values.get(key, defaults[key])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= limit:
            raise ValueError(f"증강 {key}: 0~{limit:g}의 유한한 숫자가 필요합니다")
        result[key] = float(value)
    return result


def augment_rgb(image, settings, targets=None):
    """RGB uint8; optional targets are pixel [x1,y1,x2,y2,class]."""
    pixels = np.asarray(image).copy()
    labels = None if targets is None else np.asarray(targets, dtype=np.float32).copy()
    h, w = pixels.shape[:2]
    flip = settings["horizontal_flip"]
    if flip and torch.rand(()).item() < flip:
        pixels = np.ascontiguousarray(pixels[:, ::-1])
        if labels is not None and len(labels):
            labels[:, [0, 2]] = w - labels[:, [2, 0]]
    degrees = settings["rotation"]
    if degrees:
        angle = (2 * torch.rand(()).item() - 1) * degrees
        matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.)
        # Detection must not reflect unlabeled duplicates into the image.
        pixels = cv2.warpAffine(pixels, matrix, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT if labels is not None else cv2.BORDER_REFLECT_101,
            borderValue=(114, 114, 114))
        if labels is not None and len(labels):
            corners = labels[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 4, 2)
            points = corners @ matrix[:, :2].T + matrix[:, 2]
            labels[:, :2], labels[:, 2:4] = points.min(1), points.max(1)
            labels[:, [0, 2]] = labels[:, [0, 2]].clip(0, w)
            labels[:, [1, 3]] = labels[:, [1, 3]].clip(0, h)
            labels = labels[(labels[:, 2] > labels[:, 0] + 1) & (labels[:, 3] > labels[:, 1] + 1)]
    jitter = settings["color_jitter"]
    if jitter:
        brightness, contrast, saturation = 1 + (torch.rand(3).numpy() * 2 - 1) * jitter
        values = pixels.astype(np.float32) * brightness
        mean = cv2.cvtColor(values, cv2.COLOR_RGB2GRAY).mean()
        values = (values - mean) * contrast + mean
        gray = cv2.cvtColor(values, cv2.COLOR_RGB2GRAY)[..., None]
        pixels = np.rint(gray + (values - gray) * saturation).clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(pixels), labels
