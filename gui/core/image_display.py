"""이미지 저장 정밀도를 유지하면서 Qt용 8비트 RGB를 만든다."""

import numpy as np


def display_rgb(image):
    pixels = np.asarray(image)
    if pixels.ndim == 2:
        pixels = np.repeat(pixels[..., None], 3, axis=2)
    if pixels.ndim != 3 or pixels.shape[2] not in (1, 3, 4) or min(pixels.shape[:2]) < 1:
        raise ValueError("표시 이미지 형식은 HxW, HxWx3 또는 HxWx4여야 합니다")
    if pixels.shape[2] == 1:
        pixels = np.repeat(pixels, 3, axis=2)
    if pixels.dtype == np.uint16:
        pixels = np.rint(pixels.astype(np.float32) / 257).astype(np.uint8)
    elif np.issubdtype(pixels.dtype, np.floating):
        if not np.isfinite(pixels).all():
            raise ValueError("표시 이미지에 NaN 또는 무한대가 있습니다")
        scale = 255 if pixels.min() >= 0 and pixels.max() <= 1 else 1
        pixels = np.rint(np.clip(pixels * scale, 0, 255)).astype(np.uint8)
    elif pixels.dtype != np.uint8:
        raise ValueError(f"지원하지 않는 표시 이미지 자료형: {pixels.dtype}")
    if pixels.shape[2] == 4:
        alpha = pixels[..., 3:4].astype(np.float32) / 255
        pixels = np.rint(pixels[..., :3] * alpha).astype(np.uint8)
    return np.ascontiguousarray(pixels)
