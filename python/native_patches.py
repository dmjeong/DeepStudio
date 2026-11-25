"""Versioned native-image patches: detection boxes / image-label MIL classification."""
import math

import cv2
import numpy as np
import torch


def patch_contract(task, size, scale=1., batch=8):
    if task not in {"classify", "detect"} or type(size) is not int or not 32 <= size <= 4096:
        raise ValueError("패치 태스크/크기가 잘못되었습니다")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not .1 <= scale <= 1:
        raise ValueError("패치 배율은 0.1~1.0 필요")
    if type(batch) is not int or not 1 <= batch <= 256:
        raise ValueError("패치 배치는 1~256 정수 필요")
    return {"version": 1, "task": task, "size": size, "overlap": size // 4,
            "scale": float(scale), "batch": batch,
            "aggregation": "max_logits" if task == "classify" else "class_aware_nms",
            "evaluation_unit": "image" if task == "classify" else "patch"}


def validate_patches(value, task=None):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("패치 계약은 객체여야 합니다")
    if any(type(value.get(key)) is not int for key in ("version", "size", "overlap", "batch")):
        raise ValueError("패치 버전·크기·겹침·배치는 정수여야 합니다")
    expected = patch_contract(value.get("task"), value.get("size"), value.get("scale"), value.get("batch"))
    if value != expected or (task and value["task"] != task):
        raise ValueError("패치 배포 계약이 일치하지 않습니다")
    return dict(value)


def origins(length, size, stride):
    if length <= size:
        return [0]
    result = list(range(0, length - size + 1, stride))
    if result[-1] != length - size:
        result.append(length - size)
    return result


def scaled_image(image, contract):
    h, w = image.shape[:2]
    # Explicit half-up rounding, shared with C++ (not Python's bankers rounding).
    nw, nh = max(1, int(w * contract["scale"] + .5)), max(1, int(h * contract["scale"] + .5))
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA) if (nw, nh) != (w, h) else image


def tiles(image, contract):
    size = contract["size"]
    stride = size - contract["overlap"]
    h, w = image.shape[:2]
    for y in origins(h, size, stride):
        for x in origins(w, size, stride):
            pw, ph = min(size, w-x), min(size, h-y)
            tile = np.full((size, size, 3), 114, np.uint8)
            tile[:ph, :pw] = image[y:y+ph, x:x+pw]
            yield tile, (x, y, pw, ph)


class PatchBagTransform:
    """One source image remains one training label; no per-tile pseudo-labels."""
    def __init__(self, contract, augment=False, augmentation=None):
        from libreyolo_classification import FullFrameTransform
        self.contract = validate_patches(contract, "classify")
        self.transform = FullFrameTransform(contract["size"], augment=augment, augmentation=augmentation)

    def __call__(self, image):
        pixels = scaled_image(np.asarray(image.convert("RGB")), self.contract)
        return torch.stack([self.transform(tile) for tile, _ in tiles(pixels, self.contract)])


def bag_logits(model, images, chunk=8):
    """B=1 for variable-size bags; gradients flow only through each class's maxima."""
    if images.ndim == 4:
        return model(images)
    if images.ndim != 5 or images.shape[0] != 1:
        raise ValueError("패치 분류는 원본 이미지 1장씩 묶어 처리합니다")
    logits = None
    for block in images[0].split(chunk):
        # A tail with one tile cannot estimate pooled-head BatchNorm variance.
        # Use saved running statistics only for singleton forwards, preserving
        # affine gradients; never duplicate a tile and pretend it is a new sample.
        single_bn = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                     and m.training] if block.shape[0] == 1 else []
        for m in single_bn:
            m.eval()
        try:
            value = model(block).amax(dim=0, keepdim=True)
        finally:
            for m in single_bn:
                m.train()
        logits = value if logits is None else torch.maximum(logits, value)
    return logits


def detection_nms(rows, threshold=.5, maximum=300):
    """Deterministic class-aware NMS over original-coordinate boxes."""
    if not rows:
        return []
    boxes = np.asarray([r["bbox"] for r in rows], np.float64)
    labels = np.asarray([r["class_id"] for r in rows])
    scores = np.asarray([r["confidence"] for r in rows])
    order = np.argsort(-scores, kind="stable")
    area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    kept = []
    while len(order) and len(kept) < maximum:
        first, rest = order[0], order[1:]
        kept.append(rows[int(first)])
        intersection = np.maximum(0, np.minimum(boxes[first, 2:], boxes[rest, 2:]) -
                                  np.maximum(boxes[first, :2], boxes[rest, :2])).prod(axis=1)
        union = area[first] + area[rest] - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
        order = rest[(labels[rest] != labels[first]) | (overlap <= threshold)]
    return kept
