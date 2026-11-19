"""Native detector image contract shared by train, app and ONNX SDK."""
from types import MethodType

import cv2
import numpy as np
import torch
from opencv_preprocess import IMPLEMENTATION
from upstream_original_size import detector_size, input_hw


def preprocessing_contract(family, size):
    return {"policy": "studio_detector_v1", "resize_implementation": IMPLEMENTATION,
            "resize_mode": "letterbox_top_left" if family == "yolo9" else "stretch",
            "resize_interpolation": "linear_exact", "input_size": list(input_hw(size)),
            "in_channels": 3, "color_order": "RGB", "normalize_mean": [0., 0., 0.],
            "normalize_std": [1., 1., 1.], "center_crop": None}


def detector_pixels(rgb, size, family, original_region=None):
    h, w = rgb.shape[:2]
    th, tw = input_hw(size)
    if original_region and (w, h) != tuple(original_region):
        raise ValueError(f"원본 학습 영역 {original_region[0]}×{original_region[1]} px가 필요합니다.")
    if (h, w) == (th, tw):
        return rgb, (1., 1.)
    if original_region:
        if th < h or tw < w:
            raise ValueError("탐지 입력이 원본 영역보다 작습니다.")
        pixels = np.full((th, tw, 3), 114 if family == "yolo9" else 0, np.uint8)
        pixels[:h, :w] = rgb
        return pixels, (1., 1.)
    if family == "yolo9":
        ratio = min(th / h, tw / w)
        nw, nh = max(1, int(w * ratio)), max(1, int(h * ratio))
        pixels = np.full((th, tw, 3), 114, np.uint8)
        pixels[:nh, :nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR_EXACT)
        return pixels, (ratio, ratio)
    return cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LINEAR_EXACT), (tw / w, th / h)


def _preprocess(wrapper, image, color_format="auto", input_size=None):
    from libreyolo.utils.image_loader import ImageLoader
    source = ImageLoader.load(image, color_format=color_format)
    native = getattr(wrapper, "studio_original_size_region", None)
    if native and source.size != tuple(native):
        raise ValueError(f"원본 학습 영역 {native[0]}×{native[1]} px가 필요합니다.")
    size = detector_size(wrapper.FAMILY, *native[::-1]) if native else input_size or wrapper.input_size
    pixels, _ = detector_pixels(np.asarray(source.convert("RGB")), size, wrapper.FAMILY, native)
    tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))).float().div_(255)
    return tensor.unsqueeze(0), source, source.size, 1.


class DetectorValidationTransform:
    """Original pixels and boxes enter once; validation uses deployment resizing."""
    wants_unresized_image = True
    custom_normalization = True
    normalize = True

    def __init__(self, family, size, original_region=None):
        self.original_region = original_region
        self.family, self.img_size, self.max_labels = family, input_hw(size), 300
        self.uses_letterbox = bool(original_region) or family == "yolo9"

    def letterbox_scale(self, orig_h, orig_w, imgsz):
        if self.original_region:
            return 1., 0., 0.
        height, width = input_hw(imgsz)
        return min(height / orig_h, width / orig_w), 0., 0.

    def __call__(self, image, targets, input_size):
        pixels, (sx, sy) = detector_pixels(image[:, :, ::-1], input_size, self.family, self.original_region)
        if len(targets) > self.max_labels:
            raise ValueError("검증 이미지의 박스 개수가 설정 한도를 넘습니다")
        labels = np.zeros((self.max_labels, 5), np.float32)
        if len(targets):
            labels[:len(targets)] = targets
            labels[:len(targets), :4] *= np.asarray([sx, sy, sx, sy], np.float32)
        return np.ascontiguousarray(pixels.transpose(2, 0, 1), dtype=np.float32) / 255, labels


def _get_val_preprocessor(wrapper, img_size=None):
    return DetectorValidationTransform(wrapper.FAMILY, img_size or wrapper.input_size,
                                       getattr(wrapper, "studio_original_size_region", None))


def source_frame_outputs(family, region, outputs):
    """Map raw canvas predictions into the valid source region before ranking/NMS."""
    width, height = region
    canvas_h, canvas_w = input_hw(detector_size(family, height, width))
    if (canvas_h, canvas_w) == (height, width):
        return outputs
    if family == "yolo9":
        pred = outputs[0]
        # Inference emits [B,4+classes,anchors] (a validation slice may omit B).
        coords, scores = pred[..., :4, :], pred[..., 4:, :]
        x1 = coords[..., 0:1, :].clamp(0, width)
        y1 = coords[..., 1:2, :].clamp(0, height)
        x2 = coords[..., 2:3, :].clamp(0, width)
        y2 = coords[..., 3:4, :].clamp(0, height)
        scores = torch.where((x2 > x1) & (y2 > y1), scores, 0.)
        return [torch.cat((x1, y1, x2, y2, scores), -2)]
    logits, boxes = outputs
    scale = boxes.new_tensor([canvas_w / width, canvas_h / height])
    center = boxes[..., :2] * scale
    half_extent = boxes[..., 2:] * scale * .5
    lo, hi = (center - half_extent).clamp(0, 1), (center + half_extent).clamp(0, 1)
    extent = hi - lo
    logits = torch.where((extent[..., :1] > 0) & (extent[..., 1:] > 0), logits, -100.)
    return [logits, torch.cat(((hi + lo) * .5, extent), -1)]


def _postprocess(wrapper, output, conf_thres, iou_thres, original_size, **kwargs):
    native = getattr(wrapper, "studio_original_size_region", None)
    if native:
        if tuple(original_size) != tuple(native):
            raise ValueError("탐지 예측 좌표와 원본 크기가 다릅니다.")
        keys = ["predictions"] if wrapper.FAMILY == "yolo9" else ["pred_logits", "pred_boxes"]
        outputs = source_frame_outputs(wrapper.FAMILY, native, [output[k] for k in keys])
        output = {**output, **dict(zip(keys, outputs))}
        kwargs.update(input_size=native[::-1], letterbox=False)
    return type(wrapper)._postprocess(wrapper, output, conf_thres, iou_thres, original_size, **kwargs)


def configure_detector(wrapper):
    wrapper._preprocess = MethodType(_preprocess, wrapper)
    wrapper._postprocess = MethodType(_postprocess, wrapper)
    wrapper._get_val_preprocessor = MethodType(_get_val_preprocessor, wrapper)
    return wrapper
