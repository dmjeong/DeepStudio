"""Use the SDK's detection contract for INT8 calibration and comparison."""
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from center_crop import center_crop_box
from native_patches import detection_nms, origins, scaled_image, validate_patches
from opencv_preprocess import convert_channels

_NATIVE_DETR = {"libreyolo_rtdetrv4", "libreyolo_dfine"}
_MATCH_IOU = .5
_MAX_UNMATCHED_FRACTION = .20


class DetectionInputs:
    """Prepare SDK input windows, keeping coordinates in the cropped image."""

    def __init__(self, config, path):
        self.config = config
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise ValueError(f"INT8 보정 영상 읽기 실패: {path.name}")
        crop = config.get("preprocessing", {}).get("center_crop")
        left, top, right, bottom = center_crop_box((image.shape[1], image.shape[0]), crop)
        region = image[top:bottom, left:right, ::-1]
        self.patches = validate_patches(config.get("patches"), "detect")
        self.pixels = scaled_image(region, self.patches) if self.patches else region
        self.height, self.width = self.pixels.shape[:2]
        self.positions = [(0, 0)]
        if self.patches:
            size = self.patches["size"]
            stride = size - self.patches["overlap"]
            self.positions = [(x, y) for y in origins(self.height, size, stride)
                              for x in origins(self.width, size, stride)]
        self.mean = np.asarray(config["normalize_mean"], np.float32)
        self.std = np.asarray(config["normalize_std"], np.float32)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        x, y = self.positions[index]
        pixels = self.pixels
        visible_w, visible_h = self.width, self.height
        if self.patches:
            size = self.patches["size"]
            visible_w, visible_h = min(size, self.width - x), min(size, self.height - y)
            pixels = np.full((size, size, 3), 114, np.uint8)
            pixels[:visible_h, :visible_w] = self.pixels[y:y + visible_h, x:x + visible_w]
        pixels = convert_channels(pixels, self.config["input_channels"])
        w, h = self.config["input_width"], self.config["input_height"]
        scale_x, scale_y = float(w), float(h)
        if self.config["preprocessing"].get("resize_mode") == "letterbox_top_left":
            source_h, source_w = pixels.shape[:2]
            ratio = min(w / source_w, h / source_h)
            scale_x, scale_y = source_w * ratio, source_h * ratio
            resized = cv2.resize(pixels, (max(1, int(scale_x)), max(1, int(scale_y))),
                                 interpolation=cv2.INTER_LINEAR_EXACT)
            pixels = np.full((h, w, 3), 114, np.uint8)
            pixels[:resized.shape[0], :resized.shape[1]] = resized
        else:
            pixels = cv2.resize(pixels, (w, h), interpolation=cv2.INTER_LINEAR_EXACT)
        pixels = pixels.reshape(h, w, self.config["input_channels"]).astype(np.float32)
        tensor = ((pixels / 255 - self.mean) / self.std).transpose(2, 0, 1)[None]
        return (x, y, visible_w, visible_h, scale_x, scale_y), np.ascontiguousarray(tensor)


def _sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -80, 80)))


def _detections(config, outputs, size):
    if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in outputs):
        raise ValueError("탐지 INT8 비교 출력에 형식 오류 또는 NaN/Inf가 있습니다")
    backend, classes, post = config["backend"], config["num_classes"], config["postprocessing"]
    native_detr = backend in _NATIVE_DETR
    if native_detr or backend == "redetr_v4":
        logits, boxes = outputs if native_detr else outputs[::-1]
        if (logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] == 0
                or logits.shape[2] != classes or boxes.shape != (1, logits.shape[1], 4)):
            raise ValueError("탐지 INT8 비교 DETR 출력 형상이 다릅니다")
        scores, boxes = logits[0], boxes[0]
        if post["class_scores"] == "softmax":
            scores = np.exp(scores - scores.max(1, keepdims=True))
            scores /= scores.sum(1, keepdims=True)
        else:
            scores = _sigmoid(scores)
    elif backend == "libreyolo_yolo9":
        value, = outputs
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] != 4 + classes or not value.shape[2]:
            raise ValueError("탐지 INT8 비교 LibreYOLO9 출력 형상이 다릅니다")
        boxes, scores = value[0, :4].T, value[0, 4:].T
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("탐지 INT8 비교 LibreYOLO9 확률 범위가 다릅니다")
        boxes = boxes / np.asarray([size[0], size[1]] * 2, np.float32)
    else:
        value, = outputs
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != 5 + classes or not value.shape[1]:
            raise ValueError("탐지 INT8 비교 Custom CSP 출력 형상이 다릅니다")
        boxes = value[0, :, :4]
        scores = _sigmoid(value[0, :, 5:]) * _sigmoid(value[0, :, 4:5])
    if post["box_format"] == "normalized_cxcywh":
        half = boxes[:, 2:] * .5
        boxes = np.concatenate((boxes[:, :2] - half, boxes[:, :2] + half), axis=1)
    boxes = boxes.clip(0, 1)
    topk = post.get("selection") == "query_class_topk"
    if topk:
        indices, labels = np.where(scores > post["confidence_threshold"])
        confidence = scores[indices, labels]
    else:
        labels = scores.argmax(1)
        indices = np.arange(len(scores))
        confidence = scores[indices, labels]
    rows = [{"class_id": int(label), "confidence": float(score), "bbox": boxes[index].tolist()}
            for index, label, score in zip(indices, labels, confidence)
            if score > post["confidence_threshold"]
            and np.all(boxes[index, 2:] > boxes[index, :2])]
    maximum = post["max_detections"]
    if topk:
        return sorted(rows, key=lambda row: -row["confidence"])[:maximum]
    return detection_nms(rows, post["iou_threshold"], maximum)


def _predict(config, session, inputs):
    candidates = []
    expected = config["output_names"]
    for index in range(len(inputs)):
        geometry, tensor = inputs[index]
        x, y, w, h, scale_x, scale_y = geometry
        outputs = session.run(expected, {config["input_name"]: tensor})
        rows = _detections(config, outputs, (scale_x, scale_y))
        if not inputs.patches:
            return rows
        size = inputs.patches["size"]
        for row in rows:
            x1, y1, x2, y2 = np.minimum(np.asarray(row["bbox"]) * size, [w, h, w, h])
            if x2 > x1 and y2 > y1:
                row["bbox"] = [(x1+x)/inputs.width, (y1+y)/inputs.height,
                               (x2+x)/inputs.width, (y2+y)/inputs.height]
                candidates.append(row)
    post = config["postprocessing"]
    return detection_nms(candidates, post["iou_threshold"], post["max_detections"])


def _pair_detections(before, after):
    if not before or not after:
        return [], []
    a, b = (np.asarray([row["bbox"] for row in rows]) for rows in (before, after))
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                              np.maximum(a[:, None, :2], b[None, :, :2])).prod(2)
    union = (a[:, 2:] - a[:, :2]).prod(1)[:, None] + (b[:, 2:] - b[:, :2]).prod(1)[None] - intersection
    overlap = intersection / union
    classes_a, classes_b = (np.array([row["class_id"] for row in rows]) for rows in (before, after))
    allowed = (overlap >= _MATCH_IOU) & (classes_a[:, None] == classes_b[None, :])
    # Maximize the number of matched decisions before optimizing localization.
    weights = allowed * (1 + overlap / (min(len(a), len(b)) + 1))
    i, j = linear_sum_assignment(weights, maximize=True)
    i, j = i[allowed[i, j]], j[allowed[i, j]]
    differences = [abs(before[x]["confidence"] - after[y]["confidence"]) for x, y in zip(i, j)]
    return overlap[i, j].tolist(), differences


def compare_detection_sessions(config, fp32, int8, paths):
    """Compare decisions at the manifest's threshold on real input images."""
    result = {
        "scope": "FP32 vs INT8 calibration decisions; independent accuracy not measured",
        "confidence_threshold": config["postprocessing"]["confidence_threshold"],
        "match_iou": _MATCH_IOU,
        "images": len(paths), "fp32_positive_images": 0, "changed_image_decisions": 0,
        "fp32_boxes": 0, "int8_boxes": 0, "matched_boxes": 0, "lost_boxes": 0, "gained_boxes": 0,
        "per_image": [],
    }
    all_ious, all_scores = [], []
    for path in paths:
        inputs = DetectionInputs(config, path)
        before, after = (_predict(config, session, inputs) for session in (fp32, int8))
        ious, scores = _pair_detections(before, after)
        row = {
            "image": Path(path).name, "changed_image_decisions": int(bool(before) != bool(after)),
            "fp32_boxes": len(before), "int8_boxes": len(after), "matched_boxes": len(ious),
            "lost_boxes": len(before) - len(ious), "gained_boxes": len(after) - len(ious),
            "mean_matched_iou": float(np.mean(ious)) if ious else None,
            "max_matched_confidence_delta": max(scores, default=None),
        }
        result["per_image"].append(row)
        result["fp32_positive_images"] += bool(before)
        for key in ("changed_image_decisions", "fp32_boxes", "int8_boxes",
                    "matched_boxes", "lost_boxes", "gained_boxes"):
            result[key] += row[key]
        all_ious.extend(ious)
        all_scores.extend(scores)
    if not result["images"]:
        raise ValueError("탐지 INT8 비교 영상이 없습니다")
    result["mean_matched_iou"] = float(np.mean(all_ious)) if all_ious else None
    result["min_matched_iou"] = min(all_ious, default=None)
    result["max_matched_confidence_delta"] = max(all_scores, default=None)
    return result


def validate_detection_quantization(report, log):
    log("탐지 INT8 보정 비교 "
        f"· FP32 {report['fp32_boxes']}개 / INT8 {report['int8_boxes']}개 "
        f"· 누락 {report['lost_boxes']}개 / 추가 {report['gained_boxes']}개 "
        f"· 정상·불량 판정 변경 {report['changed_image_decisions']}장")
    if report["fp32_positive_images"] == 0:
        log("보정 영상에서 FP32 검출 결과가 0개였습니다. 불량 검출 일치는 확인되지 않았습니다.")
    lost, gained = report["lost_boxes"], report["gained_boxes"]
    excessive = ((lost >= 2 and lost > _MAX_UNMATCHED_FRACTION * report["fp32_boxes"])
                 or (gained >= 2 and gained > _MAX_UNMATCHED_FRACTION * report["int8_boxes"]))
    if report["changed_image_decisions"] or excessive:
        raise ValueError("INT8 변환 후 탐지 판정 차이가 기준을 넘었습니다. "
                         "FP32를 선택하거나 대표 정상·불량 보정 영상을 다시 선택하세요.")
