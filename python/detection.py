"""커스텀 검출의 공통 좌표, confidence, 클래스별 NMS 후처리."""
import numpy as np


def postprocess_detections(predictions, confidence=0.25, iou_threshold=0.5, max_detections=300):
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 6 or not np.isfinite(values).all():
        raise ValueError('검출 출력은 유한한 (N, 5+C) 배열 필요')
    if not 0 <= confidence <= 1 or not 0 <= iou_threshold <= 1 or max_detections < 1:
        raise ValueError('검출 후처리 설정 범위 오류')
    probabilities = 1 / (1 + np.exp(-np.clip(values[:, 4:], -80, 80)))
    classes = probabilities[:, 1:].argmax(1)
    scores = probabilities[:, 0] * probabilities[np.arange(len(values)), classes + 1]
    boxes = np.concatenate((values[:, :2] - values[:, 2:4] / 2,
                            values[:, :2] + values[:, 2:4] / 2), axis=1).clip(0, 1)
    valid = (scores >= confidence) & (values[:, 2] > 0) & (values[:, 3] > 0)
    valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    order = np.flatnonzero(valid)
    order = order[np.argsort(-scores[order], kind='stable')][:30000]
    selected = []
    while len(order) and len(selected) < max_detections:
        first, rest = order[0], order[1:]
        selected.append(first)
        low = np.maximum(boxes[first, :2], boxes[rest, :2])
        high = np.minimum(boxes[first, 2:], boxes[rest, 2:])
        intersection = np.maximum(high - low, 0).prod(1)
        area_first = np.prod(boxes[first, 2:] - boxes[first, :2])
        areas = (boxes[rest, 2:] - boxes[rest, :2]).prod(1)
        overlap = intersection / np.maximum(area_first + areas - intersection, 1e-12)
        order = rest[(classes[rest] != classes[first]) | (overlap <= iou_threshold)]
    return [{'class_id': int(classes[i]), 'confidence': float(scores[i]), 'bbox': boxes[i].tolist()}
            for i in selected]


def detection_metric_records(outputs, targets, image_index):
    """검증과 최종 평가에 동일한 변환을 사용한다. AP 계산은 낮은 score부터 수집한다."""
    predictions = postprocess_detections(outputs[image_index].detach().float().cpu().numpy(), confidence=.001)
    if targets.ndim == 3:
        rows = targets[image_index]
    elif targets.ndim == 2 and targets.shape[-1] == 6:
        rows = targets[targets[:, 0] == image_index, 1:]
    else:
        raise ValueError('검출 정답 형식 오류')
    ground_truth = []
    for cls, cx, cy, width, height in rows.detach().cpu().numpy():
        if width > 0 and height > 0:
            box = np.clip([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2], 0, 1)
            ground_truth.append({'class_id': int(cls), 'bbox': box.tolist()})
    return predictions, ground_truth
