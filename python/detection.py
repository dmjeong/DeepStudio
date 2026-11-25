"""커스텀 검출의 공통 좌표, confidence, 클래스별 NMS 후처리."""
import numpy as np


def foreground_supervision_loss(logits, targets):
    """Train a stride-4 foreground head on the cells covered by true boxes."""
    import torch
    from torch.nn import functional as F

    _, _, height, width = logits.shape
    boxes = targets[..., 1:5].float()
    sizes = boxes[..., 2:]
    valid = (sizes > 0).all(dim=-1).to(torch.float32)
    scale = boxes.new_tensor([width, height])
    left, top = ((boxes[..., :2] - sizes / 2).clamp(0, 1) * scale).floor().long().unbind(-1)
    right, bottom = ((boxes[..., :2] + sizes / 2).clamp(0, 1) * scale).ceil().long().unbind(-1)

    # Add box corners then prefix-sum, so overlapping boxes share the same
    # foreground mask without allocating a separate feature map for each box.
    stride = width + 1
    raster = valid.new_zeros((logits.shape[0], (height + 1) * stride))
    corners = torch.cat((top * stride + left, bottom * stride + right,
                         top * stride + right, bottom * stride + left), dim=1)
    raster.scatter_add_(1, corners, torch.cat((valid, valid, -valid, -valid), dim=1))
    foreground = raster.reshape(-1, height + 1, stride).cumsum(1).cumsum(2)[:, :height, :width] > 0
    mask = foreground.to(torch.float32)
    losses = F.binary_cross_entropy_with_logits(logits[:, 0].float(), mask, reduction="none")
    background = 1 - mask
    return ((losses * mask).sum() / mask.sum().clamp_min(1)
            + (losses * background).sum() / background.sum().clamp_min(1)) * .5


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
