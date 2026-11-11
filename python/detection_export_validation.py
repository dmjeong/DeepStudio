"""Compare DETR detection sets; query indices are not object identities."""
import numpy as np
from scipy.optimize import linear_sum_assignment

CONFIDENCE_FLOOR = .25


def validate_detection_sets(expected, actual, *, atol=.001, rtol=.001):
    from export_onnx import validate_outputs
    left_logits, left_boxes = (np.asarray(v) for v in expected)
    right_logits, right_boxes = (np.asarray(v) for v in actual)
    if (left_logits.shape != right_logits.shape or left_boxes.shape != right_boxes.shape or
            left_logits.ndim != 3 or left_boxes.shape != (*left_logits.shape[:2], 4) or
            not all(np.isfinite(v).all() for v in (left_logits, left_boxes, right_logits, right_boxes))):
        raise ValueError("DETR 검증: 출력 크기 또는 NaN/Inf 오류")
    for batch in range(len(left_logits)):
        lb, rb = left_boxes[batch], right_boxes[batch]
        ll, rl = left_logits[batch], right_logits[batch]
        cost = np.abs(lb[:, None] - rb[None]).max(2)
        rows, columns = linear_sum_assignment(cost)
        try:
            validate_outputs(lb[rows], rb[columns], atol=atol, rtol=rtol)
            validate_outputs(ll[rows], rl[columns], atol=atol, rtol=rtol)
            continue
        except ValueError:
            pass
        # Encoder TopK can choose different below-threshold queries on a tie.
        # Compare all deployable detections and bind the export to this floor.
        def selected(logits, boxes):
            probs = 1. / (1. + np.exp(-np.clip(logits.astype(np.float64), -80, 80)))
            flat = probs.ravel()
            idx = np.argsort(-flat, kind="stable")[:300]
            idx = idx[flat[idx] > CONFIDENCE_FLOOR]
            return boxes[idx // logits.shape[1]], idx % logits.shape[1], flat[idx]
        a, ac, ap = selected(ll, lb)
        b, bc, bp = selected(rl, rb)
        if len(a) != len(b):
            raise ValueError("DETR 검증: 유효 검출 개수가 달라졌습니다")
        if not len(a):
            continue
        cost = np.abs(a[:, None] - b[None]).max(2) + np.abs(ap[:, None] - bp[None]) + (ac[:, None] != bc[None]) * 1e6
        rows, columns = linear_sum_assignment(cost)
        if not np.array_equal(ac[rows], bc[columns]):
            raise ValueError("DETR 검증: 검출 클래스가 달라졌습니다")
        validate_outputs(a[rows], b[columns], atol=atol, rtol=rtol)
        validate_outputs(ap[rows], bp[columns], atol=atol, rtol=rtol)
    return True
