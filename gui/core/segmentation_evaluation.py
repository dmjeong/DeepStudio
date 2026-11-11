"""Ground-truth evaluation for semantic inference, independent of model confidence."""
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
from PIL import Image

from core.metrics import SegmentationMetrics


def project_snapshot(project):
    """Only immutable evaluation metadata crosses the inference worker boundary."""
    get = project.get if isinstance(project, dict) else lambda k, d=None: getattr(project, k, d)
    data = get("data", {})
    field = data.get if isinstance(data, dict) else lambda k, d=None: getattr(data, k, d)
    return {"task": get("task", ""), "data": {
        **{key: str(field(key, "") or "") for key in ("root", "train_dir", "val_dir", "test_dir")},
        "class_names": list(field("class_names", []) or []),
    }}


def _report(metrics):
    result = metrics.compute()
    result.update(status="ok" if result["evaluable"] else "unavailable",
                  reason="" if result["evaluable"] else "평가할 정답 픽셀 없음 (모두 255)",
                  class_names=list(metrics.class_names),
                  correct_pixels=int(metrics.correct_pixels), total_pixels=int(metrics.total_pixels))
    for field in ("intersection", "union", "pred_area", "gt_area"):
        result[field] = getattr(metrics, field).tolist()
    if not result["evaluable"]:
        for field in ("mIoU", "dice_score", "pixel_accuracy"):
            result[field] = None
    return result


def evaluate_segmentation(prediction, image_path, original_shape, class_names, project=None, crop_box=None):
    """Compare at native GT resolution; absent annotations are never invented as background."""
    started = time.perf_counter()
    try:
        result = _evaluate(prediction, image_path, original_shape, class_names, project, crop_box)
    except Exception as exc:
        # Evaluation failure must not discard a valid model prediction/preview.
        result = {"status": "error", "reason": f"정답 비교 불가: {exc}"}
    result["evaluation_sec"] = time.perf_counter() - started
    result["policy"] = "semantic_native_gt_v1"
    return result


def _evaluate(prediction, image_path, original_shape, class_names, project, crop_box):
    from core.dataset_editor import sidecars
    snapshot = project_snapshot(project)
    if snapshot["task"] != "segment":
        return {"status": "unavailable", "reason": "정답 마스크 없음: 세그멘테이션 프로젝트를 열어 주세요"}
    project = SimpleNamespace(task="segment", data=SimpleNamespace(**snapshot["data"]))
    image = Path(image_path).resolve()
    split = None
    for candidate in ("train", "val", "test"):
        root = getattr(project.data, candidate + "_dir")
        if root and image.is_relative_to(Path(root).resolve()):
            split = candidate
            setattr(project.data, candidate + "_dir", str(Path(root).resolve()))
            break
    if split is None:
        return {"status": "unavailable", "reason": "정답 마스크 없음: 프로젝트 데이터셋 밖의 이미지"}
    related = sidecars(project, image, split)
    if not related:
        return {"status": "unavailable", "reason": "정답 마스크 없음 (미라벨 이미지는 평가 제외)"}
    names = list(class_names or [])
    if not names or names != project.data.class_names or len(set(names)) != len(names):
        raise ValueError("모델과 프로젝트의 클래스 이름·순서가 다릅니다. 학습 당시 프로젝트를 사용하세요")
    masks = [path for kind, path in related if kind == "masks"]
    if len(masks) > 1:
        raise ValueError("대응하는 정답 마스크가 여러 개입니다")
    if masks:
        # Saved PNG pixels are authoritative; editing metadata need not be rebuilt.
        with Image.open(masks[0]) as source:
            target = np.array(source)
    else:
        labels = [path for kind, path in related if kind == "labels"]
        if len(labels) > 1:
            raise ValueError("대응하는 폴리곤 라벨이 여러 개입니다")
        from core.mask_annotations import load_mask_document
        target = load_mask_document(project, image, split).render(copy_pixels=False)
    if target.ndim != 2 or target.shape != tuple(original_shape[:2]) or target.dtype.kind not in "iu":
        raise ValueError("원본 이미지와 같은 크기의 정수 클래스 마스크가 필요합니다")
    valid = target != 255
    if ((target[valid] < 0) | (target[valid] >= len(names))).any():
        raise ValueError("정답 마스크의 클래스 번호가 범위를 벗어났습니다")
    height, width = target.shape
    box = tuple(crop_box) if crop_box is not None else (0, 0, width, height)
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("평가 영역이 원본 이미지 범위를 벗어났습니다")
    target = target[y1:y2, x1:x2]
    pred = np.asarray(prediction)
    if (pred.ndim != 2 or not pred.size or pred.dtype.kind not in "iu"
            or pred.min() < 0 or pred.max() >= len(names)):
        raise ValueError("예측 마스크의 클래스 번호·형식 오류")
    # Match the preview/SDK pixel-centre nearest mapping. Never shrink the GT:
    # downsampling it could erase the tiny defects being evaluated.
    if pred.shape != target.shape:
        pred = np.asarray(Image.fromarray(pred.astype(np.int32)).resize(
            (target.shape[1], target.shape[0]), Image.Resampling.NEAREST))
    metrics = SegmentationMetrics(len(names), names)
    metrics.update(pred, target)
    return {**_report(metrics), "sources": [str(path) for _, path in related],
            "split": split, "crop_box": list(box), "original_size": [width, height]}


def aggregate_evaluations(evaluations):
    """Pool pixel counts, not per-image IoU; reject incompatible class contracts."""
    evaluations = list(evaluations)
    accepted = [row for row in evaluations if row.get("status") == "ok"]
    if not accepted:
        return {"status": "unavailable", "evaluated": 0, "total": len(evaluations), "per_class": []}
    names = accepted[0]["class_names"]
    if any(row["class_names"] != names for row in accepted):
        return {"status": "error", "reason": "클래스 구성이 다른 결과는 합산할 수 없습니다",
                "evaluated": 0, "total": len(evaluations), "per_class": []}
    metrics = SegmentationMetrics(len(names), names)
    for row in accepted:
        for field in ("intersection", "union", "pred_area", "gt_area"):
            getattr(metrics, field)[:] += np.asarray(row[field], dtype=np.int64)
        metrics.correct_pixels += int(row["correct_pixels"])
        metrics.total_pixels += int(row["total_pixels"])
    return {**_report(metrics), "evaluated": len(accepted), "total": len(evaluations)}


def percent(value):
    return "—" if value is None else f"{value * 100:.2f}%"


def evaluation_lines(evaluation):
    if evaluation.get("status") != "ok":
        return [evaluation.get("reason") or "정답 비교 미실행 — 다시 추론하면 평가됩니다"]
    lines = [f"mIoU {percent(evaluation['mIoU'])} · Dice {percent(evaluation['dice_score'])} · "
             f"픽셀 정확도 {percent(evaluation['pixel_accuracy'])}"]
    for row in evaluation["per_class"]:
        lines.append(f"{row['name']}: IoU {percent(row['iou'])} · Dice {percent(row['dice'])}"
                     + (" (정답 픽셀 없음 · 평균 제외)" if row["iou"] is None else ""))
    lines.append("정답이 있는 클래스 평균 (배경 포함) · 255 제외 · 선택한 추론 영역만 비교")
    return lines
