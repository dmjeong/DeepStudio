"""Review saved inference scores without changing model weights or raw results."""
from dataclasses import asdict, replace
from math import isfinite
from pathlib import Path, PureWindowsPath


def finite(value):
    try:
        return value is not None and isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def source_class(image_path, project=None):
    """Only known dataset split roots establish ground-truth classes."""
    data = (project or {}).get("data", {}) if isinstance(project, dict) else getattr(project, "data", None)
    if data is None:
        return ""
    get = data.get if isinstance(data, dict) else lambda key, default="": getattr(data, key, default)
    path_type = PureWindowsPath if "\\" in str(image_path) else Path
    image = path_type(image_path)
    roots = [get(key, "") for key in ("train_dir", "val_dir", "test_dir")]
    if get("root", ""):
        roots += [path_type(get("root")) / split for split in ("train", "val", "test")]
    for root in roots:
        if not root:
            continue
        try:
            parts = image.relative_to(path_type(root)).parts
            if len(parts) >= 2 and ".." not in parts:
                return parts[0]
        except ValueError:
            pass
    return ""


def normalized_patchcore(record):
    details = record.get("details") or {}
    return details.get("engine") == "patchcore" and details.get("score_space") == "normalized_0_1"


def classification_decision(record):
    """Read the predicted label from model output, never split a display summary."""
    details = record.get("details") or {}
    names, probabilities = details.get("class_names") or [], details.get("probabilities") or []
    if names and len(names) == len(probabilities) and all(finite(p) for p in probabilities):
        return str(names[max(range(len(probabilities)), key=lambda i: float(probabilities[i]))])
    return record.get("summary") or "대기"


def review_score(record):
    """Return the scalar score that belongs in the batch table for this task.

    Anomaly models already provide an image score. Classification stores class
    probabilities in details; detection stores per-object confidences there.
    Segmentation has no single per-image scalar score.
    """
    if record.get("status") == "error":
        return None
    task = record.get("task")
    if task == "anomaly":
        value = record.get("score")
        return float(value) if finite(value) else None
    details = record.get("details") or {}
    if task == "classify":
        probabilities = details.get("probabilities") or []
        values = [float(value) for value in probabilities if finite(value)]
        return max(values) if values else None
    if task == "detect":
        detections = details.get("detections") or []
        values = [float(item.get("confidence", item.get("score")))
                  for item in detections
                  if isinstance(item, dict) and finite(item.get("confidence", item.get("score")))]
        return max(values) if values else None
    return None


def review_score_description(task):
    return {
        "classify": "예측 클래스의 최고 확률 (0~1)",
        "detect": "검출된 객체 중 가장 높은 confidence (0~1)",
        "anomaly": "모델이 계산한 이미지 이상 점수",
        "segment": "분할에는 이미지 단위 스칼라 점수가 없습니다",
    }.get(task, "해당 작업의 점수")


def review_filter_options(rows, *, project=None, class_names=()):
    """Build choices from the whole batch, independent of active filters/pagination."""
    get = project.get if isinstance(project, dict) else lambda key, default=None: getattr(project, key, default)
    data = get("data", {})
    project_names = (data.get("class_names", []) if isinstance(data, dict)
                     else getattr(data, "class_names", []))
    output_names = [name for row in rows for name in (row.get("details") or {}).get("class_names", [])]
    model_names = list(dict.fromkeys(output_names or class_names or project_names))
    classes = list(dict.fromkeys([*project_names, *model_names, *(row.get("source_class", "") for row in rows)]))
    tasks = {row.get("task") for row in rows if row.get("task")}
    classify = "classify" in tasks if tasks else get("task", "classify") == "classify"
    decisions = list(dict.fromkeys([*(model_names if classify else []), *(row["decision"] for row in rows)]))
    return {"classes": classes, "decisions": decisions}


def review_record(record, threshold=None):
    """Return an effective display copy; status=ok means execution succeeded, not OK."""
    if threshold is not None and not finite(threshold):
        raise ValueError("임계값은 유한한 숫자여야 합니다")
    row = dict(record)
    # Older saved jobs identified by their native PatchCore heatmap retain their
    # raw JSON on disk; convert only this review copy using their saved boundary.
    if (row.get("task") == "anomaly" and row.get("status") != "error" and
            (row.get("heatmap") or {}).get("kind") == "PatchCore" and not normalized_patchcore(row)):
        from core.paths import ensure_python_path
        ensure_python_path()
        from patchcore_scores import score_normalization, normalize_score, normalize_threshold, normalized_details
        spec = score_normalization(row.get("threshold"))
        row["details"] = {**(row.get("details") or {}), **normalized_details(row["score"], row.get("threshold"), spec)}
        row["score"] = normalize_score(row["score"], spec)
        row["threshold"] = normalize_threshold(row.get("threshold"), spec)
    if normalized_patchcore(row) and row.get("status") != "error":
        for value in (row.get("score"), row.get("threshold"), threshold):
            if value is not None and (not finite(value) or not 0 <= float(value) <= 1):
                raise ValueError("PatchCore 정규화 점수와 임계값은 0~1 범위여야 합니다")
    row["filename"] = str(row.get("image_path", "")).replace("\\", "/").rsplit("/", 1)[-1]
    row["source_class"] = row.get("source_class") or ""
    row["display_score"] = review_score(row)
    row["score_description"] = review_score_description(row.get("task"))
    row["saved_threshold"] = row.get("threshold")
    if row.get("status") == "error":
        row["decision"] = "ERROR"
    elif row.get("task") == "classify":
        row["decision"] = classification_decision(row)
    elif row.get("task") == "anomaly" and finite(row.get("score")):
        value = threshold if threshold is not None else row.get("threshold")
        row["threshold"] = float(value) if finite(value) else None
        if row["threshold"] is None:
            row.update(decision="미보정", status="uncalibrated", color="#E5A832")
        else:
            ng = float(row["score"]) >= row["threshold"]
            row.update(decision="NG" if ng else "OK", status="ok", color="#E05555" if ng else "#34C759")
        row["summary"] = f"{row['decision']} {float(row['score']):.4f}"
    else:
        row["decision"] = row.get("summary") or "대기"
    return row


def effective_result(result, threshold=None):
    if threshold is None:
        return result
    row = review_record(asdict(result), threshold)
    changes = {key: row[key] for key in ("status", "summary", "color", "threshold")}
    return replace(result, **changes) if any(getattr(result, key) != value for key, value in changes.items()) else result


def restored_result(row):
    """Build a UI result after upgrading a persisted review record in memory."""
    from dataclasses import fields
    from core.inference_types import InferenceResult
    record = review_record(row)
    return InferenceResult(**{field.name: record[field.name] for field in fields(InferenceResult)
                              if field.name in record})


def review_page(records, *, threshold=None, class_name="", decision="", search="",
                selected=None, selected_only=False, sort="index", descending=False, offset=0, limit=60,
                project=None):
    """Filter/sort the complete job before pagination. Index always identifies the raw file."""
    if sort not in {"index", "source_class", "filename", "decision", "score", "inference_sec"}:
        raise ValueError("지원하지 않는 정렬 열입니다")
    rows = [review_record(row, threshold) for row in records]
    options = review_filter_options(rows, project=project)
    scores = [float(row["score"]) for row in rows if row.get("task") == "anomaly" and finite(row.get("score"))]
    thresholds = sorted({float(row["saved_threshold"]) for row in rows if row.get("task") == "anomaly" and finite(row.get("saved_threshold"))})
    scored = [row for row in rows if row.get("task") == "anomaly" and finite(row.get("score")) and row.get("status") != "error"]
    normalized = bool(scored) and all(normalized_patchcore(row) for row in scored)
    total = len(rows)
    counts = {key: sum(row["decision"] == key for row in rows) for key in ("OK", "NG", "미보정", "ERROR")}
    needle = search.casefold().strip()
    chosen = set(selected or ())
    rows = [row for row in rows if
            (not class_name or row["source_class"] == ("" if class_name == "__unknown__" else class_name)) and
            (not decision or row["decision"] == decision) and
            (not needle or needle in row["filename"].casefold()) and
            (not selected_only or row["index"] in chosen)]
    numeric = sort in {"index", "score", "inference_sec"}
    sort_value = lambda row: row.get("display_score") if sort == "score" else row.get(sort)
    valid = [r for r in rows if not numeric or finite(sort_value(r))]
    missing = [r for r in rows if numeric and not finite(sort_value(r))]
    valid.sort(key=lambda r: r["index"])
    valid.sort(key=lambda r: float(sort_value(r)) if numeric else str(r.get(sort, "")).casefold(), reverse=descending)
    rows = valid + sorted(missing, key=lambda r: r["index"])
    return {"total": total, "filtered_total": len(rows), "results": rows[offset:offset+limit],
            **options, "counts": counts, "anomaly": bool(scores),
            "score_normalized": normalized,
            "score_range": [min(scores), max(scores)] if scores else [], "saved_thresholds": thresholds}
