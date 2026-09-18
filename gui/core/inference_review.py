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
    row["saved_threshold"] = row.get("threshold")
    if row.get("status") == "error":
        row["decision"] = "ERROR"
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
                selected=None, selected_only=False, sort="index", descending=False, offset=0, limit=60):
    """Filter/sort the complete job before pagination. Index always identifies the raw file."""
    if sort not in {"index", "source_class", "filename", "decision", "score", "inference_sec"}:
        raise ValueError("지원하지 않는 정렬 열입니다")
    rows = [review_record(row, threshold) for row in records]
    classes = sorted({row["source_class"] for row in rows})
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
    valid = [r for r in rows if not numeric or finite(r.get(sort))]
    missing = [r for r in rows if numeric and not finite(r.get(sort))]
    valid.sort(key=lambda r: r["index"])
    valid.sort(key=lambda r: float(r[sort]) if numeric else str(r.get(sort, "")).casefold(), reverse=descending)
    rows = valid + sorted(missing, key=lambda r: r["index"])
    return {"total": total, "filtered_total": len(rows), "results": rows[offset:offset+limit],
            "classes": classes, "counts": counts, "anomaly": bool(scores),
            "score_normalized": normalized,
            "score_range": [min(scores), max(scores)] if scores else [], "saved_thresholds": thresholds}
