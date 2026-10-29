"""분할 추론 결과를 결함 단위로 바꾸고 OK/NG를 판정한다. Qt와 학습 엔진에 의존하지 않는다.

픽셀 비율("클래스 1: 0.8%")만으로는 큰 결함 하나와 작은 점 100개를 구분할 수 없다.
검사 프로그램처럼 예측 마스크를 결함(같은 클래스 픽셀의 8-연결 덩어리)으로 나눠
클래스 · 위치 박스 · 면적 · 점수를 보고, 판정 기준(최소 면적·최소 점수)으로 OK/NG를 정한다.

    예측 마스크 ──▶ 클래스별 연결 성분 ──▶ 결함 목록 [클래스, x, y, w, h, 면적, 점수]
         │                                       │
         │                                       ▼
         │                         판정 기준(최소 면적·점수) ──▶ OK / NG
         ▼
    정답 마스크가 있으면 ──▶ 픽셀 혼동 행렬(클래스별 IoU·Dice) + 결함 짝(검출 / 놓침 / 오검출)

모든 값은 JSON으로 저장되므로(추론 기록·프로세스 작업) 리스트와 기본형만 쓴다.
판정은 저장된 결함 목록으로 다시 계산하므로 기준을 바꿔도 다시 추론하지 않는다.
"""

from pathlib import Path

import numpy as np

# 노이즈 점이 수만 개 나와도 결과 파일과 표가 감당할 수 있도록 큰 결함부터 이만큼만 남긴다.
MAX_DEFECTS = 1000
MASK_EXTENSIONS = (".png", ".bmp", ".tif", ".tiff")
DEFAULT_RULES = {"min_area": 1, "min_score": 0.0}


def _components(binary):
    """8-연결 성분: (개수, 성분 번호 맵, stats[x, y, w, h, area])."""
    import cv2
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    return count, labels, stats


def _nearest_index(size, source):
    """PIL NEAREST 확대와 같은 대응: 출력 픽셀 i가 가져오는 입력 픽셀 번호.

    식으로 흉내 내면 반올림이 한 픽셀씩 어긋나는 크기가 있어, 한 줄짜리 번호 이미지를 PIL로 직접 확대해 얻는다.
    """
    from PIL import Image
    index = Image.fromarray(np.arange(source, dtype=np.int32)[None, :])
    return np.asarray(index.resize((int(size), 1), Image.Resampling.NEAREST), dtype=np.int64)[0]


def upsample_labels(mask, width, height):
    """모델 해상도 클래스 맵을 (width, height)로 nearest 확대. PIL NEAREST와 같은 결과를 np.take로 만든다."""
    mask = np.asarray(mask)
    rows = _nearest_index(height, mask.shape[0])
    cols = _nearest_index(width, mask.shape[1])
    return np.take(np.take(mask, rows, axis=0), cols, axis=1)


def _histogram(values):
    """정수 맵의 값별 개수(길이 256 이상). uint8은 OpenCV로 센다 (np.bincount는 int64 변환에 12MP에서 0.4초)."""
    values = np.asarray(values)
    if values.dtype == np.uint8:
        import cv2
        return cv2.calcHist([np.ascontiguousarray(values)], [0], None, [256], [0, 256]).ravel().astype(np.int64)
    return np.bincount(values.ravel().astype(np.int64), minlength=256)


def _band(binary):
    """참 픽셀을 모두 담는 행·열 범위 (y0, y1, x0, x1). 없으면 None. 큰 이미지에서 작은 결함만 다루기 위함."""
    rows = np.flatnonzero(binary.any(axis=1))
    if not rows.size:
        return None
    cols = np.flatnonzero(binary[rows[0]:rows[-1] + 1].any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _name(class_names, class_id):
    return str(class_names[class_id]) if 0 <= class_id < len(class_names) else f"Class {class_id}"


def analyze_segmentation(mask, probabilities=None, *, size=None, labels=None, class_names=(), offset=(0, 0),
                         truth=None, num_classes=None, max_defects=MAX_DEFECTS):
    """예측 마스크를 결함 목록으로 바꾸고, 정답이 있으면 비교한다.

    mask: 모델 해상도 (h, w) 정수 클래스 맵. 0은 배경이라 결함이 아니다.
    probabilities: mask와 같은 해상도의 (C, h, w) softmax. 결함 점수 = 결함 픽셀의 그 클래스 평균 확률.
    size: 결함 좌표를 낼 원본(판정 영역) 크기 (W, H). 없으면 mask 크기.
    labels: 이미 확대해 둔 원본 크기 라벨 맵(정답 비교에만 쓴다. 없으면 만든다).
    offset: 중앙 크롭일 때 원본 이미지 좌표로 옮길 (x, y).
    truth: 원본(판정 영역) 크기 정답 맵(255 = 학습 제외, 비교에서 뺀다). 없으면 None.

    속도: 결함은 모델 해상도(예: 512²)에서 찾고 원본 좌표로 옮긴다. nearest 확대에서 모델 픽셀 하나는
    원본의 연속한 직사각형 블록이 되므로 연결 성분·박스·면적·평균 점수가 원본에서 계산한 값과 같다.

        모델 픽셀 열 x ──▶ 원본 열 [start_x[x], start_x[x] + col_w[x])   (col_w = 그 열로 가는 원본 열 수)
        면적 = Σ 성분 픽셀의 row_h × col_w,  점수 = Σ 확률 × 블록 면적 / 면적

    12MP 이미지에서 원본 해상도 계산은 약 0.4초였고, 이 방식은 수 ms다.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("분할 결과는 2차원 클래스 맵이어야 합니다")
    height, width = mask.shape
    out_w, out_h = (int(size[0]), int(size[1])) if size else (width, height)
    classes = int(num_classes or max(len(class_names), int(mask.max(initial=0)) + 1))
    ox, oy = int(offset[0]), int(offset[1])
    rows, cols = _nearest_index(out_h, height), _nearest_index(out_w, width)
    row_h, col_w = np.bincount(rows, minlength=height), np.bincount(cols, minlength=width)
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=np.float32)
        if probabilities.shape[1:] != mask.shape:
            raise ValueError("확률 맵과 마스크 해상도가 다릅니다")
    if (row_h == 0).any() or (col_w == 0).any():
        # 원본이 모델 입력보다 작으면(축소) 사라지는 모델 픽셀이 있어 원본 해상도에서 그대로 센다 (작아서 빠르다).
        mask = labels if labels is not None else upsample_labels(mask, out_w, out_h)
        if probabilities is not None:
            probabilities = np.take(np.take(probabilities, rows, axis=1), cols, axis=2)
        labels = mask
        rows, cols = np.arange(out_h), np.arange(out_w)
        row_h, col_w = np.ones(out_h, np.int64), np.ones(out_w, np.int64)
    start_y = np.concatenate([[0], np.cumsum(row_h)[:-1]])
    start_x = np.concatenate([[0], np.cumsum(col_w)[:-1]])
    pixel_area = (row_h[:, None] * col_w[None, :]).astype(np.float64)  # 모델 픽셀 하나가 덮는 원본 픽셀 수
    histogram = None
    if truth is not None:
        truth = np.asarray(truth)
        if truth.shape != (out_h, out_w):
            raise ValueError("정답 마스크와 예측 크기가 다릅니다")
        if labels is None:
            labels = upsample_labels(mask, out_w, out_h)
        histogram = _histogram(truth)
        if histogram[classes:255].any() or len(histogram) > 256:
            raise ValueError("정답 마스크에 모델 클래스 범위를 벗어난 값이 있습니다")
    defects, truth_defects, pairs = [], [], []
    for class_id in range(1, classes):
        predicted = mask == class_id
        in_truth = histogram is not None and histogram[class_id] > 0
        if not predicted.any() and not in_truth:
            continue
        count, components, stats = _components(predicted)
        if count > 1:
            flat = components.ravel()
            areas = np.bincount(flat, weights=pixel_area.ravel(), minlength=count)
            scores = None
            if probabilities is not None and class_id < probabilities.shape[0]:
                weighted = (probabilities[class_id] * pixel_area).ravel()
                scores = np.bincount(flat, weights=weighted, minlength=count) / np.maximum(areas, 1)
        first = len(defects)
        for index in range(1, count):
            x, y, w, h = (int(v) for v in stats[index][:4])
            left, top = int(start_x[x]), int(start_y[y])
            defects.append({"id": len(defects), "class_id": class_id, "class_name": _name(class_names, class_id),
                            "x": left + ox, "y": top + oy,
                            "w": int(start_x[x + w - 1] + col_w[x + w - 1]) - left,
                            "h": int(start_y[y + h - 1] + row_h[y + h - 1]) - top,
                            "area": int(round(areas[index])),
                            "score": None if scores is None else round(float(scores[index]), 4),
                            "truth": []})
        if truth is None:
            continue
        if not in_truth:
            continue
        # 정답 결함은 그 클래스 픽셀이 있는 범위만 잘라 센다 (원본 전체 연결 성분 계산을 피한다).
        expected = truth == class_id
        y0, y1, x0, x1 = _band(expected)
        expected = expected[y0:y1, x0:x1]
        truth_count, truth_components, truth_stats = _components(expected)
        base = len(truth_defects)
        for index in range(1, truth_count):
            x, y, w, h, area = (int(v) for v in truth_stats[index])
            truth_defects.append({"id": len(truth_defects), "class_id": class_id,
                                  "class_name": _name(class_names, class_id),
                                  "x": x + x0 + ox, "y": y + y0 + oy, "w": w, "h": h, "area": area, "predicted": []})
        if count > 1 and truth_count > 1:
            # 같은 클래스의 예측 결함과 정답 결함이 한 픽셀이라도 겹치면 짝이다 (같은 범위의 원본 해상도에서 비교).
            full = np.take(np.take(components, rows[y0:y1], axis=0), cols[x0:x1], axis=1)
            both = (full > 0) & (truth_components > 0)
            if both.any():
                keys = np.unique(full[both].astype(np.int64) * truth_count + truth_components[both])
                pairs.extend((first + int(k // truth_count) - 1, base + int(k % truth_count) - 1) for k in keys)
    for predicted_id, truth_id in pairs:
        defects[predicted_id]["truth"].append(truth_id)
        truth_defects[truth_id]["predicted"].append(predicted_id)
    # 큰 결함부터 남기고 번호를 다시 매긴다 (짝 정보도 같이 옮긴다).
    order = sorted(range(len(defects)), key=lambda i: (-defects[i]["area"], i))
    kept = order[:max_defects]
    renumber = {old: new for new, old in enumerate(kept)}
    result = []
    for old in kept:
        item = dict(defects[old])
        item["id"] = renumber[old]
        result.append(item)
    for item in truth_defects:
        item["predicted"] = sorted(renumber[i] for i in item["predicted"] if i in renumber)
    analysis = {"defects": result, "defects_dropped": len(defects) - len(result)}
    if truth is not None:
        # 배경-배경 픽셀이 대부분이므로 결함(예측 또는 정답)이 있는 픽셀만 세고, [0, 0]은 나머지로 채운다.
        foreground = np.flatnonzero(((labels != 0) | (truth != 0)).ravel())
        truth_values = truth.ravel()[foreground].astype(np.int64)
        predicted_values = labels.ravel()[foreground].astype(np.int64)
        keep = truth_values != 255  # 학습 제외 픽셀은 비교하지 않는다
        flat = truth_values[keep] * classes + predicted_values[keep]
        confusion = np.bincount(flat, minlength=classes * classes).reshape(classes, classes)
        confusion[0, 0] = truth.size - int(histogram[255]) - int(confusion.sum())
        analysis["ground_truth"] = {"confusion": confusion.tolist(), "defects": truth_defects,
                                    "ignored_pixels": int(histogram[255])}
    return analysis


def class_iou(confusion, class_names=()):
    """혼동 행렬(정답 행 × 예측 열)에서 클래스별 IoU·Dice. 배경(0)은 제외하고 정답이나 예측이 있는 클래스만."""
    confusion = np.asarray(confusion, dtype=np.int64)
    rows = []
    for class_id in range(1, confusion.shape[0]):
        intersection = int(confusion[class_id, class_id])
        truth = int(confusion[class_id].sum())
        predicted = int(confusion[:, class_id].sum())
        union = truth + predicted - intersection
        if union == 0:
            continue
        rows.append({"class_id": class_id, "class_name": _name(class_names, class_id),
                     "iou": intersection / union, "dice": 2 * intersection / (truth + predicted),
                     "truth_pixels": truth, "predicted_pixels": predicted})
    return rows


def rules_of(details, rules=None):
    """사용자가 바꾼 기준이 있으면 그것, 없으면 추론 때 저장한 기준."""
    merged = dict(DEFAULT_RULES)
    merged.update((details or {}).get("defect_rules") or {})
    merged.update(rules or {})
    return {"min_area": max(1, int(merged["min_area"])), "min_score": float(merged["min_score"])}


def judge(details, rules=None):
    """저장된 결함 목록을 판정 기준으로 거른다. 결함이 하나라도 남으면 NG.

    정답 비교가 있으면 같은 기준으로 결함 단위 검출/놓침/오검출을 센다(픽셀 IoU는 기준과 무관).
    """
    details = details or {}
    rules = rules_of(details, rules)
    kept = [item for item in details.get("defects") or []
            if item["area"] >= rules["min_area"]
            and (item.get("score") is None or item["score"] >= rules["min_score"])]
    counts = {}
    for item in kept:
        counts[item["class_name"]] = counts.get(item["class_name"], 0) + 1
    verdict = {"decision": "NG" if kept else "OK", "kept": kept, "counts": counts, "rules": rules,
               "max_score": max((item["score"] for item in kept if item.get("score") is not None), default=None)}
    truth = details.get("ground_truth")
    if truth:
        kept_ids = {item["id"] for item in kept}
        truth_defects = truth.get("defects") or []
        found = sum(any(i in kept_ids for i in item["predicted"]) for item in truth_defects)
        verdict.update(truth_defects=len(truth_defects), found=found, missed=len(truth_defects) - found,
                       false_positives=sum(not item["truth"] for item in kept),
                       truth_decision="NG" if truth_defects else "OK",
                       per_class=class_iou(truth["confusion"], details.get("class_names") or ()))
    return verdict


def summary_text(verdict):
    """결과 표·격자에 쓰는 한 줄: 'NG 3' / 'OK'."""
    return f"NG {len(verdict['kept'])}" if verdict["decision"] == "NG" else "OK"


def batch_truth_summary(details_list, rules=None, class_names=()):
    """정답이 있는 이미지들의 픽셀 혼동 행렬을 합쳐 데이터셋 기준 IoU와 결함·판정 정확도를 낸다."""
    total, images = None, 0
    truth_defects = found = false_positives = correct = 0
    for details in details_list:
        truth = (details or {}).get("ground_truth")
        if not truth:
            continue
        confusion = np.asarray(truth["confusion"], dtype=np.int64)
        if total is None:
            total = confusion
        elif total.shape == confusion.shape:
            total = total + confusion
        else:
            continue
        verdict = judge(details, rules)
        images += 1
        truth_defects += verdict["truth_defects"]
        found += verdict["found"]
        false_positives += verdict["false_positives"]
        correct += verdict["decision"] == verdict["truth_decision"]
    if not images:
        return None
    per_class = class_iou(total, class_names)
    return {"images": images, "per_class": per_class,
            "miou": float(np.mean([row["iou"] for row in per_class])) if per_class else None,
            "truth_defects": truth_defects, "found": found, "missed": truth_defects - found,
            "false_positives": false_positives, "decision_accuracy": correct / images}


def truth_mask_path(image_path, split_dirs):
    """데이터셋 이미지의 정답 마스크 경로. split_dirs = [(이미지 폴더, 마스크 폴더), ...]. 없으면 None.

    학습 로더와 같은 짝 규칙(같은 하위 폴더 · 같은 파일 이름 · 손실 없는 마스크 확장자)을 쓴다.
    """
    image = Path(image_path).resolve()
    for image_dir, mask_dir in split_dirs:
        try:
            relative = image.relative_to(Path(image_dir).resolve())
        except ValueError:
            continue
        for extension in MASK_EXTENSIONS:
            candidate = Path(mask_dir) / relative.parent / (relative.stem + extension)
            if candidate.is_file():
                return str(candidate)
        return None
    return None


def load_truth(path, size, box=None):
    """정답 마스크를 읽어 (크롭한) 정수 맵으로. 크기가 원본 이미지와 다르면 비교하지 않는다(None)."""
    from PIL import Image
    with Image.open(path) as source:
        if source.mode not in ("L", "P", "I", "I;16", "I;16L", "I;16B"):
            return None
        if source.size != tuple(size):
            return None
        truth = np.asarray(source)  # uint8 그대로 (정수 변환은 비교할 때 필요한 만큼만)
    if box is not None:
        truth = truth[box[1]:box[3], box[0]:box[2]]
    return truth


def segment_split_dirs(project):
    """프로젝트의 (이미지 폴더, 마스크 폴더) 목록. 학습 로더와 같은 images/{split} ↔ masks/{split} 규칙."""
    data = getattr(project, "data", None)
    root = getattr(data, "root", "") if data is not None else ""
    if not root:
        return []
    result = []
    for split in ("train", "val", "test"):
        image_dir = Path(getattr(data, f"{split}_dir", "") or Path(root) / "images" / split)
        mask_dir = (image_dir.parent.parent / "masks" / image_dir.name if image_dir.parent.name == "images"
                    else Path(root) / "masks" / split)
        result.append([str(image_dir), str(mask_dir)])
    return result
