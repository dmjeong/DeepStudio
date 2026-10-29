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
    """PIL NEAREST 확대와 같은 대응: 출력 픽셀 중심이 들어가는 입력 픽셀 번호."""
    return np.minimum(((np.arange(size) + 0.5) * source / size).astype(np.int64), source - 1)


def _name(class_names, class_id):
    return str(class_names[class_id]) if 0 <= class_id < len(class_names) else f"Class {class_id}"


def analyze_segmentation(labels, probabilities=None, *, class_names=(), offset=(0, 0), truth=None,
                         num_classes=None, max_defects=MAX_DEFECTS):
    """예측 라벨 맵(원본 해상도)을 결함 목록으로 바꾸고, 정답이 있으면 비교한다.

    labels: (H, W) 정수 클래스 맵. 0은 배경이라 결함이 아니다.
    probabilities: (C, h, w) 모델 해상도 softmax. 결함 점수 = 그 결함 픽셀의 해당 클래스 평균 확률.
    offset: 중앙 크롭일 때 원본 이미지 좌표로 옮길 (x, y).
    truth: labels와 같은 크기의 정답 맵(255 = 학습 제외, 비교에서 뺀다). 없으면 None.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError("분할 결과는 2차원 클래스 맵이어야 합니다")
    height, width = labels.shape
    classes = int(num_classes or max(len(class_names), int(labels.max(initial=0)) + 1))
    ox, oy = int(offset[0]), int(offset[1])
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=np.float32)
        rows = _nearest_index(height, probabilities.shape[1])
        cols = _nearest_index(width, probabilities.shape[2])
    if truth is not None:
        truth = np.asarray(truth)
        if truth.shape != labels.shape:
            raise ValueError("정답 마스크와 예측 크기가 다릅니다")
        valid = truth != 255
        if np.any(truth[valid] >= classes):
            raise ValueError("정답 마스크에 모델 클래스 범위를 벗어난 값이 있습니다")
    defects, truth_defects, pairs = [], [], []
    for class_id in range(1, classes):
        predicted = labels == class_id
        expected = (truth == class_id) if truth is not None else None
        if not predicted.any() and (expected is None or not expected.any()):
            continue
        count, components, stats = _components(predicted)
        scores = None
        if count > 1 and probabilities is not None and class_id < probabilities.shape[0]:
            # 모델 해상도 확률을 원본 해상도로 nearest 대응시켜 성분별 평균을 한 번에 구한다.
            full = probabilities[class_id][rows[:, None], cols[None, :]]
            sums = np.bincount(components.ravel(), weights=full.ravel(), minlength=count)
            scores = sums / np.maximum(stats[:, 4], 1)
        first = len(defects)
        for index in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[index])
            defects.append({"id": len(defects), "class_id": class_id, "class_name": _name(class_names, class_id),
                            "x": x + ox, "y": y + oy, "w": w, "h": h, "area": area,
                            "score": None if scores is None else round(float(scores[index]), 4),
                            "truth": [], "_component": index})
        if expected is None:
            continue
        truth_count, truth_components, truth_stats = _components(expected & valid)
        base = len(truth_defects)
        for index in range(1, truth_count):
            x, y, w, h, area = (int(v) for v in truth_stats[index])
            truth_defects.append({"id": len(truth_defects), "class_id": class_id,
                                  "class_name": _name(class_names, class_id),
                                  "x": x + ox, "y": y + oy, "w": w, "h": h, "area": area, "predicted": []})
        # 같은 클래스의 예측 결함과 정답 결함이 한 픽셀이라도 겹치면 짝이다.
        both = (components > 0) & (truth_components > 0)
        if both.any():
            keys = np.unique(components[both].astype(np.int64) * truth_count + truth_components[both])
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
        item = {key: value for key, value in defects[old].items() if key != "_component"}
        item["id"] = renumber[old]
        result.append(item)
    for item in truth_defects:
        item["predicted"] = sorted(renumber[i] for i in item["predicted"] if i in renumber)
    analysis = {"defects": result, "defects_dropped": len(defects) - len(result)}
    if truth is not None:
        flat = truth[valid].astype(np.int64) * classes + labels[valid].astype(np.int64)
        confusion = np.bincount(flat, minlength=classes * classes).reshape(classes, classes)
        analysis["ground_truth"] = {"confusion": confusion.tolist(), "defects": truth_defects,
                                    "ignored_pixels": int((~valid).sum())}
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
        truth = np.asarray(source).astype(np.int64)
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
