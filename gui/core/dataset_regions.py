"""이미지 안의 클래스 영역(blob·박스·폴리곤) 집계와 영역 단위 편집.

탐지·분할의 클래스는 이미지 전체가 아니라 이미지 안의 영역에 붙는다. 이 모듈은 데이터셋 화면이
영역을 세고(개수·크기), 썸네일에 겹쳐 보여주고, 목록에서 고른 영역 하나만 클래스를 바꾸거나
지울 수 있게 한다.

영역의 정의 (작업실과 같은 단위)
    분할   작업실 객체(소유자)마다 같은 클래스 픽셀의 8-연결 성분 하나.
           · 객체가 한 덩어리면 영역 1개 = 작업실 객체 1개
           · 맞닿은 두 객체는 영역 2개 (작업실도 2개)
           · 외부 PNG의 blob N개는 영역 N개 (작업실의 '연결 부분 나누기' 결과와 같음)
           배경(0)과 학습 제외(255)는 영역이 아니다.
    탐지   정답 한 줄(박스 또는 회전 박스).
    폴리곤 정답(.txt)만 있는 분할 이미지는 작업실과 같이 마스크 문서로 렌더링해 센다.
"""

import colorsys
import threading
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

# 썸네일 겹침 지도의 최대 크기 (데이터셋 격자 썸네일 132×120에 맞춘다)
OVERLAY_SIZE = (132, 120)

_CACHE = OrderedDict()
_CACHE_LIMIT = 20000
_LOCK = threading.Lock()


def class_rgb(class_id):
    """작업실 오버레이와 같은 클래스 색 (QColor.fromHsv((i*67+205)%360, 180, 255))."""
    red, green, blue = colorsys.hsv_to_rgb(((class_id * 67 + 205) % 360) / 360, 180 / 255, 1.0)
    return round(red * 255), round(green * 255), round(blue * 255)


def _file_key(path):
    stat = Path(path).stat()
    return str(Path(path).resolve()), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino


def _overlay_size(width, height, limit=OVERLAY_SIZE):
    scale = min(limit[0] / width, limit[1] / height, 1.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


def class_overlay(pixels, size, ids):
    """썸네일 크기의 클래스 지도. 축소로 사라지는 얇은 결함도 보이도록 클래스별 coverage > 0.

    2048²→132²처럼 정수배가 아닌 면적 축소는 느리다(클래스당 약 6ms). 먼저 정수배 f로 줄이고
    (OpenCV 정수배 INTER_AREA 빠른 경로, 1px도 0이 아닌 값으로 남도록 f ≤ 15), 작은 지도에서 마무리한다.
    """
    import cv2

    width, height = size
    source_h, source_w = pixels.shape
    factor = max(1, min(source_w // width, source_h // height, 15))
    result = np.zeros((height, width), np.uint8)
    for class_id in ids:
        if class_id in (0, 255):
            continue
        member = cv2.inRange(pixels, int(class_id), int(class_id))
        if factor > 1:
            pad_h, pad_w = -source_h % factor, -source_w % factor
            if pad_h or pad_w:
                member = cv2.copyMakeBorder(member, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            member = cv2.resize(member, (member.shape[1] // factor, member.shape[0] // factor),
                                interpolation=cv2.INTER_AREA)
            # 1px 결함은 여기서 값 1 정도라 다음 비정수 축소에서 0으로 반올림된다. 다시 0/255로 만든다.
            member = cv2.threshold(member, 0, 255, cv2.THRESH_BINARY)[1]
        present = cv2.resize(member, (width, height), interpolation=cv2.INTER_AREA) > 0
        result[present] = int(class_id)
    return result


def stroke_width(part):
    """영역의 굵기 px = 가장 큰 내접원의 지름 (1px 선 → 1, 3px 흠집 → 3, 5×5 점 → 5).

    비스듬한 흠집은 bbox가 커도 실제 폭은 몇 px뿐이라, 축소 후 사라지는지는 bbox가 아니라 이 값으로 본다.
    """
    import cv2

    padded = np.pad(part.astype(np.uint8), 1)
    # 정밀 마스크: 1px·3px 직선을 정확히 1·3으로 잰다 (3×3 근사 마스크는 0.91·2.82)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return max(1, int(round(2 * float(distance.max()) - 1)))


def _components(pixels, owners, class_id, document):
    """한 클래스의 영역 목록. owners가 None이면 외부 PNG(소유자 = 클래스 한 층)."""
    import cv2

    member = (pixels == class_id).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(member, connectivity=8)
    items = []
    for component in range(1, count):
        x, y, w, h, _area = (int(v) for v in stats[component])
        inside = labels[y:y + h, x:x + w] == component
        if owners is None:
            groups = [(f"base:{class_id}", inside)]
        else:
            # 맞닿은 서로 다른 객체는 작업실처럼 따로 센다
            own = owners[y:y + h, x:x + w]
            values = np.unique(own[inside])
            groups = []
            for value in values:
                part = inside & (own == value)
                if len(values) == 1:
                    groups.append((document.key(int(value)), part))
                    continue
                sub_count, sub_labels = cv2.connectedComponents(part.astype(np.uint8), connectivity=8)
                groups.extend((document.key(int(value)), sub_labels == k) for k in range(1, sub_count))
        for key, part in groups:
            ys, xs = np.nonzero(part)
            first = int(np.argmax(part.ravel()))  # 영역 안의 한 점 (편집·선택 기준)
            items.append({
                "class_id": int(class_id), "key": key, "area": int(len(xs)),
                "bbox": [x + int(xs.min()), y + int(ys.min()),
                         int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
                "width": stroke_width(part),
                "seed": [x + first % w, y + first // w],
            })
    return items


def segment_regions(project, image_path, split):
    """분할 이미지의 (픽셀 클래스 번호, 영역 목록, 이미지 크기, 썸네일 클래스 지도). 파일이 같으면 캐시."""
    from core.dataset_editor import sidecars
    from core.mask_annotations import METADATA_KEY, MaskDocument, load_mask_document

    related = sorted(sidecars(project, image_path, split), key=lambda row: str(row[1]))
    classes = max(1, len(project.data.class_names))
    key = (_file_key(image_path), tuple(_file_key(path) for _, path in related), classes)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            return cached
    masks = [path for kind, path in related if kind == "masks"]
    owners = document = None
    if len(masks) == 1:
        with Image.open(masks[0]) as mask:
            # 편집 정보가 있는 PNG만 문서로 렌더링한다 (외부 PNG는 픽셀이 곧 정답, 소유자 = 클래스 층).
            if METADATA_KEY in mask.info:
                with Image.open(image_path) as image:
                    size = image.size
                document = MaskDocument.from_image(mask, *size, classes)
            else:
                pixels = np.asarray(mask)
                if pixels.ndim != 2:
                    raise ValueError("정수 클래스 마스크 필요")
                size = (pixels.shape[1], pixels.shape[0])
    else:
        document = load_mask_document(project, image_path, split)  # 폴리곤 정답 또는 마스크 여러 개(오류)
        size = (document.width, document.height)
    if document is not None:
        document.ensure_object_ids(str(Path(image_path).resolve()))
        pixels = document.render(copy_pixels=False)
        # 원본 PNG 층만 있는 문서는 외부 PNG와 같이 다룬다 (소유자 = 클래스 층, 키 = base:클래스)
        owners = document._owners if document.shapes else None
    import cv2

    if pixels.dtype == np.uint8:
        ids = {int(i) for i in np.flatnonzero(cv2.calcHist([pixels], [0], None, [256], [0, 256]).ravel())}
    else:
        ids = {int(i) for i in np.unique(pixels)}
    ids.discard(255)
    if any(i < 0 or i >= classes for i in ids):
        raise ValueError("마스크 클래스 범위 오류")
    items = []
    for class_id in sorted(ids - {0}):
        items.extend(_components(pixels, owners, class_id, document))
    result = (tuple(sorted(ids)), tuple(items), size, class_overlay(pixels, _overlay_size(*size), sorted(ids)))
    with _LOCK:
        _CACHE[key] = result
        while len(_CACHE) > _CACHE_LIMIT:
            _CACHE.popitem(last=False)
    return result


def label_regions(rows, task, size):
    """박스(detect)·회전 박스(obb)·폴리곤 정답 줄을 영역으로. 좌표는 0~1 정규화."""
    width, height = size
    items = []
    for index, row in enumerate(rows):
        values = np.asarray(row["coordinates"], dtype=float)
        if task == "detect":
            cx, cy, w, h = values
            xs = np.array([cx - w / 2, cx + w / 2]) * width
            ys = np.array([cy - h / 2, cy + h / 2]) * height
            area = w * width * h * height
        else:
            points = values.reshape(-1, 2) * [width, height]
            xs, ys = points[:, 0], points[:, 1]
            area = abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))) / 2
        x0, y0 = max(0, int(np.floor(xs.min()))), max(0, int(np.floor(ys.min())))
        x1, y1 = min(width, int(np.ceil(xs.max()))), min(height, int(np.ceil(ys.max())))
        bbox = [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]
        if task == "detect":
            width_px = min(bbox[2], bbox[3])
        else:  # 회전 박스·폴리곤: 면적 / 긴 변 ≈ 짧은 변
            edges = np.hypot(np.diff(np.append(xs, xs[0])), np.diff(np.append(ys, ys[0])))
            width_px = area / max(1.0, float(edges.max()))
        items.append({
            "class_id": int(row["class_id"]), "key": f"row:{index}", "area": int(round(area)),
            "bbox": bbox, "width": max(1, int(round(width_px))),
            "seed": [int((x0 + x1) // 2), int((y0 + y1) // 2)],
        })
    return items


def training_scale(project, size):
    """원본 1px이 학습 입력에서 몇 px인지 (sx, sy).

    분할은 (중앙 크롭 후) 입력 크기 정사각형으로 늘려 맞추고, 탐지는 긴 변 기준으로 맞춘다.
    """
    from core.paths import ensure_python_path

    ensure_python_path()
    from center_crop import configured_center_crop

    width, height = size
    crop = configured_center_crop(project.training)
    if crop:
        width, height = min(width, crop["width"]), min(height, crop["height"])
    target = max(1, int(project.training.input_size or 1))
    if project.task == "segment":
        return target / width, target / height
    scale = target / max(width, height)
    return scale, scale


def training_footprint(item, scale):
    """학습 입력에서의 (면적 px, 굵기 px). 가로·세로 배율이 다르면 굵기는 두 배율의 기하 평균으로 줄인다."""
    sx, sy = scale
    return item["area"] * sx * sy, item.get("width", 1) * float(np.sqrt(sx * sy))


# ── 영역 단위 편집 ──────────────────────────────────────────────

def edit_regions(project, split, image_path, regions, class_id=None):
    """한 이미지의 영역들만 클래스를 바꾸거나(class_id) 지운다(None).

    탐색 뒤 정답이 바뀌었으면(영역의 점·클래스·객체·면적이 다르면) 아무것도 바꾸지 않는다.
    저장은 작업실 저장과 같은 edit_dataset 경로(백업, 실패 시 원상 복구)를 쓴다.
    """
    from core.dataset_editor import annotation_revision, edit_dataset, read_annotations, sidecars

    names = project.data.class_names
    if class_id is not None and not (0 < class_id < len(names) if project.task == "segment"
                                     else 0 <= class_id < len(names)):
        raise ValueError("바꿀 클래스 번호 오류")
    if not regions:
        raise ValueError("편집할 영역이 없습니다.")
    revision = annotation_revision(project, image_path, split)
    changed = "이미지를 탐색한 뒤 정답이 바뀌었습니다. 데이터셋을 새로 고친 뒤 다시 선택하세요."
    if project.task == "segment":
        import cv2
        from core.mask_annotations import load_mask_document

        document = load_mask_document(project, image_path, split)
        document.ensure_object_ids(str(Path(image_path).resolve()))
        pixels = document.render()
        document._ensure_cache()
        owners = document._owners.copy()
        masks = []
        for region in regions:
            x, y = region["seed"]
            if not (0 <= x < document.width and 0 <= y < document.height):
                raise ValueError(changed)
            source = int(pixels[y, x])
            owner = int(owners[y, x])
            if source != region["class_id"] or document.key(owner) != region["key"]:
                raise ValueError(changed)
            if document.objects.get(document.key(owner), {}).get("locked"):
                raise ValueError("잠긴 영역은 바꾸지 않습니다. 작업실에서 잠금을 먼저 해제하세요.")
            plane = ((pixels == source) & (owners == owner)).astype(np.uint8)
            cv2.floodFill(plane, None, (int(x), int(y)), 2, flags=8)
            mask = plane == 2
            if int(mask.sum()) != region["area"]:
                raise ValueError(changed)
            masks.append((mask, source))
        for mask, source in masks:
            if class_id is None:
                document.shapes.append(document.raster_shape(mask, 0, scope_class=source))
            else:
                document.shapes.append(document.raster_shape(
                    mask, class_id, object_id=uuid.uuid4().hex, scope_class=source))
        document.invalidate()
        return edit_dataset(project, "mask_annotations", [image_path], split=split,
                            annotations=document.payload(), expected_revision=revision)

    labels = [path for kind, path in sidecars(project, image_path, split) if kind == "labels"]
    if len(labels) != 1:
        raise ValueError(changed)
    rows = read_annotations(labels[0], project.task, len(names))
    indices = set()
    for region in regions:
        index = int(region["key"].split(":")[1]) if region["key"].startswith("row:") else -1
        if not 0 <= index < len(rows) or rows[index]["class_id"] != region["class_id"]:
            raise ValueError(changed)
        indices.add(index)
    if class_id is None:
        rows = [row for i, row in enumerate(rows) if i not in indices]
    else:
        for index in indices:
            rows[index]["class_id"] = class_id
    return edit_dataset(project, "annotations", [image_path], split=split,
                        annotations=rows, expected_revision=revision)
