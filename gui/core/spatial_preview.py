"""검출 박스와 클래스 인덱스 마스크를 원본 해상도에 표시한다."""
import colorsys
import numpy as np
from PIL import Image, ImageDraw


def class_color(index):
    return tuple(int(value * 255) for value in colorsys.hsv_to_rgb((index * .61803398875) % 1, .75, 1))


def draw_detection_boxes(original, detections):
    image = Image.fromarray(np.asarray(original, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for item in detections:
        box = np.clip(item['bbox'], 0, 1) * [width, height, width, height]
        x1, y1, x2, y2 = np.minimum(box, [width - 1, height - 1, width - 1, height - 1]).tolist()
        color = class_color(item['class_id'])
        if "corners" in item:
            points = (np.asarray(item["corners"]) * [width, height]).tolist()
            draw.line([tuple(p) for p in points + points[:1]], fill=color, width=max(1, min(width, height) // 200))
        else:
            draw.rectangle((x1, y1, x2, y2), outline=color, width=max(1, min(width, height) // 200))
        text = f"C{item['class_id']} {item['confidence']:.0%}"
        y = max(0, y1 - 13)
        draw.rectangle(draw.textbbox((x1, y), text), fill=(0, 0, 0))
        draw.text((x1, y), text, fill=color)
    return np.ascontiguousarray(image)


def segmentation_labels(mask, width, height):
    """모델 해상도 클래스 마스크를 원본(판정 영역) 크기로 nearest 확대한 라벨 맵.

    PIL NEAREST와 같은 픽셀 대응을 np.take로 만든다. 클래스가 256개 미만이면 uint8로 확대해
    12MP에서 약 100ms → 수십 ms로 줄인다.
    """
    from core.segment_defects import upsample_labels
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype.kind not in 'iu' or not mask.size:
        raise ValueError('분할 결과는 정수 2차원 마스크 필요')
    if mask.min() >= 0 and mask.max() < 256:
        mask = mask.astype(np.uint8)
    return upsample_labels(mask, width, height)


def segmentation_preview(mask, original, num_classes, labels=None, tint_background=True):
    """클래스 색을 원본에 45% 섞은 미리보기와 클래스별 픽셀 수.

    tint_background=False면 배경(0) 픽셀은 원본 그대로 두고 결함 클래스만 칠한다(결함 판정 화면).
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype.kind not in 'iu' or not mask.size:
        raise ValueError('분할 결과는 정수 2차원 마스크 필요')
    if mask.min() < 0 or mask.max() >= num_classes:
        raise ValueError('분할 결과 클래스 번호 범위 오류')
    height, width = original.shape[:2]
    if labels is None:
        labels = segmentation_labels(mask, width, height)
    palette = np.asarray([class_color(index) for index in range(num_classes)], dtype=np.float32)
    if tint_background:
        overlay = (.55 * np.asarray(original, dtype=np.float32) + .45 * palette[labels]).clip(0, 255).astype(np.uint8)
    else:
        overlay = _tint_defects(original, labels, palette)
    from core.segment_defects import _histogram
    histogram = _histogram(labels)  # np.unique는 전체 정렬이라 대형 이미지에서 느리다
    counts = {int(i): int(histogram[i]) for i in np.flatnonzero(histogram)}
    return np.ascontiguousarray(overlay), counts


def _tint_defects(original, labels, palette):
    """결함 판정 화면: 배경은 원본 그대로, 결함 픽셀만 클래스 색을 45% 섞는다.

    결함이 있는 행·열 범위만 다루고(드문 결함은 거의 공짜), 그 안에서는 OpenCV 마스크 연산을 쓴다.
    결함이 화면을 덮는 경우에도 12MP에서 약 0.1초 (전체 float 계산은 약 0.5초). 색 차이는 반올림 1 이하.
    """
    import cv2
    from core.segment_defects import _band
    overlay = np.array(original, dtype=np.uint8, copy=True)
    band = _band(labels != 0)
    if band is None:
        return overlay
    y0, y1, x0, x1 = band
    window = np.ascontiguousarray(overlay[y0:y1, x0:x1])
    classes = labels[y0:y1, x0:x1]
    if classes.dtype != np.uint8 or window.ndim != 3 or window.shape[2] != 3:
        foreground = np.nonzero(classes)  # 256개 이상 클래스 등 드문 경우: numpy로
        mixed = .55 * window[foreground].astype(np.float32) + .45 * palette[classes[foreground]]
        window[foreground] = mixed.clip(0, 255).astype(np.uint8)
    else:
        classes = np.ascontiguousarray(classes)
        scaled = cv2.convertScaleAbs(window, alpha=.55)
        for class_id in np.flatnonzero(cv2.calcHist([classes], [0], None, [256], [0, 256]).ravel()):
            if class_id == 0:
                continue
            color = tuple(float(.45 * value) for value in palette[class_id]) + (0.0,)
            cv2.copyTo(cv2.add(scaled, color), cv2.compare(classes, int(class_id), cv2.CMP_EQ), window)
    overlay[y0:y1, x0:x1] = window
    return overlay


def draw_defect_boxes(original, defects, missed=(), selected=None):
    """분할 결함마다 클래스 색 사각형과 '이름 점수' 라벨을 그린다.

    defects: 판정 기준을 통과한 결함, missed: 정답에는 있는데 못 찾은 결함(흰색 'MISS'),
    selected: 표에서 고른 결함 id(노란 굵은 선). 작은 결함도 보이도록 사각형을 2px 키운다.
    """
    image = Image.fromarray(np.asarray(original, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line = max(1, min(width, height) // 300)

    def name(item):
        # PIL 기본 글꼴은 한글을 그리지 못하므로 ASCII가 아닌 이름은 클래스 번호로 쓴다(표에는 이름).
        return item["class_name"] if str(item["class_name"]).isascii() else f"C{item['class_id']}"

    def box(item, color, text, thick):
        x1, y1 = max(0, item["x"] - 2), max(0, item["y"] - 2)
        x2, y2 = min(width - 1, item["x"] + item["w"] + 1), min(height - 1, item["y"] + item["h"] + 1)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=thick)
        y = max(0, y1 - 13)
        draw.rectangle(draw.textbbox((x1, y), text), fill=(0, 0, 0))
        draw.text((x1, y), text, fill=color)

    for item in missed:
        box(item, (255, 255, 255), f"MISS {name(item)}", line)
    for item in defects:
        if item["id"] == selected:
            continue
        score = "" if item.get("score") is None else f" {item['score']:.2f}"
        box(item, class_color(item["class_id"]), f"{name(item)}{score}", line)
    for item in defects:
        if item["id"] == selected:  # 선택한 결함은 마지막에 그려 다른 사각형에 가려지지 않게 한다
            score = "" if item.get("score") is None else f" {item['score']:.2f}"
            box(item, (255, 214, 10), f"{name(item)}{score}", line * 3)
    return np.ascontiguousarray(image)
