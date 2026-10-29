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
    """모델 해상도 클래스 마스크를 원본(판정 영역) 크기로 nearest 확대한 라벨 맵."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype.kind not in 'iu' or not mask.size:
        raise ValueError('분할 결과는 정수 2차원 마스크 필요')
    return np.asarray(Image.fromarray(mask.astype(np.int32)).resize((width, height), Image.Resampling.NEAREST))


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
    overlay = (.55 * np.asarray(original, dtype=np.float32) + .45 * palette[labels]).clip(0, 255).astype(np.uint8)
    if not tint_background:
        background = labels == 0
        overlay[background] = np.asarray(original, dtype=np.uint8)[background]
    ids, counts = np.unique(labels, return_counts=True)
    return np.ascontiguousarray(overlay), dict(zip(ids.tolist(), counts.tolist()))


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
