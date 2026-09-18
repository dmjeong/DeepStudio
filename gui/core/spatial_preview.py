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


def segmentation_preview(mask, original, num_classes):
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype.kind not in 'iu' or not mask.size:
        raise ValueError('분할 결과는 정수 2차원 마스크 필요')
    if mask.min() < 0 or mask.max() >= num_classes:
        raise ValueError('분할 결과 클래스 번호 범위 오류')
    height, width = original.shape[:2]
    labels = np.asarray(Image.fromarray(mask.astype(np.int32)).resize((width, height), Image.Resampling.NEAREST))
    palette = np.asarray([class_color(index) for index in range(num_classes)], dtype=np.float32)
    overlay = (.55 * np.asarray(original, dtype=np.float32) + .45 * palette[labels]).clip(0, 255).astype(np.uint8)
    ids, counts = np.unique(labels, return_counts=True)
    return np.ascontiguousarray(overlay), dict(zip(ids.tolist(), counts.tolist()))
