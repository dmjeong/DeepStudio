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


def segmentation_preview(mask, original, num_classes, *, class_names=(), details_out=None):
    """Paint predicted foreground only; labels are a separate, zoom-aware UI layer."""
    import cv2
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype.kind not in 'iu' or not mask.size:
        raise ValueError('분할 결과는 정수 2차원 마스크 필요')
    if mask.min() < 0 or mask.max() >= num_classes:
        raise ValueError('분할 결과 클래스 번호 범위 오류')
    height, width = original.shape[:2]
    labels = np.asarray(Image.fromarray(mask.astype(np.int32)).resize((width, height), Image.Resampling.NEAREST))
    overlay = np.asarray(original, dtype=np.uint8).copy()
    ids, counts = np.unique(labels, return_counts=True)
    regions, region_counts = [], {}
    kernel = np.ones((3, 3), dtype=np.uint8)
    for class_id in ids:
        class_id = int(class_id)
        if class_id == 0:  # Reserved background is not a found object.
            continue
        selected = (labels == class_id).astype(np.uint8)
        foreground = selected.astype(bool)
        color = np.asarray(class_color(class_id), dtype=np.float32)
        overlay[foreground] = (.75 * overlay[foreground] + .25 * color).astype(np.uint8)
        inner = cv2.erode(selected, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        edge = (selected != 0) & (inner == 0)
        overlay[edge] = color.astype(np.uint8)
        if details_out is not None:
            count, _, stats, _ = cv2.connectedComponentsWithStats(selected, connectivity=8)
            region_counts[class_id] = int(count - 1)
            # Limit annotation metadata, never the segmentation mask or pixel counts.
            order = np.argsort(-stats[1:, cv2.CC_STAT_AREA], kind="stable")[:256] + 1
            for index in order:
                x, y, w, h, area = map(int, stats[index])
                regions.append({"class_id": class_id,
                    "name": class_names[class_id] if class_id < len(class_names) else f"클래스 {class_id}",
                    "index": int(index), "bbox": [x, y, x + w, y + h], "area_px": area,
                    "color": "#%02x%02x%02x" % tuple(map(int, color))})
    if details_out is not None:
        regions.sort(key=lambda item: (-item["area_px"], item["class_id"], item["index"]))
        total = sum(region_counts.values())
        details_out.update(segmentation_regions=regions[:256], region_counts=region_counts,
                           region_count=total, region_labels_truncated=total > 256,
                           segmentation_display="foreground_v1")
    return np.ascontiguousarray(overlay), dict(zip(ids.tolist(), counts.tolist()))
