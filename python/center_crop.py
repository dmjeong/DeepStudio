"""Pixel-exact centered ROI shared by all training and inference tasks."""
from numbers import Integral


def validate_center_crop(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"width", "height"}:
        raise ValueError("중앙 크롭 설정은 가로와 세로 크기가 필요합니다")
    if any(isinstance(v, bool) or not isinstance(v, Integral) or not 1 <= v <= 65536
           for v in value.values()):
        raise ValueError("중앙 크롭 가로와 세로는 1~65536 px의 정수여야 합니다")
    return {key: int(value[key]) for key in ("width", "height")}


def configured_center_crop(config):
    # Keep the original project keys so saved PatchCore projects/drafts remain compatible.
    enabled = getattr(config, "patchcore_crop_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("중앙 크롭 사용 설정은 참/거짓이어야 합니다")
    size = validate_center_crop({"width": getattr(config, "patchcore_crop_width", 1024),
                                 "height": getattr(config, "patchcore_crop_height", 1024)})
    return size if enabled else None


def center_crop_box(image_size, crop):
    """Return PIL's half-open (left, top, right, bottom), odd remainder right/bottom."""
    width, height = image_size
    crop = validate_center_crop(crop)
    if crop is None:
        return 0, 0, width, height
    crop_w, crop_h = crop["width"], crop["height"]
    if crop_w > width or crop_h > height:
        raise ValueError(f"중앙 크롭 {crop_w}×{crop_h} px가 원본 {width}×{height} px보다 큽니다. 크롭 크기를 줄여 주세요")
    left, top = (width - crop_w) // 2, (height - crop_h) // 2
    return left, top, left + crop_w, top + crop_h


def checkpoint_center_crop(checkpoint):
    """Read standalone Custom/PatchCore/native checkpoints without project settings."""
    if not isinstance(checkpoint, dict):
        return None
    if "train_args" in checkpoint or checkpoint.get("model") is not None:
        model = checkpoint.get("ema")
        if model is None:
            model = checkpoint.get("model")
        return validate_center_crop(getattr(model, "deep_studio_center_crop", None))
    if "center_crop" in checkpoint:
        return validate_center_crop(checkpoint["center_crop"])
    preprocessing = checkpoint.get("preprocessing")
    # PatchCore stores a preprocessing version string; Custom stores a mapping.
    return validate_center_crop(preprocessing.get("center_crop") if isinstance(preprocessing, dict) else None)


def load_crop_image(path, crop=None, *, exif=False):
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source) if exif else source.copy()
        return image.crop(center_crop_box(image.size, crop)) if crop else image.copy()


def restore_detections(detections, original_size, crop):
    """Map normalized ROI xyxy boxes back to normalized full-image coordinates."""
    left, top, right, bottom = center_crop_box(original_size, crop)
    width, height = original_size
    result = []
    for item in detections:
        x1, y1, x2, y2 = [min(1., max(0., v)) for v in item["bbox"]]
        result.append({**item, "bbox": [(left + x1 * (right-left))/width,
                                        (top + y1 * (bottom-top))/height,
                                        (left + x2 * (right-left))/width,
                                        (top + y2 * (bottom-top))/height]})
    return result


def paste_crop_preview(original, preview, crop):
    import numpy as np
    from PIL import Image
    left, top, right, bottom = center_crop_box((original.shape[1], original.shape[0]), crop)
    result = original.copy()
    result[top:bottom, left:right] = np.asarray(Image.fromarray(preview).resize((right-left, bottom-top)))
    return result
