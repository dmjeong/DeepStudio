"""Saved 1:1 input geometry for the pinned LibreYOLO adapters."""


def input_hw(size):
    if type(size) is int:
        size = (size, size)
    if (not isinstance(size, (list, tuple)) or len(size) != 2 or
            any(type(v) is not int or v < 1 for v in size)):
        raise ValueError("모델 입력 크기는 양의 정수 높이·너비여야 합니다.")
    return tuple(size)


def checkpoint_region(checkpoint):
    region = checkpoint.get("original_size_region")
    if region is None:
        return None
    if not isinstance(region, (list, tuple)):
        raise ValueError("체크포인트의 원본 크기는 [너비, 높이]여야 합니다.")
    width, height = input_hw(region)
    if checkpoint.get("studio_patches"):
        raise ValueError("원본 크기 모델은 패치를 함께 사용할 수 없습니다.")
    from center_crop import checkpoint_center_crop
    crop = checkpoint_center_crop(checkpoint)
    if crop and (crop["width"], crop["height"]) != (width, height):
        raise ValueError("체크포인트의 원본 크기와 중앙 크롭이 다릅니다.")
    if checkpoint["task"] == "detect":
        canvas = input_hw(detector_size(checkpoint["model_family"], height, width))
        saved = (checkpoint.get("imgsz_h", checkpoint.get("imgsz")),
                 checkpoint.get("imgsz_w", checkpoint.get("imgsz")))
        if saved != canvas:
            raise ValueError("저장한 탐지 모델 입력 크기와 원본 크기가 다릅니다.")
    return (width, height)


def detector_size(family, height, width):
    """Smallest supported network canvas containing the source at one pixel per pixel."""
    height, width = input_hw((height, width))
    height, width = ((length + 31) // 32 * 32 for length in (height, width))
    if family == "yolo9":
        return height if height == width else (height, width)
    if family not in {"dfine", "rtdetrv4"}:
        raise ValueError(f"지원하지 않는 탐지 모델: {family}")
    # The native DETR trainers require square canvases and top-300 encoder queries.
    strides = (16, 32) if family == "dfine" else (8, 16, 32)
    side = max(height, width)
    while sum((side // stride) ** 2 for stride in strides) < 300:
        side += 32
    return side
