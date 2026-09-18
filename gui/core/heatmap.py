"""모델 재실행 없이 활성화 지도의 표시 범위와 투명도를 적용한다.

표시 범위는 이미지별 정규화된 활성화 강도다. 이상 탐지 판정 임계값이나
모델의 예측 점수를 바꾸지 않는다. 배열은 원본을 수정하지 않고 처리한다.
"""

import numpy as np
from PIL import Image


def _as_activation_map(activation_map: np.ndarray) -> np.ndarray:
    """비어 있지 않은 2차원 실수 지도를 검증한다."""
    values = np.asarray(activation_map)
    if values.ndim != 2 or not values.size:
        raise ValueError("활성화 지도는 비어 있지 않은 2차원 배열이어야 합니다")
    if values.dtype.kind not in "iuf":
        raise ValueError("활성화 지도는 실수 또는 정수 배열이어야 합니다")
    if not np.isfinite(values).all():
        raise ValueError("활성화 지도에 유효하지 않은 값 포함")
    return values


def normalize_activation_map(activation_map: np.ndarray) -> np.ndarray:
    """점수 지도를 이미지별 최소/최대로 정규화한다. 상수 지도는 0이다.

    PatchCore처럼 원시 점수 범위가 정해져 있지 않은 지도에 사용한다.
    Grad-CAM의 정규화된 결과를 표시할 때는 다시 정규화하지 않는다.
    """
    values = _as_activation_map(activation_map).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("활성화 지도 값을 float64로 표현할 수 없습니다")
    low, high = values.min(), values.max()
    if low == high:
        return np.zeros(values.shape, dtype=np.float32)
    # 큰 양수와 음수 사이의 차이를 구할 때 발생하는 오버플로 방지
    scale = max(abs(low), abs(high))
    scaled = values / scale
    result = (scaled - low / scale) / (high / scale - low / scale)
    return np.ascontiguousarray(np.clip(result, 0.0, 1.0), dtype=np.float32)


def reproject_classification_cam(
    activation_map: np.ndarray,
    original_shape: tuple[int, int],
    resized_shape: tuple[int, int],
    crop_shape: tuple[int, int],
    *,
    return_valid_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Resize 후 CenterCrop으로 본 영역의 CAM을 원본 좌표로 되돌린다.

    모든 shape는 실제 처리된 (높이, 너비)다. Resize 크기 인자나 추정 크기를
    전달하지 않는다. CenterCrop이 없는 경우 crop_shape와 resized_shape를
    동일하게 지정한다. 다른 기하 변환은 호출하는 쪽에서 지원 여부를 확인한다.
    패딩의 활성화는 버리고, 모델 입력에서 잘린 원본 영역은 정확히 0으로 둔다.
    return_valid_mask를 켜면 계산된 0값과 입력에서 제외된 영역을 구분하는 마스크도 반환한다.
    """
    shapes = []
    for name, shape in (("원본", original_shape), ("Resize", resized_shape),
                        ("CenterCrop", crop_shape)):
        values = np.asarray(shape)
        if (values.ndim != 1 or values.size != 2 or values.dtype.kind not in "iu"
                or np.any(values <= 0)):
            raise ValueError(f"{name} 크기는 양의 정수 (높이, 너비)여야 합니다")
        shapes.append(tuple(int(value) for value in values))
    (height, width), (resize_h, resize_w), (crop_h, crop_w) = shapes
    values = _as_activation_map(activation_map)
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError("활성화 지도는 0~1 범위로 정규화해야 합니다")
    values = values.astype(np.float32, copy=False)
    if values.shape != (crop_h, crop_w):
        values = np.asarray(
            Image.fromarray(values).resize((crop_w, crop_h), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )

    # torchvision CenterCrop: 부족한 크기는 왼쪽/위 floor, 오른쪽/아래 ceil 패딩.
    pad_top = max(crop_h - resize_h, 0) // 2
    pad_left = max(crop_w - resize_w, 0) // 2
    top = int(round((max(resize_h, crop_h) - crop_h) / 2.0)) - pad_top
    left = int(round((max(resize_w, crop_w) - crop_w) / 2.0)) - pad_left
    y_start, y_end = max(top, 0), min(top + crop_h, resize_h)
    x_start, x_end = max(left, 0), min(left + crop_w, resize_w)
    # 패딩을 먼저 제외해야 원본 가장자리에 패딩의 CAM이 보간되지 않는다.
    viewed = values[y_start - top:y_end - top, x_start - left:x_end - left]

    # 픽셀 중심 좌표로 역변환한다. 잘린 경계에서는 유효 영역의 끝값으로 보간해
    # 바깥의 가짜 0값 때문에 유효한 CAM 강도가 약해지는 현상을 피한다.
    ys = (np.arange(height, dtype=np.float64) + 0.5) * resize_h / height
    xs = (np.arange(width, dtype=np.float64) + 0.5) * resize_w / width
    valid_y = (ys >= y_start) & (ys < y_end)
    valid_x = (xs >= x_start) & (xs < x_end)
    ys = np.clip(ys - 0.5 - y_start, 0.0, viewed.shape[0] - 1)
    xs = np.clip(xs - 0.5 - x_start, 0.0, viewed.shape[1] - 1)
    y0, x0 = np.floor(ys).astype(np.intp), np.floor(xs).astype(np.intp)
    y1 = np.minimum(y0 + 1, viewed.shape[0] - 1)
    x1 = np.minimum(x0 + 1, viewed.shape[1] - 1)
    wy = (ys - y0).astype(np.float32)[:, None]
    wx = (xs - x0).astype(np.float32)[None, :]
    upper = viewed[y0[:, None], x0] * (1.0 - wx) + viewed[y0[:, None], x1] * wx
    lower = viewed[y1[:, None], x0] * (1.0 - wx) + viewed[y1[:, None], x1] * wx
    result = upper * (1.0 - wy) + lower * wy
    result[~valid_y, :] = 0.0
    result[:, ~valid_x] = 0.0
    np.clip(result, 0.0, 1.0, out=result)
    result = np.ascontiguousarray(result, dtype=np.float32)
    if return_valid_mask:
        return result, np.ascontiguousarray(valid_y[:, None] & valid_x[None, :])
    return result


def reproject_center_crop(activation_map, original_shape, crop):
    """Place the cropped model view back into its exact source-image ROI."""
    from core.paths import ensure_python_path
    ensure_python_path()
    from center_crop import center_crop_box
    height, width = original_shape
    left, top, right, bottom = center_crop_box((width, height), crop)
    values = _as_activation_map(activation_map).astype(np.float32)
    viewed = np.asarray(Image.fromarray(values).resize((right - left, bottom - top), Image.Resampling.BILINEAR))
    restored = np.zeros((height, width), dtype=np.float32)
    valid = np.zeros((height, width), dtype=bool)
    restored[top:bottom, left:right] = viewed
    valid[top:bottom, left:right] = True
    return restored, valid


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    """정규화된 강도를 기존 JET RGB 색상으로 변환한다."""
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)


def render_heatmap(
    activation_map: np.ndarray,
    original_rgb: np.ndarray,
    alpha: float = 0.5,
    lower: float = 0.0,
    upper: float = 1.0,
    *,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """정규화된 지도에서 히트맵과 원본 위 오버레이를 생성한다.

    Args:
        activation_map: 0~1 범위의 2차원 활성화 지도.
        original_rgb: (H, W, 3) RGB uint8 원본 이미지.
        alpha: 히트맵 투명도. 0은 원본, 1은 히트맵 색상.
        lower: 표시할 최소 강도. 이 값 미만은 원본 픽셀 유지.
        upper: 가장 강한 색상의 시작 강도. 초과 값은 같은 색상 사용.
        valid_mask: 원본 해상도의 bool 배열. 모델이 실제로 본 영역만 True.

    Returns:
        원본과 같은 크기의 RGB uint8 히트맵과 오버레이.
        lower=0이면 계산된 0값도 최저 강도 색상으로 표시한다.
        lower 미만 또는 valid_mask가 False인 픽셀은 원본을 유지한다.
    """
    try:
        alpha, lower, upper = float(alpha), float(lower), float(upper)
    except (TypeError, ValueError) as exc:
        raise ValueError("표시 범위와 투명도는 숫자여야 합니다") from exc
    if not np.isfinite([alpha, lower, upper]).all():
        raise ValueError("표시 범위 또는 투명도에 유효하지 않은 값 포함")
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("표시 범위는 0 <= 최소값 < 최대값 <= 1이어야 합니다")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("투명도는 0~1 범위여야 합니다")
    original = np.asarray(original_rgb)
    if (original.ndim != 3 or original.shape[2] != 3 or not original.size
            or original.dtype != np.uint8):
        raise ValueError("원본 이미지는 비어 있지 않은 RGB uint8 배열이어야 합니다")
    values = _as_activation_map(activation_map)
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError("활성화 지도는 0~1 범위로 정규화해야 합니다")
    values = values.astype(np.float32, copy=False)
    height, width = original.shape[:2]
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask)
        if valid_mask.dtype != np.bool_ or valid_mask.shape != (height, width):
            raise ValueError("유효 영역 마스크는 원본 해상도의 bool 배열이어야 합니다")
    if values.shape != (height, width):
        # float32 상태에서 보간해 색상 변환 전 양자화 손실을 피한다.
        values = np.asarray(
            Image.fromarray(values).resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
    visible = values >= lower
    if valid_mask is not None:
        visible &= valid_mask
    display_values = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    heatmap = _jet_colormap(display_values)
    heatmap[~visible] = 0
    overlay = original.copy()
    if alpha > 0.0:
        blended = alpha * heatmap[visible].astype(np.float32)
        blended += (1.0 - alpha) * original[visible].astype(np.float32)
        overlay[visible] = np.clip(blended, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(heatmap), np.ascontiguousarray(overlay)
