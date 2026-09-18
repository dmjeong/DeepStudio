"""OpenCV image transforms shared by training, validation and deployment."""
import warnings
import numpy as np

IMPLEMENTATION = "opencv_linear_exact_v1"


def resize_contract(preprocessing=None):
    result = dict(preprocessing or {})
    previous = result.get("resize_implementation")
    if previous in (None, "pillow_u8_bilinear"):
        result["migrated_from"] = previous or "legacy_unspecified"
        warnings.warn("기존 이미지 전처리를 OpenCV로 전환합니다. 기존 모델의 판정 결과를 재검증하세요.",
                      UserWarning, stacklevel=2)
        result.pop("antialias", None)
        result.pop("resize_implementation", None)
    expected = {"resize": "bilinear", "resize_implementation": IMPLEMENTATION,
                "interpolation": "INTER_LINEAR_EXACT", "antialias": False,
                "layout": "NCHW", "value_scale": 255.0}
    for key, value in expected.items():
        if key in result and (result[key] != value or
                              (key == "antialias" and type(result[key]) is not bool)):
            raise ValueError(f"미지원 OpenCV 전처리 계약: {key}={result[key]!r}")
    result.update(expected)
    return result


def read_rgb(path, *, exif=False):
    import cv2
    # Match the existing raw-pixel Custom/EfficientNet and EXIF-aware native
    # dataset policies. imdecode supports Unicode filenames on Windows.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8),
                         cv2.IMREAD_COLOR | (0 if exif else cv2.IMREAD_IGNORE_ORIENTATION))
    if image is None:
        raise ValueError(f"이미지 읽기 실패: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_image(path, channels=1):
    """흑백 파일은 2D uint8로 읽는다. 컬러 파일은 기존 OpenCV 색상식으로 변환한다."""
    import cv2
    if type(channels) is not int or channels not in (1, 3):
        raise ValueError("입력 채널은 1 또는 3 필요")
    if channels == 3:
        return read_rgb(path)
    # ANYDEPTH를 사용하지 않아 기존 파일 로더의 uint8 디코딩을 유지한다.
    # IMREAD_GRAYSCALE의 codec별 색상 변환 대신 기존 cvtColor 계산을 유지한다.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8),
                         cv2.IMREAD_ANYCOLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"이미지 읽기 실패: {path}")
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def require_resume_contract(preprocessing):
    if (preprocessing or {}).get("resize_implementation") != IMPLEMENTATION:
        raise ValueError("OpenCV 전처리 변경 전 체크포인트는 중단 재개할 수 없습니다. 내 가중치로 추가 학습을 선택하세요.")


def convert_channels(image, channels):
    import cv2
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("OpenCV 전처리에는 uint8 이미지 필요")
    if not image.size or (image.ndim == 3 and image.shape[2] not in (1, 3, 4)):
        raise ValueError("이미지 채널은 GRAY, RGB 또는 RGBA 필요")
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if channels == 1:
        return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if channels == 3:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB) if image.shape[2] == 4 else image
    raise ValueError("입력 채널은 1 또는 3 필요")


def resize(image, size):
    import cv2
    size = [size] if isinstance(size, int) else list(size)
    if len(size) not in (1, 2) or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError("양의 resize 크기 필요")
    image = np.asarray(image)
    height, width = image.shape[:2]
    if len(size) == 1:
        short = size[0]
        target = (short, short * height // width) if width <= height else (short * width // height, short)
    else:
        target = (size[1], size[0])
    return cv2.resize(image, target, interpolation=cv2.INTER_LINEAR_EXACT)


def center_crop(image, size):
    size = (size, size) if isinstance(size, int) else tuple(size)
    if len(size) == 1:
        size *= 2
    height, width = size
    image = np.asarray(image)
    if height < 1 or width < 1 or image.shape[0] < height or image.shape[1] < width:
        raise ValueError("중앙 크롭이 입력 이미지를 초과합니다")
    top, left = round((image.shape[0] - height) / 2), round((image.shape[1] - width) / 2)
    return np.ascontiguousarray(image[top:top + height, left:left + width])


class OpenCVResize:
    def __init__(self, size):
        self.size = size
        self.max_size = None

    def __call__(self, image):
        return resize(image, self.size)


class OpenCVCenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return center_crop(image, self.size)
