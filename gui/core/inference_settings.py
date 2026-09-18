"""화면과 CPU 도구에서 동일하게 해석하는 모델 입력 설정."""

from numbers import Integral


def input_shape(value):
    """양의 정수 또는 H, W 목록을 검증하고 두 축으로 정규화한다."""
    shape = [value] if isinstance(value, Integral) else value
    if not isinstance(shape, (list, tuple)) or len(shape) not in (1, 2):
        raise ValueError("모델 입력 크기는 양의 정수 또는 H, W 목록 필요")
    if any(isinstance(size, bool) or not isinstance(size, Integral) or size <= 0 for size in shape):
        raise ValueError("모델 입력 크기는 양의 정수 필요")
    return tuple(int(size) for size in (list(shape) * 2 if len(shape) == 1 else shape))


def require_saved_input_shape(override, saved):
    """크기 변경을 지원하지 않는 엔진에서 옵션을 조용히 무시하지 않는다."""
    if override is not None and input_shape(override) != input_shape(saved):
        raise ValueError("이 엔진의 입력 크기는 체크포인트에 저장된 크기와 일치 필요")
