"""EfficientNet 외부 입력과 실제 첫 Conv의 채널 계약. Torch 없이 검증 가능."""

NATIVE_INPUT = "native"
LEGACY_GRAY_INPUT = "rgb_repeat_inside_model"
IMPLEMENTATION_VERSION = 2


def adapt_rgb_stem(weight):
    """동일 gray 3면의 Conv와 동등한 1면 가중치. 원본 가중치는 수정하지 않는다."""
    if tuple(getattr(weight, "shape", ())) != (32, 3, 3, 3):
        raise ValueError("공식 EfficientNet RGB 첫 Conv 가중치 [32,3,3,3] 필요")
    return weight[:, 0:1] + weight[:, 1:2] + weight[:, 2:3]


def model_input_contract(config, in_channels):
    if type(in_channels) is not int or in_channels not in (1, 3):
        raise ValueError("EfficientNet 입력 채널은 1 또는 3 필요")
    version = config.get("implementation_version")
    if type(version) is not int or version not in (1, IMPLEMENTATION_VERSION):
        raise ValueError("EfficientNet 모델 정의 또는 버전 불일치")
    adapter = LEGACY_GRAY_INPUT if version == 1 and in_channels == 1 else NATIVE_INPUT
    if config.get("input_adapter", adapter) != adapter:
        raise ValueError("EfficientNet 구현 버전과 입력 변환 방식 불일치")
    stem = 3 if adapter == LEGACY_GRAY_INPUT else in_channels
    for key, expected in (("in_channels", in_channels), ("stem_in_channels", stem)):
        if key in config and (type(config[key]) is not int or config[key] != expected):
            raise ValueError(f"EfficientNet {key} 메타데이터 불일치")
    return {"implementation_version": version, "input_adapter": adapter,
            "in_channels": in_channels, "stem_in_channels": stem}


def checkpoint_input_contract(checkpoint, in_channels):
    contract = model_input_contract(checkpoint.get("model_config", {}), in_channels)
    preprocessing = checkpoint.get("preprocessing") or {}
    for section in (checkpoint, preprocessing, checkpoint.get("training_config") or {}):
        if "in_channels" in section and (type(section["in_channels"]) is not int or
                                          section["in_channels"] != in_channels):
            raise ValueError("EfficientNet 체크포인트의 입력 채널 메타데이터 충돌")
    adapter = preprocessing.get("grayscale_adapter")
    if adapter is not None and adapter != contract["input_adapter"]:
        raise ValueError("EfficientNet 전처리와 모델의 입력 변환 방식 불일치")
    color = preprocessing.get("color_order")
    if color is not None and color != ("GRAY" if in_channels == 1 else "RGB"):
        raise ValueError("EfficientNet 입력 채널과 색상 순서 불일치")
    weight = checkpoint.get("model_state_dict", {}).get("features.0.0.weight")
    expected = (32, contract["stem_in_channels"], 3, 3)
    if weight is None or tuple(getattr(weight, "shape", ())) != expected:
        raise ValueError(f"EfficientNet 첫 Conv 가중치 형태 불일치: {expected} 필요")
    return contract


def channel_description(contract):
    text = f"입력 {contract['in_channels']}ch / 첫 Conv {contract['stem_in_channels']}ch"
    if contract["input_adapter"] == LEGACY_GRAY_INPUT:
        text += " (구형 RGB 확장 모델)"
    return text
