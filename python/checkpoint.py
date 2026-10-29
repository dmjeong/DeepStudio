"""모델을 독립적으로 복원하기 위한 체크포인트 메타데이터."""


def tiling_contract(patch_size, scale=1.0):
    """패치 학습 모델의 추론 규칙: 원본 해상도에서 tile×tile 창을 overlap만큼 겹쳐 민다.

    겹침은 타일의 1/4. 각 타일은 이미지 안쪽으로 난 가장자리 overlap/2 띠를 거의 믿지 않고(가중치 1e-4),
    이웃 타일의 안쪽이 그 자리를 채운다 (blend = center_weighted_softmax). None이면 전체 이미지 리사이즈 모델.
    """
    if not patch_size:
        return None
    tile = int(patch_size)
    if tile < 32:
        raise ValueError("타일 크기는 32px 이상이어야 합니다")
    scale = float(scale)
    if not 0 < scale <= 1:
        raise ValueError("패치 학습 배율은 0보다 크고 1 이하여야 합니다")
    # scale: 추론 전에 판정 영역을 줄이는 배율 (학습 패치를 자르기 전에 줄인 것과 같다). 1 = 원본 해상도.
    return {"tile": tile, "overlap": tile // 4, "blend": "center_weighted_softmax", "scale": scale}


def make_checkpoint_metadata(task, num_classes, class_names, input_size,
                             in_channels, model_config=None, center_crop=None, tiling=None):
    """기존 키를 유지하면서 모델 구조와 전처리 규칙을 함께 저장한다."""
    from center_crop import validate_center_crop
    center_crop = validate_center_crop(center_crop)
    if in_channels not in (1, 3):
        raise ValueError("입력 채널은 1 또는 3만 지원")
    size = ([int(input_size), int(input_size)]
            if isinstance(input_size, (int, float)) else list(input_size))
    if len(size) != 2 or any(int(v) <= 0 for v in size):
        raise ValueError("입력 크기는 양수 (H, W) 필요")
    size = [int(v) for v in size]
    architecture = {
        "backbone_channels": [32, 64, 128, 256, 512],
        "csp_depth": [1, 2, 3, 2],
        "dropout": 0.2,
    }
    architecture.update(model_config or {})
    if task == "detect":
        architecture.setdefault("detection_box_encoding", "grid_sigmoid_xywh")
    mean, std = ([0.449], [0.226]) if in_channels == 1 else (
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    result = {
        "schema_version": 1, "engine": "custom", "task": task,
        "architecture_name": "Custom CSP",
        "num_classes": int(num_classes), "class_names": list(class_names),
        "input_size": size[0] if size[0] == size[1] else size,
        "in_channels": int(in_channels), "model_config": architecture,
        "preprocessing": {
            "input_size": size, "in_channels": int(in_channels),
            "mean": mean, "std": std, "resize": "bilinear",
            "resize_implementation": "opencv_linear_exact_v1", "antialias": False,
            "interpolation": "INTER_LINEAR_EXACT",
            "layout": "NCHW",
            "color_order": "GRAY" if in_channels == 1 else "RGB",
            "value_scale": 255.0,
        },
    }
    if tiling:
        if task != "segment":
            raise ValueError("타일 추론은 분할 모델만 지원합니다")
        result["tiling"] = dict(tiling)
        result["preprocessing"]["tiling"] = dict(tiling)
    if task == "anomaly":
        result.update(anomaly_threshold=None,
                      score_definition="reconstruction_mse_mean",
                      threshold_comparator=">=")
    if center_crop is not None:
        result["center_crop"] = center_crop
        result["preprocessing"]["center_crop"] = center_crop
    return result
