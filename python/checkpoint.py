"""모델을 독립적으로 복원하기 위한 체크포인트 메타데이터."""


def make_checkpoint_metadata(task, num_classes, class_names, input_size,
                             in_channels, model_config=None, center_crop=None):
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
    if task == "anomaly":
        result.update(anomaly_threshold=None,
                      score_definition="reconstruction_mse_mean",
                      threshold_comparator=">=")
    if center_crop is not None:
        result["center_crop"] = center_crop
        result["preprocessing"]["center_crop"] = center_crop
    return result
