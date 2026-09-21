"""체크포인트를 검증된 ONNX 모델과 배포 설정으로 내보낸다."""

import argparse
import json
import inspect
import math
import os
from pathlib import Path
import tempfile

import numpy as np

OUTPUT_NAMES = {
    "classify": "class_logits", "segment": "seg_mask",
    "detect": "detections", "anomaly": "reconstruction",
}

RE_DETR_VARIANTS = frozenset({"Small", "Medium", "Large"})
UPSTREAM_FAMILIES = frozenset({"mobilenetv4", "yolo9", "rtdetrv4"})


class ExportVerificationError(ValueError):
    def __init__(self, message, diagnostic):
        super().__init__(message)
        self.numerical_diagnostic = diagnostic

# CPU FP32 ONNX kernels may reassociate fused convolution/normalization
# operations. A 1e-3 absolute floor is the usual FP32 deployment parity
# boundary; top-1 is checked separately so this does not hide a class change.
# Every task uses the same bounded FP32 floor unless a task has a stricter
# contract (classification additionally requires top-1 agreement).
VERIFY_TOLERANCES = {
    # ONNX Runtime may fuse/reassociate FP32 kernels differently from
    # PyTorch.  A 1e-3 absolute floor is still far below a meaningful logit,
    # while avoiding false failures for valid exports around zero.
    "default": {"atol": 1e-3, "rtol": 1e-3},
    "classify": {"atol": 1e-3, "rtol": 5e-4},
    "segment": {"atol": 1e-3, "rtol": 1e-3},
    "detect": {"atol": 1e-3, "rtol": 1e-3},
    "anomaly": {"atol": 1e-3, "rtol": 1e-3},
}


def verification_tolerances(task="classify"):
    """Return the explicit numerical parity profile for an exported task."""
    profile = VERIFY_TOLERANCES.get(task, VERIFY_TOLERANCES["default"])
    return dict(profile)


def validate_classification_outputs(expected, actual, *, atol=None, rtol=None):
    """Validate logits and require the same top-1 class for every sample."""
    tolerances = verification_tolerances("classify")
    if atol is not None:
        tolerances["atol"] = atol
    if rtol is not None:
        tolerances["rtol"] = rtol
    validate_outputs(expected, actual, **tolerances)
    expected_array, actual_array = np.asarray(expected), np.asarray(actual)
    if expected_array.ndim != 2 or actual_array.ndim != 2:
        raise ValueError("분류 ONNX 출력은 [batch, classes] 형태여야 합니다.")
    expected_top = np.argmax(expected_array, axis=1)
    actual_top = np.argmax(actual_array, axis=1)
    if not np.array_equal(expected_top, actual_top):
        raise ValueError(f"ONNX 검증 실패: 분류 top-1 불일치 "
                         f"(PyTorch={expected_top.tolist()}, ONNX={actual_top.tolist()})")
    return True


def checkpoint_backend(checkpoint):
    """가중치 포맷을 확인하고 지원하지 않는 포맷은 즉시 거부한다."""
    if not isinstance(checkpoint, dict):
        raise ValueError("체크포인트는 메타데이터를 포함하는 사전이어야 합니다.")
    if checkpoint.get("type") == "patchcore" or checkpoint.get("backend") == "patchcore":
        return "patchcore"
    if checkpoint.get("type") == "builtin_model" or checkpoint.get("backend") == "builtin":
        return "builtin"
    if checkpoint.get("type") in {"redetr_v4", "redetr"} or checkpoint.get("backend") == "redetr_v4":
        return "redetr_v4"
    if checkpoint.get("type") == "sam2" or checkpoint.get("backend") == "sam2":
        return "sam2"
    if (checkpoint.get("model_family") in UPSTREAM_FAMILIES and
            checkpoint.get("task") in {"classify", "detect"}):
        return "libreyolo"
    if "model_state_dict" in checkpoint:
        if checkpoint.get("engine") == "efficientnet":
            return "efficientnet"
        return "custom"
    raise ValueError("지원하지 않는 체크포인트 포맷")


def _upstream_spec(checkpoint):
    """Read the schema written by the shipped LibreYOLO trainers."""
    family = checkpoint.get("model_family")
    task = checkpoint.get("task")
    if family not in UPSTREAM_FAMILIES or task not in {"classify", "detect"}:
        raise ValueError("지원하지 않는 LibreYOLO 체크포인트 메타데이터")
    num_classes = _positive_int(checkpoint.get("nc"), "nc")
    names = checkpoint.get("names")
    if isinstance(names, dict):
        class_names = [str(names.get(index, names.get(str(index), f"Class {index}")))
                       for index in range(num_classes)]
    elif isinstance(names, (list, tuple)) and len(names) == num_classes:
        class_names = [str(value) for value in names]
    else:
        raise ValueError("LibreYOLO 체크포인트의 names 메타데이터 오류")
    size = _positive_int(checkpoint.get("imgsz"), "imgsz")
    if task == "classify" and family != "mobilenetv4":
        raise ValueError("기본 제공 LibreYOLO 분류 체크포인트 모델군 불일치")
    if task == "detect" and family not in {"yolo9", "rtdetrv4"}:
        raise ValueError("기본 제공 LibreYOLO 검출 체크포인트 모델군 불일치")
    return {"family": family, "task": task, "num_classes": num_classes,
            "class_names": class_names, "input_height": size, "input_width": size,
            "in_channels": 3,
            "preprocessing": {"normalize_mean": [0.485, 0.456, 0.406],
                                "normalize_std": [0.229, 0.224, 0.225],
                                "input_size": [size, size], "in_channels": 3,
                                "color_order": "RGB"}}


def _upstream_reference_outputs(model, family, dummy):
    """Return the exact tensor schema selected by LibreYOLO's ONNX exporter."""
    import torch
    network = model.model.cpu().eval()
    head = getattr(network, "head", None)
    old_export = getattr(head, "export", None)
    if family == "yolo9" and old_export is not None:
        head.export = True
    try:
        with torch.no_grad():
            value = network(dummy.cpu())
    finally:
        if family == "yolo9" and old_export is not None:
            head.export = old_export
    if family == "rtdetrv4":
        if not isinstance(value, dict) or not {"pred_logits", "pred_boxes"} <= set(value):
            raise ValueError("LibreYOLO Re-DETR v4 PyTorch 출력 형식 오류")
        return [value["pred_logits"], value["pred_boxes"]]
    if isinstance(value, (tuple, list)):
        raise ValueError("LibreYOLO ONNX 단일 출력 모델 형식 오류")
    return [value]


def verify_upstream_onnx(onnx_path, model, spec, dummy):
    """Compare every raw exported tensor with the public native checkpoint."""
    import onnxruntime as ort
    actual = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(
        None, {"images": dummy.cpu().numpy()})
    expected = _upstream_reference_outputs(model, spec["family"], dummy)
    if len(actual) != len(expected):
        raise ValueError(f"LibreYOLO ONNX 출력 개수 불일치: PyTorch={len(expected)}, ONNX={len(actual)}")
    tolerances = verification_tolerances(spec["task"])
    for index, (reference, converted) in enumerate(zip(expected, actual)):
        reference = reference.detach().cpu().numpy()
        if spec["task"] == "classify":
            validate_classification_outputs(reference, converted, **tolerances)
        else:
            try:
                validate_outputs(reference, converted, **tolerances)
            except ValueError as exc:
                raise ValueError(f"LibreYOLO ONNX 출력 {index} 검증 실패: {exc}") from exc
    return True


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name}: 양의 정수 필요")
    return int(value)


def resolve_checkpoint_spec(checkpoint, overrides=None):
    """체크포인트를 기준으로 모델 구조와 전처리 규약을 복원한다.

    CLI 값은 메타데이터가 없는 구형 체크포인트에만 보충할 수 있다.
    기존 메타데이터와 충돌하는 값은 암묵적으로 덮어쓰지 않는다.
    """
    backend = checkpoint_backend(checkpoint)
    if backend == "sam2":
        raise ValueError("SAM2 multi-graph checkpoint uses export_sam2_checkpoint")
    if backend not in {"custom", "efficientnet", "builtin", "redetr_v4"}:
        raise ValueError("커스텀 또는 기본 모델 체크포인트 필요")
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    preprocessing = dict(checkpoint.get("preprocessing") or {})
    from center_crop import checkpoint_center_crop
    crop = checkpoint_center_crop(checkpoint)
    if crop:
        preprocessing["center_crop"] = crop
    model_config = dict(checkpoint.get("model_config") or {})
    if backend == "redetr_v4":
        if checkpoint.get("task", "detect") != "detect":
            raise ValueError("Re-DETR v4 checkpoint는 detect 태스크여야 합니다.")
        variant = checkpoint.get("variant", model_config.get("variant"))
        if variant not in RE_DETR_VARIANTS:
            raise ValueError("Re-DETR v4 variant는 Small, Medium 또는 Large여야 합니다.")
        model_config["variant"] = variant
        model_config.setdefault("boxes_format", checkpoint.get("boxes_format", "normalized_cxcywh"))
        if model_config["boxes_format"] not in {"normalized_cxcywh", "normalized_xyxy"}:
            raise ValueError("Re-DETR boxes_format은 normalized_cxcywh 또는 normalized_xyxy여야 합니다.")
        model_config.setdefault("score_activation", checkpoint.get("score_activation", "sigmoid"))
        if model_config["score_activation"] not in {"sigmoid", "softmax"}:
            raise ValueError("Re-DETR score_activation은 sigmoid 또는 softmax여야 합니다.")
    if backend == "builtin":
        from builtin_models import get_builtin_spec
        model_id = checkpoint.get("model_id") or model_config.get("model_id")
        if not isinstance(model_id, str):
            raise ValueError("기본 모델 체크포인트의 model_id 메타데이터 누락")
        builtin = get_builtin_spec(model_id)
        stored_task = checkpoint.get("task")
        if stored_task is not None and stored_task != builtin.task:
            raise ValueError("기본 모델 task와 model_id가 일치하지 않습니다.")
        model_config["model_id"] = model_id
    names = checkpoint.get("class_names") or []
    if not isinstance(names, (list, tuple)) or any(not isinstance(n, str) for n in names):
        raise ValueError("class_names: 문자열 목록 필요")

    def select(key, fallback=None):
        stored = checkpoint.get(key, preprocessing.get(key, model_config.get(key)))
        supplied = overrides.get(key)
        if stored is not None and supplied is not None and stored != supplied:
            raise ValueError(f"{key}: 체크포인트 설정과 명령행 설정 불일치")
        value = stored if stored is not None else supplied
        if value is None:
            value = fallback
        if value is None:
            raise ValueError(f"구형 체크포인트의 {key} 메타데이터 누락: CLI에서 명시 필요")
        return value

    task = select("task", "detect" if backend == "redetr_v4" else None)
    if task not in OUTPUT_NAMES:
        raise ValueError(f"지원하지 않는 태스크: {task}")
    num_classes = _positive_int(select("num_classes", len(names) or None), "num_classes")
    channels = _positive_int(select("in_channels"), "in_channels")
    if channels not in (1, 3):
        raise ValueError("입력 채널은 1 또는 3만 지원합니다.")
    if backend == "efficientnet":
        from efficientnet_contract import checkpoint_input_contract
        contract = checkpoint_input_contract(checkpoint, channels)
        model_config.update(contract)
        if channels == 1:
            preprocessing["grayscale_adapter"] = contract["input_adapter"]
    size = select("input_size", preprocessing.get("size"))
    if isinstance(size, (int, np.integer)):
        size = [size, size]
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("input_size: 정수 또는 [높이, 너비] 필요")
    height, width = (_positive_int(v, "input_size") for v in size)
    mean_default = [0.449] if channels == 1 else [0.485, 0.456, 0.406]
    std_default = [0.226] if channels == 1 else [0.229, 0.224, 0.225]
    mean = preprocessing.get("normalize_mean", preprocessing.get("mean", mean_default))
    std = preprocessing.get("normalize_std", preprocessing.get("std", std_default))
    if len(mean) != channels or len(std) != channels:
        raise ValueError("정규화 계수 개수와 입력 채널 불일치")
    if not all(math.isfinite(float(x)) for x in [*mean, *std]) or any(float(x) <= 0 for x in std):
        raise ValueError("정규화 값은 유한해야 하며 std는 양수여야 합니다.")
    if names and len(names) != num_classes:
        raise ValueError("클래스 이름 개수와 num_classes 불일치")
    from opencv_preprocess import resize_contract
    preprocessing = resize_contract(preprocessing)
    preprocessing.update({"input_size": [height, width], "in_channels": channels,
                          "normalize_mean": list(mean), "normalize_std": list(std),
                          "color_order": "GRAY" if channels == 1 else "RGB"})
    return {"task": task, "num_classes": num_classes, "in_channels": channels,
            "input_height": height, "input_width": width, "model_config": model_config,
            "preprocessing": preprocessing, "class_names": list(names)}


def load_custom_model(checkpoint, spec=None):
    spec = spec or resolve_checkpoint_spec(checkpoint)
    config = spec["model_config"]
    backend = checkpoint_backend(checkpoint)
    if backend == "redetr_v4":
        import torch.nn as nn
        model = checkpoint.get("model")
        if not isinstance(model, nn.Module):
            raise ValueError("Re-DETR checkpoint는 export 가능한 nn.Module을 model 필드에 포함해야 합니다.")
        return model.cpu().eval()
    if backend == "builtin":
        from builtin_models import load_builtin_checkpoint
        return load_builtin_checkpoint(checkpoint)
    if backend == "efficientnet":
        from efficientnet import EfficientNet, VARIANTS
        from efficientnet_contract import checkpoint_input_contract, LEGACY_GRAY_INPUT, channel_description
        contract = checkpoint_input_contract(checkpoint, spec["in_channels"])
        architecture = config.get("architecture")
        if spec["task"] != "classify" or architecture not in VARIANTS:
            raise ValueError("EfficientNet 모델 정의 또는 버전 불일치")
        model = EfficientNet(architecture, spec["num_classes"], spec["in_channels"], config.get("dropout", .2),
                             input_adapter=contract["input_adapter"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if contract["input_adapter"] == LEGACY_GRAY_INPUT:
            import warnings
            warnings.warn(channel_description(contract) + ": 기존 판정을 유지하는 호환 모드입니다. "
                          "새 1ch Conv 학습은 ImageNet 사전학습 모드를 선택하세요.", UserWarning, stacklevel=2)
        return model.cpu().eval()
    from model import CustomCSP
    kwargs = {key: config[key] for key in ("backbone_channels", "csp_depth", "dropout") if key in config}
    if spec["task"] == "detect":
        # 구형 가중치를 새 좌표식으로 해석하지 않는다.
        kwargs["detection_box_encoding"] = config.get("detection_box_encoding", "legacy_raw")
    model = CustomCSP(task=spec["task"], in_channels=spec["in_channels"],
                   num_classes=spec["num_classes"], **kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.cpu().eval()


def validate_outputs(expected, actual, atol=1e-4, rtol=1e-4):
    """출력 모양, 유한성, 수치 허용 오차를 확인하며 실패 시 예외를 낸다."""
    expected, actual = np.asarray(expected), np.asarray(actual)
    if expected.shape != actual.shape:
        raise ValueError(f"ONNX 출력 형태 불일치: {expected.shape} != {actual.shape}")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("ONNX 검증 실패: NaN 또는 무한대 출력 "
                         f"(PyTorch={np.count_nonzero(~np.isfinite(expected))}, "
                         f"ONNX={np.count_nonzero(~np.isfinite(actual))})")
    if not np.allclose(expected, actual, atol=atol, rtol=rtol):
        reference, converted = expected.astype(np.float64), actual.astype(np.float64)
        difference = np.abs(reference - converted)
        # Match np.allclose's reference operand; report the most out-of-tolerance
        # element, which need not be the element with the largest absolute error.
        limit = atol + rtol * np.abs(converted)
        with np.errstate(over="ignore"):
            ratio = difference / np.maximum(limit, np.finfo(np.float64).tiny)
        index = tuple(int(value) for value in np.unravel_index(int(np.argmax(ratio)), ratio.shape))
        raise ValueError(
            f"ONNX 검증 실패: 최대 오차 {difference.max():.8g} (atol={atol}, rtol={rtol}); "
            f"실패 원소 {index}: PyTorch={reference[index]:.8g}, ONNX={converted[index]:.8g}, "
            f"허용 오차의 {ratio[index]:.4g}배; "
            f"출력 범위 PyTorch=[{reference.min():.8g}, {reference.max():.8g}], "
            f"ONNX=[{converted.min():.8g}, {converted.max():.8g}]")
    return True


def export_to_onnx(model, dummy_input, output_path, opset_version=17,
                   dynamic_batch=False, task="classify", output_names=None,
                   constant_folding=True):
    import torch
    from torch.onnx import _constants
    maximum = getattr(_constants, "ONNX_MAX_OPSET", 17)
    if type(opset_version) is not int or not 11 <= opset_version <= maximum:
        raise ValueError(f"설치된 PyTorch의 ONNX opset 범위: 11~{maximum}. 기본값 17을 사용하세요.")
    output_name = OUTPUT_NAMES[task]
    names = list(output_names or [output_name])
    axes = ({"input_image": {0: "batch_size"},
             **{name: {0: "batch_size"} for name in names}} if dynamic_batch else None)
    model.cpu().eval()
    # 명시적 legacy exporter로 버전별 기본값 변화와 외부 데이터 분리를 피한다.
    exporter_options = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    if "external_data" in inspect.signature(torch.onnx.export).parameters:
        exporter_options["external_data"] = False
    with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
        torch.onnx.export(model, dummy_input.cpu(), str(output_path), export_params=True,
                          opset_version=opset_version, do_constant_folding=constant_folding,
                          input_names=["input_image"], output_names=names,
                          dynamic_axes=axes, **exporter_options)
    return str(output_path)


def verify_onnx(onnx_path, dummy_input, pytorch_model, task="classify", atol=None,
                runtime_settings=None, reference_model=None):
    import torch
    import onnxruntime as ort
    from onnx_session import cpu_session_options
    options = cpu_session_options(**(runtime_settings or {}))
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    reference = reference_model if reference_model is not None else pytorch_model
    reference.cpu().eval()
    with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False):
        expected = reference(dummy_input.cpu()).detach().numpy()
    actual = session.run(None, {session.get_inputs()[0].name: dummy_input.cpu().numpy()})
    if len(actual) != 1:
        raise ValueError("커스텀 모델은 출력 텐서 하나만 지원합니다.")
    if task == "classify":
        return validate_classification_outputs(expected, actual[0], atol=atol)
    tolerances = verification_tolerances(task)
    # Preserve an explicit caller override for compatibility with tooling that
    # asks for a stricter absolute floor.  ``None`` selects the task profile.
    if atol is not None:
        tolerances["atol"] = atol
    return validate_outputs(expected, actual[0], **tolerances)


def verify_redetr_onnx(onnx_path, dummy_input, pytorch_model):
    """Verify both Re-DETR outputs and their required tensor shapes."""
    import torch
    import onnxruntime as ort
    pytorch_model.cpu().eval()
    with torch.no_grad():
        expected = pytorch_model(dummy_input.cpu())
    if not isinstance(expected, (tuple, list)) or len(expected) != 2:
        raise ValueError("Re-DETR model must return (pred_boxes, pred_logits)")
    actual = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(
        None, {"input_image": dummy_input.cpu().numpy()})
    if len(actual) != 2:
        raise ValueError("Re-DETR ONNX must contain pred_boxes and pred_logits outputs")
    expected_boxes, expected_logits = (value.detach().numpy() for value in expected)
    actual_boxes, actual_logits = actual
    if expected_boxes.ndim != 3 or expected_boxes.shape[-1] != 4:
        raise ValueError("Re-DETR boxes must have shape [batch, queries, 4]")
    if expected_logits.ndim != 3 or expected_logits.shape[:2] != expected_boxes.shape[:2]:
        raise ValueError("Re-DETR logits must have shape [batch, queries, classes]")
    tolerances = verification_tolerances("detect")
    validate_outputs(expected_boxes, actual_boxes, **tolerances)
    validate_outputs(expected_logits, actual_logits, **tolerances)
    return True


def simplify_onnx(onnx_path):
    import onnx
    from onnxsim import simplify
    simplified, valid = simplify(onnx.load(str(onnx_path)))
    if not valid:
        raise ValueError("ONNX 단순화 검증 실패")
    onnx.save(simplified, str(onnx_path))


def create_inference_config(output_dir, task, num_classes, input_size,
                            in_channels, onnx_filename, class_names=None,
                            preprocessing=None, backend="custom", verification="passed",
                            config_filename="inference_config.json", input_name="input_image",
                            output_names=None, detection_box_encoding=None, architecture=None,
                            model_config=None):
    size = [input_size, input_size] if isinstance(input_size, int) else list(input_size)
    preprocessing = dict(preprocessing or {})
    upstream_backends = {"libreyolo_mobilenetv4", "libreyolo_yolo9", "libreyolo_rtdetrv4"}
    if backend not in {"custom", "builtin", "efficientnet", "patchcore", "redetr_v4", *upstream_backends}:
        raise ValueError("지원하지 않는 모델 형식입니다. 등록된 모델 체크포인트를 선택하세요.")
    if backend in {"redetr_v4", "libreyolo_rtdetrv4", "libreyolo_yolo9"} and task != "detect":
        raise ValueError("Re-DETR v4 backend는 detect 태스크만 지원합니다.")
    if backend in {"redetr_v4", "libreyolo_rtdetrv4"} and (output_names is None or len(output_names) != 2):
        raise ValueError("Re-DETR v4는 pred_boxes와 pred_logits 두 출력 이름이 필요합니다.")
    detection_cpp_supported = task != "detect" or detection_box_encoding in {
        "grid_sigmoid_xywh", "normalized_cxcywh"
    }
    if backend in {"redetr_v4", "libreyolo_rtdetrv4"}:
        detection_cpp_supported = detection_box_encoding in {None, "normalized_cxcywh", "normalized_xyxy"}
    if backend == "libreyolo_yolo9":
        detection_cpp_supported = detection_box_encoding == "pixel_xyxy"
    config = {
        "schema_version": 1, "backend": backend, "model_path": onnx_filename,
        "task": task, "num_classes": num_classes, "input_channels": in_channels,
        "input_height": size[0], "input_width": size[1], "input_name": input_name,
        "output_name": (output_names or [OUTPUT_NAMES.get(task, "output")])[0],
        "output_names": output_names or [OUTPUT_NAMES.get(task, "output")],
        "normalize_mean": preprocessing.get("normalize_mean", [0.449] if in_channels == 1 else [0.485, 0.456, 0.406]),
        "normalize_std": preprocessing.get("normalize_std", [0.226] if in_channels == 1 else [0.229, 0.224, 0.225]),
        "class_names": list(class_names or []), "preprocessing": preprocessing,
        "verification": verification,
        "cpp_supported": ((backend in {"custom", "builtin"} and task in ("classify", "segment", "detect", "anomaly") and
                            detection_cpp_supported) or
                          (backend == "patchcore" and task == "anomaly")) or
                         (backend in {"efficientnet", "libreyolo_mobilenetv4"} and task == "classify") or
                         (backend in {"redetr_v4", "libreyolo_rtdetrv4", "libreyolo_yolo9"} and
                          task == "detect" and detection_cpp_supported),
    }
    if architecture is not None:
        config["architecture"] = architecture
    if backend == "efficientnet":
        config["postprocessing"] = {"output": "logits", "softmax_axis": 1}
        if model_config is not None:
            from efficientnet_contract import model_input_contract
            config["model_config"] = {**model_config, **model_input_contract(model_config, in_channels)}
    elif backend == "builtin" and model_config is not None:
        config["model_config"] = dict(model_config)
    elif backend in {"redetr_v4", "libreyolo_rtdetrv4"}:
        config["postprocessing"] = {
            "box_format": detection_box_encoding or "normalized_cxcywh",
            "objectness": "none", "class_scores": "sigmoid",
            "confidence": "class_score", "class_aware_nms": True,
            "confidence_threshold": .25, "iou_threshold": .5, "max_detections": 300,
        }
        if model_config is not None:
            config["model_config"] = dict(model_config)
            if "variant" in model_config:
                config["variant"] = model_config["variant"]
    if preprocessing.get("center_crop"):
        # Older C++ readers must reject this contract instead of ignoring the ROI.
        config["schema_version"] = 2
    if config["cpp_supported"]:
        from opencv_preprocess import resize_contract
        preprocessing = resize_contract(preprocessing)
        config["schema_version"] = 5
        config["preprocessing"] = {**preprocessing,
            "color_order": "GRAY" if in_channels == 1 else "RGB",
            "center_crop": preprocessing.get("center_crop")}
    if backend == "custom" and task == "detect":
        config["postprocessing"] = {"box_format": "normalized_cxcywh", "objectness": "sigmoid",
                                    "class_scores": "sigmoid", "confidence": "objectness_times_class",
                                    "class_aware_nms": True, "confidence_threshold": .25,
                                    "iou_threshold": .5, "max_detections": 300,
                                    "box_encoding": detection_box_encoding or "legacy_raw"}
    if backend == "custom" and task == "anomaly":
        config["postprocessing"] = {"score": "mean_absolute_reconstruction_error",
                                    "threshold": 0.0, "map": "per_pixel_mean_absolute_error"}
    path = Path(output_dir) / config_filename
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return str(path)


def _export_upstream_checkpoint(checkpoint_path, output, checkpoint, *, opset_version,
                                dynamic_batch, simplify, verify, log):
    """Export shipped LibreYOLO checkpoints without rebuilding them as CustomCSP."""
    import onnx
    import torch
    from upstream_models import load_upstream_checkpoint

    spec = _upstream_spec(checkpoint)
    family = spec["family"]
    backend = {"mobilenetv4": "libreyolo_mobilenetv4", "yolo9": "libreyolo_yolo9",
               "rtdetrv4": "libreyolo_rtdetrv4"}[family]
    output = Path(output)
    config_name = output.with_suffix(".json").name
    with tempfile.TemporaryDirectory(prefix=".onnx-export-", dir=output.parent) as temporary:
        stage = Path(temporary)
        staged_model = stage / output.name
        log(f"내보내기 시작: LibreYOLO {family}")
        model = load_upstream_checkpoint(checkpoint_path, device="cpu")
        exported = Path(model.export(format="onnx", output_path=str(staged_model),
                                     imgsz=(spec["input_height"], spec["input_width"]),
                                     opset=opset_version, dynamic=dynamic_batch,
                                     simplify=simplify, device="cpu"))
        if exported.resolve() != staged_model.resolve() or not staged_model.is_file():
            raise ValueError("LibreYOLO ONNX exporter가 요청한 출력 파일을 생성하지 않았습니다")
        onnx.checker.check_model(str(staged_model))
        if verify:
            generator = torch.Generator().manual_seed(42)
            dummy = torch.randn(1, 3, spec["input_height"], spec["input_width"], generator=generator)
            log("ONNX Runtime 검증: LibreYOLO가 내보낸 원본 PyTorch 텐서와 비교 (seeded+zero)")
            verify_upstream_onnx(staged_model, model, spec, dummy)
            verify_upstream_onnx(staged_model, model, spec, torch.zeros_like(dummy))
            if dynamic_batch:
                verify_upstream_onnx(staged_model, model, spec, dummy.repeat(2, 1, 1, 1))
        output_names = (["pred_logits", "pred_boxes"] if family == "rtdetrv4" else ["output"])
        detection_encoding = ("pixel_xyxy" if family == "yolo9" else "normalized_cxcywh")
        staged_config = create_inference_config(
            stage, spec["task"], spec["num_classes"],
            [spec["input_height"], spec["input_width"]], 3, output.name,
            spec["class_names"], preprocessing=spec["preprocessing"], backend=backend,
            verification="passed" if verify else "skipped", config_filename=config_name,
            output_names=output_names, detection_box_encoding=detection_encoding,
            model_config={"model_family": family, "upstream_export": "libreyolo_public_api"})
        manifest = json.loads(Path(staged_config).read_text(encoding="utf-8"))
        manifest["export"] = {"opset": opset_version, "precision": "float32",
                              "dynamic_batch": dynamic_batch,
                              "verification_tolerance": verification_tolerances(spec["task"]),
                              "verification_reference": "libreyolo_exported_pytorch_graph"}
        if family == "yolo9":
            manifest["postprocessing"] = {
                "box_format": "pixel_xyxy", "objectness": "none",
                "class_scores": "probabilities", "confidence": "class_score",
                "class_aware_nms": True, "confidence_threshold": .25,
                "iou_threshold": .5, "max_detections": 300,
            }
        Path(staged_config).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(staged_model, output)
        config_path = output.with_suffix(".json")
        os.replace(staged_config, config_path)
    result = {"output_path": str(output), "config_path": str(config_path),
              "file_size_mb": output.stat().st_size / (1024 * 1024), "backend": backend,
              "verification": "passed" if verify else "skipped", "task": spec["task"],
              "cpp_supported": manifest["cpp_supported"]}
    log(f"내보내기 완료: {output}\n배포 설정: {config_path}\n검증: {result['verification']}")
    return result


def _export_checkpoint(checkpoint_path, output_path, opset_version=17, dynamic_batch=False,
                      simplify=False, verify=True, overrides=None, log=print):
    """검증 완료 후 ONNX와 동일 이름의 .json을 대상 폴더에 배치한다."""
    import torch
    import onnx
    output = Path(output_path).resolve()
    if output.suffix.lower() != ".onnx":
        raise ValueError("출력 파일 확장자는 .onnx여야 합니다.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # 사용자 선택 체크포인트를 로드한다.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backend = checkpoint_backend(checkpoint)
    if backend not in {"custom", "efficientnet", "builtin", "patchcore", "redetr_v4", "sam2", "libreyolo"}:
        raise ValueError("지원하지 않는 모델 형식입니다. 등록된 모델 체크포인트를 선택하세요.")
    if output.resolve() == Path(checkpoint_path).resolve():
        raise ValueError("체크포인트와 ONNX 출력 경로가 같을 수 없습니다.")
    if backend == "patchcore":
        from export_patchcore_onnx import export_patchcore_checkpoint
        return export_patchcore_checkpoint(checkpoint_path, output_path, verify=verify,
                                           opset=opset_version, simplify=simplify, log=log)
    if backend == "sam2":
        if checkpoint.get("type") == "sam2_finetune":
            from export_sam2_onnx import export_official_sam2_checkpoint
            model_id = checkpoint.get("model_id")
            if not isinstance(model_id, str) or not model_id:
                raise ValueError("SAM2 미세조정 체크포인트에 model_id가 없습니다.")
            return export_official_sam2_checkpoint(model_id, checkpoint_path, output.parent,
                                                    verify=verify, opset=opset_version, log=log)
        from export_sam2_onnx import export_sam2_checkpoint
        return export_sam2_checkpoint(checkpoint_path, output.parent, verify=verify,
                                      opset=opset_version, log=log)
    if backend == "libreyolo":
        return _export_upstream_checkpoint(checkpoint_path, output, checkpoint,
                                           opset_version=opset_version, dynamic_batch=dynamic_batch,
                                           simplify=simplify, verify=verify, log=log)
    config_name = output.with_suffix(".json").name
    with tempfile.TemporaryDirectory(prefix=".onnx-export-", dir=output.parent) as temp:
        stage = Path(temp)
        staged_model = stage / output.name
        runtime_settings = None
        runtime_attempts = []
        log(f"내보내기 시작: {backend}")
        if backend in {"custom", "efficientnet", "builtin", "redetr_v4"}:
            spec = resolve_checkpoint_spec(checkpoint, overrides)
            model = load_custom_model(checkpoint, spec)
            reference_model = model if backend == "efficientnet" else None
            verification_profile = verification_tolerances(spec["task"])
            if backend == "efficientnet":
                from efficientnet import prepare_for_inference
                model = prepare_for_inference(model)
                log(f"추론 최적화: {model.inference_optimization}")
            generator = torch.Generator().manual_seed(42)
            dummy = torch.randn(1, spec["in_channels"], spec["input_height"], spec["input_width"], generator=generator)
            log(f"ONNX 그래프 생성: opset={opset_version}, 입력={tuple(dummy.shape)}")
            redetr_outputs = ["pred_boxes", "pred_logits"] if backend == "redetr_v4" else None
            export_kwargs = {} if redetr_outputs is None else {"output_names": redetr_outputs}
            export_to_onnx(model, dummy, staged_model, opset_version, dynamic_batch,
                           spec["task"], **export_kwargs)
            if simplify:
                simplify_onnx(staged_model)
            if verify:
                # Fusion is itself a numerical transformation. Every EfficientNet
                # candidate must agree with one original checkpoint reference.
                log("ONNX Runtime 검증: 원본 체크포인트 PyTorch 출력과 비교 "
                    f"(atol={verification_profile['atol']}, rtol={verification_profile['rtol']}, probes=seeded+zero)")
                failed_probe, failed_probe_name = dummy, "seeded"
                def verify_candidate(candidate, settings=None):
                    nonlocal failed_probe, failed_probe_name
                    probes = [("seeded", dummy), ("zero", torch.zeros_like(dummy))]
                    if dynamic_batch:
                        probes.append(("dynamic_batch_2", dummy.repeat(2, 1, 1, 1)))
                    for probe_name, probe in probes:
                        failed_probe, failed_probe_name = probe, probe_name
                        kwargs = {} if settings is None else {"runtime_settings": settings}
                        if reference_model is not None:
                            kwargs["reference_model"] = reference_model
                        verified = (verify_redetr_onnx(staged_model, probe, candidate)
                                    if backend == "redetr_v4"
                                    else verify_onnx(staged_model, probe, candidate, task=spec["task"], **kwargs))
                        if not verified:
                            raise ValueError("ONNX 검증 실패")
                try:
                    verify_candidate(model)
                except ValueError as optimized_error:
                    if backend != "efficientnet":
                        raise
                    log(f"최적화 그래프 검증 실패: {optimized_error}")
                    log("원본 FP32 그래프로 재시도: Conv/BN 사전 융합·상수 폴딩·단순화 제외")
                    model = load_custom_model(checkpoint, spec).cpu().float().eval()
                    model.inference_optimization = {
                        "conv_bn_fused": 0, "channels_last": False,
                        "constant_folding": False, "simplified": False,
                        "fallback": "unfused_fp32",
                    }
                    export_to_onnx(model, dummy, staged_model, opset_version, dynamic_batch,
                                   spec["task"], constant_folding=False)
                    try:
                        verify_candidate(model)
                    except ValueError as original_error:
                        # A diagnostic pass on one probe is insufficient. Validate
                        # every input again using the exact settings to be shipped.
                        threads = max(1, min(torch.get_num_threads(), 4))
                        for level in ("all", "basic", "disabled"):
                            settings = {"graph_optimization_level": level, "num_threads": threads}
                            log(f"원본 그래프 실행 설정 재시도: 최적화={level}, CPU {threads} threads")
                            try:
                                verify_candidate(model, settings)
                            except ValueError as runtime_error:
                                runtime_attempts.append({**settings, "passed": False,
                                                         "probe": failed_probe_name, "error": str(runtime_error)})
                            else:
                                runtime_settings = settings
                                runtime_attempts.append({**settings, "passed": True})
                                log(f"모든 입력 검증 통과: 최적화={level}, CPU {threads} threads. 배포 JSON에 설정 저장")
                                break
                        if runtime_settings is None:
                            from efficientnet import prepare_native_bn_export
                            log("BatchNorm 계수 고정 그래프로 재시도: PyTorch 정규화 계수 보존, Conv 융합 제외")
                            for emulate_fma in (False, True):
                                model = prepare_native_bn_export(reference_model, emulate_fma=emulate_fma)
                                graph_name = model.inference_optimization["fallback"]
                                log(f"BatchNorm 연산 방식: {graph_name}")
                                export_to_onnx(model, dummy, staged_model, opset_version, dynamic_batch,
                                               spec["task"], constant_folding=False)
                                # Preserve operation order; try serial reduction as well.
                                for count in dict.fromkeys((threads, 1)):
                                    settings = {"graph_optimization_level": "disabled", "num_threads": count}
                                    try:
                                        verify_candidate(model, settings)
                                    except ValueError as runtime_error:
                                        runtime_attempts.append({**settings, "graph": graph_name,
                                                                 "passed": False, "probe": failed_probe_name,
                                                                 "error": str(runtime_error)})
                                    else:
                                        runtime_settings = settings
                                        runtime_attempts.append({**settings, "graph": graph_name, "passed": True})
                                        log("BatchNorm 계수 고정 그래프: 모든 입력 원본 출력 비교 통과")
                                        break
                                if runtime_settings is not None:
                                    break
                            if runtime_settings is None:
                                from efficientnet_precision import prepare_precision_export
                                log("고정밀 ONNX 재시도: 내부 FP64 계산, 입출력 FP32 유지. 추론 속도는 느려질 수 있습니다")
                                model = prepare_precision_export(reference_model)
                                settings = {"graph_optimization_level": "disabled", "num_threads": threads}
                                try:
                                    export_to_onnx(model, dummy, staged_model, opset_version,
                                                   dynamic_batch, spec["task"], constant_folding=False)
                                    verify_candidate(model, settings)
                                except Exception as precision_error:
                                    runtime_attempts.append({**settings, "graph": "portable_fp64",
                                        "passed": False, "probe": failed_probe_name, "error": str(precision_error)})
                                else:
                                    runtime_settings = settings
                                    runtime_attempts.append({**settings, "graph": "portable_fp64", "passed": True})
                                    log("고정밀 ONNX: 모든 검사 입력이 원본 FP32 출력 비교 통과")
                                    from efficientnet_precision_tuning import tune_precision_export
                                    log("검증을 유지하며 FP64 범위 축소·추론 속도 측정 중")
                                    model, runtime_settings = tune_precision_export(
                                        reference_model, model, staged_model, dummy, runtime_settings,
                                        verify_candidate, opset=opset_version, dynamic_batch=dynamic_batch,
                                        attempts=runtime_attempts, log=log)
                            if runtime_settings is None:
                                # Diagnose the original graph, not a differently
                                # lowered rescue graph against an altered reference.
                                export_to_onnx(reference_model, dummy, staged_model, opset_version,
                                               dynamic_batch, spec["task"], constant_folding=False)
                                _raise_efficientnet_verification_error(
                                    reference_model, staged_model, failed_probe, failed_probe_name, opset_version,
                                    optimized_error, original_error, runtime_attempts, log)
            metadata = {"task": spec["task"], "num_classes": spec["num_classes"],
                        "input_size": [spec["input_height"], spec["input_width"]],
                        "in_channels": spec["in_channels"], "class_names": spec["class_names"],
                        "preprocessing": spec["preprocessing"]}
            if backend == "efficientnet":
                metadata["architecture"] = spec["model_config"]["architecture"]
                metadata["model_config"] = model.checkpoint_config()
            if backend == "builtin":
                metadata["architecture"] = spec["model_config"]["model_id"]
                metadata["model_config"] = {"model_id": spec["model_config"]["model_id"]}
            if backend == "redetr_v4":
                metadata["model_config"] = dict(spec["model_config"])
                metadata["model_config"]["output_names"] = ["pred_boxes", "pred_logits"]
                metadata["detection_box_encoding"] = metadata["model_config"].get(
                    "boxes_format", "normalized_cxcywh")
            if spec["task"] == "detect":
                metadata.setdefault("detection_box_encoding",
                                    spec["model_config"].get("detection_box_encoding", "legacy_raw"))
        else:
            raise ValueError(f"내보내기 미지원: {backend}")
        onnx.checker.check_model(str(staged_model))
        staged_config = create_inference_config(stage, onnx_filename=output.name,
            backend=backend, verification="passed" if verify else "skipped",
            config_filename=config_name, output_names=(redetr_outputs if backend == "redetr_v4" else None),
            **metadata)
        manifest = json.loads(Path(staged_config).read_text(encoding="utf-8"))
        manifest["export"] = {"opset": opset_version, "precision": "float32", "dynamic_batch": dynamic_batch}
        if backend in {"custom", "efficientnet", "builtin", "redetr_v4"}:
            manifest["export"]["verification_tolerance"] = verification_profile
        if backend == "efficientnet":
            manifest["export"]["optimization"] = model.inference_optimization
            manifest["export"]["io_precision"] = "float32"
            manifest["export"]["compute_precision"] = model.inference_optimization.get("compute_precision", "float32")
            manifest["export"]["verification_reference"] = "original_checkpoint_pytorch"
            if runtime_settings is not None:
                # Earlier SDKs must reject this contract instead of silently
                # enabling ALL again and undoing the verified fallback.
                manifest["schema_version"] = 6
                manifest["onnxruntime"] = {"graph_optimization_level": runtime_settings["graph_optimization_level"]}
                manifest["num_threads"] = runtime_settings["num_threads"]
                manifest["export"]["verified_runtime_settings"] = runtime_settings
                manifest["export"]["runtime_attempts"] = runtime_attempts
        if backend == "redetr_v4":
            manifest["export"]["output_names"] = ["pred_boxes", "pred_logits"]
            manifest["postprocessing"]["class_scores"] = spec["model_config"].get("score_activation", "sigmoid")
        Path(staged_config).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        # 검증 실패 시 기존 파일은 그대로 유지된다. 각 파일은 같은 파일시스템에서 교체한다.
        os.replace(staged_model, output)
        config_path = output.with_suffix(".json")
        os.replace(staged_config, config_path)
    result = {"output_path": str(output), "config_path": str(config_path),
              "file_size_mb": output.stat().st_size / (1024 * 1024),
              "backend": backend, "verification": "passed" if verify else "skipped",
              "task": metadata["task"], "cpp_supported": manifest["cpp_supported"]}
    if runtime_settings is not None:
        result["runtime_settings"] = runtime_settings
    if backend == "efficientnet":
        result["optimization"] = manifest["export"]["optimization"]
    log(f"내보내기 완료: {output}\n배포 설정: {config_path}\n검증: {result['verification']}")
    return result


def _raise_efficientnet_verification_error(model, path, probe, probe_name, opset,
                                         optimized_error, original_error, attempts, log):
    log("ONNX 수치 원인 진단: 실행 최적화·FP64·중간 레이어 비교 중")
    try:
        from onnx_diagnostics import diagnose_efficientnet
        numeric = diagnose_efficientnet(model, path, probe, probe_name=probe_name, opset=opset)
        summary = numeric["summary"]
        if numeric.get("first_divergent_stage"):
            summary += f"; 진단 그래프 최초 차이: {numeric['first_divergent_stage']}"
        operator = numeric.get("operator_diagnostic", {})
        if operator.get("first_local_difference"):
            summary += f"; 동일 입력 연산 비교 실패: {operator['first_local_difference']}"
        elif operator.get("first_propagated_difference"):
            summary += f"; 누적 차이 관측: {operator['first_propagated_difference']} (단일 연산 원인 미확정)"
    except Exception as diagnostic_error:
        numeric = {"diagnostic_error": str(diagnostic_error)}
        summary = f"추가 수치 진단 실패: {diagnostic_error}"
    numeric["runtime_attempts"] = attempts
    attempt_details = "\n".join(
        f"{item.get('graph', 'unfused_fp32')}/{item['graph_optimization_level']}/"
        f"{item['num_threads']} threads/{item.get('probe', 'unknown')}: {item.get('error', '통과')}"
        for item in attempts)
    log("수치 진단: " + summary)
    raise ExportVerificationError(
        f"최적화·원본 그래프 및 실행 설정 모두 ONNX 검증 실패.\n"
        f"최적화: {optimized_error}\n원본 FP32: {original_error}\n"
        f"실행 설정 재시도:\n{attempt_details}\n수치 진단: {summary}", numeric
    ) from original_error


def export_checkpoint(checkpoint_path, output_path, opset_version=17, dynamic_batch=False,
                      simplify=False, verify=True, overrides=None, log=print,
                      bundle_output=None):
    """실패 단계와 환경을 자동 기록한다. 검증 실패한 모델은 배포하지 않는다.

    ``bundle_output`` is optional so existing callers can keep the historical
    ``.onnx + .json`` output while release tooling can atomically produce a
    self-contained ``.dvdeploy`` directory for the native SDK.
    """
    import importlib.metadata
    import platform
    import traceback
    events = []
    def report(message):
        events.append(str(message))
        log(message)
    report("체크포인트 및 ONNX 의존성 로드")
    try:
        result = _export_checkpoint(checkpoint_path, output_path, opset_version, dynamic_batch,
                                    simplify, verify, overrides, report)
        if bundle_output is not None:
            from model_runtime.deployment_bundle import build_deployment_bundle
            source = result.get("output_dir") or result.get("output_path")
            if not source:
                raise ValueError("export result has no deployment source")
            bundle = build_deployment_bundle(source, bundle_output)
            result["bundle_path"] = str(bundle)
            report(f"배포 번들 생성: {bundle}")
    except Exception as error:
        versions = {"python": platform.python_version()}
        try:
            from core.version import APP_VERSION
            versions["studio"] = APP_VERSION
        except ImportError:
            versions["studio"] = "unavailable"
        for name in ("torch", "torchvision", "onnx", "onnxruntime", "onnxsim"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "not installed"
        diagnostic = {"stage": events[-1], "error": str(error), "versions": versions,
                      "opset": opset_version, "dynamic_batch": dynamic_batch,
                      "platform": {"system": platform.system(), "machine": platform.machine()},
                      "events": events, "traceback": traceback.format_exc()}
        if hasattr(error, "numerical_diagnostic"):
            diagnostic["numerical_diagnostic"] = error.numerical_diagnostic
        path = Path(output_path).with_suffix(".export-error.json")
        report_location = ""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
            report(f"내보내기 진단 저장: {path}")
            report_location = f"\n상세 진단 파일: {path}"
        except OSError:
            report(json.dumps(diagnostic, ensure_ascii=False))
        raise ValueError(f"ONNX 내보내기 실패 [{diagnostic['stage']}]: {error}{report_location}") from error
    try:
        Path(output_path).with_suffix(".export-error.json").unlink(missing_ok=True)
    except OSError:
        report("이전 진단 파일 삭제 실패: ONNX 내보내기는 완료되었습니다.")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="체크포인트 ONNX 내보내기")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="efficientnet.onnx")
    parser.add_argument("--task", choices=list(OUTPUT_NAMES))
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--in_channels", type=int)
    parser.add_argument("--input_size", type=int)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic_batch", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bundle", help="optional .dvdeploy output directory for C++/C# SDK")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    export_checkpoint(args.checkpoint, args.output, args.opset, args.dynamic_batch,
                      args.simplify, args.verify,
                      {key: getattr(args, key) for key in ("task", "num_classes", "in_channels", "input_size")},
                      bundle_output=args.bundle)


if __name__ == "__main__":
    main()
