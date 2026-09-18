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

# CPU FP32 ONNX kernels may reassociate fused convolution/normalization
# operations. A 1e-3 absolute floor is the usual FP32 deployment parity
# boundary; top-1 is checked separately so this does not hide a class change.
# Other tasks retain the historical strict profile until their output contracts
# get a task-specific parity test.
VERIFY_TOLERANCES = {
    "default": {"atol": 1e-4, "rtol": 1e-4},
    "classify": {"atol": 1e-3, "rtol": 5e-4},
}


def verification_tolerances(task="classify"):
    """Return the explicit numerical parity profile for an exported task."""
    profile = VERIFY_TOLERANCES.get(task, VERIFY_TOLERANCES["default"])
    return dict(profile)


def validate_classification_outputs(expected, actual):
    """Validate logits and require the same top-1 class for every sample."""
    validate_outputs(expected, actual, **verification_tolerances("classify"))
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
    if "model_state_dict" in checkpoint:
        if checkpoint.get("engine") == "efficientnet":
            return "efficientnet"
        return "custom"
    raise ValueError("지원하지 않는 체크포인트 포맷")


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
    if backend not in {"custom", "efficientnet", "builtin"}:
        raise ValueError("커스텀 또는 기본 모델 체크포인트 필요")
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    preprocessing = dict(checkpoint.get("preprocessing") or {})
    from center_crop import checkpoint_center_crop
    crop = checkpoint_center_crop(checkpoint)
    if crop:
        preprocessing["center_crop"] = crop
    model_config = dict(checkpoint.get("model_config") or {})
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

    task = select("task")
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
                   dynamic_batch=False, task="classify"):
    import torch
    from torch.onnx import _constants
    maximum = getattr(_constants, "ONNX_MAX_OPSET", 17)
    if type(opset_version) is not int or not 11 <= opset_version <= maximum:
        raise ValueError(f"설치된 PyTorch의 ONNX opset 범위: 11~{maximum}. 기본값 17을 사용하세요.")
    output_name = OUTPUT_NAMES[task]
    axes = {"input_image": {0: "batch_size"}, output_name: {0: "batch_size"}} if dynamic_batch else None
    model.cpu().eval()
    # 명시적 legacy exporter로 버전별 기본값 변화와 외부 데이터 분리를 피한다.
    exporter_options = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    if "external_data" in inspect.signature(torch.onnx.export).parameters:
        exporter_options["external_data"] = False
    with torch.no_grad():
        torch.onnx.export(model, dummy_input.cpu(), str(output_path), export_params=True,
                          opset_version=opset_version, do_constant_folding=True,
                          input_names=["input_image"], output_names=[output_name],
                          dynamic_axes=axes, **exporter_options)
    return str(output_path)


def verify_onnx(onnx_path, dummy_input, pytorch_model, task="classify", atol=1e-4):
    import torch
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    pytorch_model.cpu().eval()
    with torch.no_grad():
        expected = pytorch_model(dummy_input.cpu()).detach().numpy()
    actual = session.run(None, {session.get_inputs()[0].name: dummy_input.cpu().numpy()})
    if len(actual) != 1:
        raise ValueError("커스텀 모델은 출력 텐서 하나만 지원합니다.")
    if task == "classify" and atol == 1e-4:
        return validate_classification_outputs(expected, actual[0])
    tolerances = verification_tolerances(task)
    # Preserve an explicit caller override for compatibility with tooling that
    # asks for a stricter absolute floor.
    tolerances["atol"] = atol
    return validate_outputs(expected, actual[0], **tolerances)


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
    if backend not in {"custom", "builtin", "efficientnet", "patchcore", "redetr_v4"}:
        raise ValueError("지원하지 않는 모델 형식입니다. 등록된 모델 체크포인트를 선택하세요.")
    if backend == "redetr_v4" and task != "detect":
        raise ValueError("Re-DETR v4 backend는 detect 태스크만 지원합니다.")
    if backend == "redetr_v4" and (output_names is None or len(output_names) != 2):
        raise ValueError("Re-DETR v4는 pred_boxes와 pred_logits 두 출력 이름이 필요합니다.")
    detection_cpp_supported = task != "detect" or detection_box_encoding in {
        "grid_sigmoid_xywh", "normalized_cxcywh"
    }
    if backend == "redetr_v4":
        detection_cpp_supported = detection_box_encoding in {None, "normalized_cxcywh", "normalized_xyxy"}
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
                         (backend == "efficientnet" and task == "classify") or
                         (backend == "redetr_v4" and task == "detect" and detection_cpp_supported),
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
    elif backend == "redetr_v4":
        config["postprocessing"] = {
            "box_format": detection_box_encoding or "normalized_cxcywh",
            "objectness": "none", "class_scores": "sigmoid",
            "confidence": "class_score", "class_aware_nms": True,
            "confidence_threshold": .25, "iou_threshold": .5, "max_detections": 300,
        }
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
    if backend not in {"custom", "efficientnet", "builtin", "patchcore"}:
        raise ValueError("지원하지 않는 모델 형식입니다. 등록된 모델 체크포인트를 선택하세요.")
    if output.resolve() == Path(checkpoint_path).resolve():
        raise ValueError("체크포인트와 ONNX 출력 경로가 같을 수 없습니다.")
    if backend == "patchcore":
        from export_patchcore_onnx import export_patchcore_checkpoint
        return export_patchcore_checkpoint(checkpoint_path, output_path, verify=verify,
                                           opset=opset_version, simplify=simplify, log=log)
    config_name = output.with_suffix(".json").name
    with tempfile.TemporaryDirectory(prefix=".onnx-export-", dir=output.parent) as temp:
        stage = Path(temp)
        staged_model = stage / output.name
        log(f"내보내기 시작: {backend}")
        if backend in {"custom", "efficientnet", "builtin"}:
            spec = resolve_checkpoint_spec(checkpoint, overrides)
            model = load_custom_model(checkpoint, spec)
            if backend == "efficientnet":
                from efficientnet import prepare_for_inference
                model = prepare_for_inference(model)
                log(f"추론 최적화: {model.inference_optimization}")
            generator = torch.Generator().manual_seed(42)
            dummy = torch.randn(1, spec["in_channels"], spec["input_height"], spec["input_width"], generator=generator)
            log(f"ONNX 그래프 생성: opset={opset_version}, 입력={tuple(dummy.shape)}")
            export_to_onnx(model, dummy, staged_model, opset_version, dynamic_batch, spec["task"])
            if simplify:
                simplify_onnx(staged_model)
            if verify:
                # Compare against the exact PyTorch graph that was exported.
                # Conv/BatchNorm fusion is mathematically equivalent but can
                # reassociate FP32 operations; comparing it with the unfused
                # checkpoint made valid graphs fail at near-zero logits.
                log("ONNX Runtime 검증: 내보낸 PyTorch 그래프와 비교")
                for probe in (dummy, torch.zeros_like(dummy)):
                    if not verify_onnx(staged_model, probe, model, task=spec["task"]):
                        raise ValueError("ONNX 검증 실패")
                if dynamic_batch:
                    if not verify_onnx(staged_model, dummy.repeat(2, 1, 1, 1), model, task=spec["task"]):
                        raise ValueError("동적 배치 ONNX 검증 실패")
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
            if spec["task"] == "detect":
                metadata["detection_box_encoding"] = spec["model_config"].get("detection_box_encoding", "legacy_raw")
        else:
            raise ValueError(f"내보내기 미지원: {backend}")
        onnx.checker.check_model(str(staged_model))
        staged_config = create_inference_config(stage, onnx_filename=output.name,
            backend=backend, verification="passed" if verify else "skipped",
            config_filename=config_name, **metadata)
        manifest = json.loads(Path(staged_config).read_text(encoding="utf-8"))
        manifest["export"] = {"opset": opset_version, "precision": "float32", "dynamic_batch": dynamic_batch}
        if backend == "efficientnet":
            manifest["export"]["optimization"] = model.inference_optimization
            manifest["export"]["verification_reference"] = "exported_pytorch_graph"
        Path(staged_config).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        # 검증 실패 시 기존 파일은 그대로 유지된다. 각 파일은 같은 파일시스템에서 교체한다.
        os.replace(staged_model, output)
        config_path = output.with_suffix(".json")
        os.replace(staged_config, config_path)
    result = {"output_path": str(output), "config_path": str(config_path),
              "file_size_mb": output.stat().st_size / (1024 * 1024),
              "backend": backend, "verification": "passed" if verify else "skipped",
              "task": metadata["task"], "cpp_supported": manifest["cpp_supported"]}
    log(f"내보내기 완료: {output}\n배포 설정: {config_path}\n검증: {result['verification']}")
    return result


def export_checkpoint(checkpoint_path, output_path, opset_version=17, dynamic_batch=False,
                      simplify=False, verify=True, overrides=None, log=print):
    """실패 단계와 환경을 자동 기록한다. 검증 실패한 모델은 배포하지 않는다."""
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
    except Exception as error:
        versions = {"python": platform.python_version()}
        for name in ("torch", "torchvision", "onnx", "onnxruntime", "onnxsim"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "not installed"
        diagnostic = {"stage": events[-1], "error": str(error), "versions": versions,
                      "opset": opset_version, "dynamic_batch": dynamic_batch,
                      "events": events, "traceback": traceback.format_exc()}
        path = Path(output_path).with_suffix(".export-error.json")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
            report(f"내보내기 진단 저장: {path}")
        except OSError:
            report(json.dumps(diagnostic, ensure_ascii=False))
        raise ValueError(f"ONNX 내보내기 실패 [{diagnostic['stage']}]: {error}") from error
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
    return parser.parse_args(argv)


def main():
    args = parse_args()
    export_checkpoint(args.checkpoint, args.output, args.opset, args.dynamic_batch,
                      args.simplify, args.verify,
                      {key: getattr(args, key) for key in ("task", "num_classes", "in_channels", "input_size")})


if __name__ == "__main__":
    main()
