"""CLI 측정과 평가를 위한 Qt 없는 체크포인트 로더."""

from pathlib import Path
import math
import torch
from core.inference_engine import InferenceEngine
from core.gradcam import GradCAM
from core.inference_settings import require_saved_input_shape


def load_inference_engine(path, *, gradcam=False, input_size=None, runtime="pytorch", official=False, device="cpu",
                          input_region=None, threads=4):
    if official:
        raise ValueError("지원하지 않는 모델 형식입니다. EfficientNet 체크포인트를 선택하세요.")
    device = torch.device(device)
    if runtime not in ("auto", "pytorch") and device.type != "cpu":
        raise ValueError("내보낸 런타임의 이 로더는 CPU 장치 사용")
    state = {"_infer_device": device, "_active_checkpoint": str(path), "class_names": []}
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    from center_crop import checkpoint_center_crop
    state["_center_crop"] = checkpoint_center_crop(checkpoint)
    if not isinstance(checkpoint, dict):
        raise ValueError("모델 설정이 포함된 체크포인트 필요")
    elif checkpoint.get("type") == "patchcore":
        if runtime not in ("auto", "pytorch"):
            raise ValueError("PatchCore CPU 측정은 pytorch 런타임 사용")
        from patchcore import PatchCore
        model = PatchCore.load(str(path), device=device)
        require_saved_input_shape(input_size, model.input_size)
        state.update(_patchcore_model=model, _anomaly_threshold=model.normalized_threshold,
                     _input_size=(model.input_size, model.input_size))
    else:
        if runtime not in ("auto", "pytorch") and not (runtime == "onnx" and checkpoint.get("engine") == "efficientnet"):
            raise ValueError("이 ONNX 경로는 EfficientNet 체크포인트를 지원합니다")
        from export_onnx import resolve_checkpoint_spec, load_custom_model
        spec = resolve_checkpoint_spec(checkpoint)
        threshold = checkpoint.get("anomaly_threshold")
        if threshold is not None and not math.isfinite(float(threshold)):
            raise ValueError("이상 판정 임계값이 유한하지 않습니다")
        if checkpoint.get("threshold_comparator", ">=") != ">=":
            raise ValueError("미지원 임계값 비교 연산")
        if spec["task"] == "anomaly" and checkpoint.get("score_definition", "reconstruction_mse_mean") != "reconstruction_mse_mean":
            raise ValueError("이상 점수 정의 불일치")
        require_saved_input_shape(input_size, (spec["input_height"], spec["input_width"]))
        model = load_custom_model(checkpoint, spec).to(device).eval()
        preprocessing = spec["preprocessing"]
        mean = preprocessing.get("normalize_mean", preprocessing.get("mean"))
        std = preprocessing.get("normalize_std", preprocessing.get("std"))
        state.update(model=model, class_names=spec["class_names"],
                     _input_size=(spec["input_height"], spec["input_width"]),
                     _normalization=(mean, std) if mean is not None and std is not None else None,
                     _anomaly_threshold=checkpoint.get("anomaly_threshold"))
        if gradcam:
            state["_gradcam"] = GradCAM(model)
    engine = InferenceEngine(state, gradcam=gradcam, input_region=input_region, runtime=runtime, threads=threads)
    engine.checkpoint_name = Path(path).name
    return engine


def load_cpu_engine(path, *, gradcam=False, input_size=None, runtime="pytorch", official=False, input_region=None, threads=4):
    return load_inference_engine(path, gradcam=gradcam, input_size=input_size,
                                 runtime=runtime, official=official, device="cpu", input_region=input_region, threads=threads)
