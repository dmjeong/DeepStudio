"""CLI 측정과 평가를 위한 Qt 없는 체크포인트 로더."""

from pathlib import Path
import math
import torch
from core.inference_engine import InferenceEngine
from core.gradcam import GradCAM
from core.inference_settings import require_saved_input_shape


def _sam2_model_id(path, checkpoint):
    """Studio SAM2 미세조정 체크포인트는 model_id를 담는다. 공식 파일(sam2.1_hiera_*.pt)은 파일 이름으로 찾는다."""
    from sam2_assets import SAM2_ASSETS
    if checkpoint.get("backend") == "sam2" or checkpoint.get("type") in {"sam2", "sam2_finetune"}:
        model_id = checkpoint.get("model_id")
        if model_id not in SAM2_ASSETS:
            raise ValueError(f"SAM2 체크포인트의 모델 ID를 알 수 없습니다: {model_id}")
        return model_id
    if isinstance(checkpoint.get("model"), dict):
        for model_id, asset in SAM2_ASSETS.items():
            if Path(path).name == asset.filename:
                return model_id
    return None


def load_inference_engine(path, *, gradcam=False, input_size=None, runtime="pytorch", official=False, device="cpu",
                          input_region=None, threads=4, segment_review=None, sam2=None):
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
    sam2_model_id = _sam2_model_id(path, checkpoint)
    if sam2_model_id is not None:
        # SAM2: 공식 Hiera 가중치(설치본·가중치 폴더) 위에 미세조정한 프롬프트·마스크 디코더를 얹는다.
        if runtime not in ("auto", "pytorch"):
            raise ValueError("SAM2 추론은 PyTorch 런타임을 사용합니다")
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2_assets import load_sam2_checkpoint, load_sam2_pretrained
        if checkpoint.get("backend") == "sam2" or checkpoint.get("type") == "sam2_finetune":
            model = load_sam2_checkpoint(sam2_model_id, device=str(device), checkpoint_path=path)
        else:
            model = load_sam2_pretrained(sam2_model_id, device=str(device), checkpoint_path=path)
        names = checkpoint.get("class_names") or []
        state.update(_sam2_predictor=SAM2ImagePredictor(model), _sam2_model_id=sam2_model_id,
                     class_names=list(names), _input_size=(1024, 1024), _center_crop=None)
        engine = InferenceEngine(state, gradcam=False, input_region=input_region, runtime="pytorch",
                                 threads=threads, segment_review=segment_review, sam2=sam2)
        engine.checkpoint_name = Path(path).name
        return engine
    if checkpoint.get("type") == "patchcore":
        if runtime not in ("auto", "pytorch"):
            raise ValueError("PatchCore CPU 측정은 pytorch 런타임 사용")
        from patchcore import PatchCore
        model = PatchCore.load(str(path), device=device)
        require_saved_input_shape(input_size, model.input_size)
        state.update(_patchcore_model=model, _anomaly_threshold=model.normalized_threshold,
                     _input_size=(model.input_size, model.input_size))
    else:
        from upstream_models import is_upstream_checkpoint
        if is_upstream_checkpoint(checkpoint):
            if runtime not in ("auto", "pytorch"):
                raise ValueError("LibreYOLO .pt 추론은 현재 PyTorch 런타임을 사용합니다")
            from upstream_models import load_upstream_checkpoint
            task = str(checkpoint["task"])
            names = checkpoint.get("names", {})
            if isinstance(names, dict):
                class_names = [str(names.get(index, names.get(str(index), f"Class {index}")))
                               for index in range(int(checkpoint["nc"]))]
            elif isinstance(names, (list, tuple)):
                class_names = [str(name) for name in names]
            else:
                raise ValueError("LibreYOLO 체크포인트의 클래스 이름 형식 오류")
            size = int(checkpoint.get("imgsz", 0))
            if size < 1:
                raise ValueError("LibreYOLO 체크포인트의 입력 크기(imgsz) 누락")
            require_saved_input_shape(input_size, (size, size))
            model = load_upstream_checkpoint(path, device=str(device))
            state.update(_upstream_model=model, class_names=class_names, _input_size=(size, size),
                         _normalization=None, _upstream_task=task)
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
                         _tiling=(preprocessing.get("tiling") or checkpoint.get("tiling")
                                  if spec["task"] == "segment" else None),
                         _input_size=(spec["input_height"], spec["input_width"]),
                         _normalization=(mean, std) if mean is not None and std is not None else None,
                         _anomaly_threshold=checkpoint.get("anomaly_threshold"))
            if gradcam:
                state["_gradcam"] = GradCAM(model)
    engine = InferenceEngine(state, gradcam=gradcam, input_region=input_region, runtime=runtime, threads=threads,
                             segment_review=segment_review, sam2=sam2)
    engine.checkpoint_name = Path(path).name
    return engine


def load_cpu_engine(path, *, gradcam=False, input_size=None, runtime="pytorch", official=False, input_region=None, threads=4):
    return load_inference_engine(path, gradcam=gradcam, input_size=input_size,
                                 runtime=runtime, official=official, device="cpu", input_region=input_region, threads=threads)
