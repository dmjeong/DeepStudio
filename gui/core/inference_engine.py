"""Qt 객체를 참조하지 않는 모델 추론과 Grad-CAM 계산."""
import os
import time
from dataclasses import replace
import numpy as np
import torch
import torch.nn.functional as F
from core.paths import ensure_python_path
from core.gradcam import GradCAM
from core.heatmap import normalize_activation_map, reproject_classification_cam
from core.inference_timing import StageTimer
from core.inference_types import InferenceResult, make_anomaly_result
ensure_python_path()
from center_crop import center_crop_box, load_crop_image, restore_detections, paste_crop_preview


class InferenceOperations:
    def _run_custom_inference(self, image_path: str):
        from torchvision import transforms
        from opencv_preprocess import OpenCVResize, read_image
        # 채널별 정규화 상수 임포트
        # core.paths로 이미 등록됨 (PyInstaller EXE 호환)
        from dataset import get_normalize_params

        with self._measure_inference_stage("inference"):
            phase_started = time.perf_counter()
            # 모델의 in_channels 확인 (체크포인트에서 로드된 값)
            in_channels = getattr(self.model, 'in_channels', 3)

            # 전처리 (학습과 동일 — 채널 수에 맞게)
            input_size = self._input_size
            mean, std = self._normalization or get_normalize_params(in_channels)
            transform = transforms.Compose([
                OpenCVResize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

            device = self._infer_device
            # 실행 시작 시 선택한 경우에만 계산한다. 결과 탐색은 저장된 맵만 사용한다.
            use_gradcam = self._gradcam is not None and getattr(self, "gradcam_enabled", True)

            # 채널 수에 따라 RGB 또는 Grayscale 로드
            crop = getattr(self, "_center_crop", None)
            decode_started = time.perf_counter()
            original = read_image(image_path, in_channels)
            decoded = time.perf_counter()
            box = center_crop_box((original.shape[1], original.shape[0]), crop)
            img = original[box[1]:box[3], box[0]:box[2]]

            # 입력 텐서를 선택된 디바이스로 전송
            input_tensor = transform(img).unsqueeze(0).to(device)
            prepared = time.perf_counter()
            onnx = getattr(self, "_onnx_runtime", None)
            with torch.inference_mode():
                output = (torch.from_numpy(onnx.logits(input_tensor.numpy())) if onnx is not None
                          else self.model(input_tensor))
            inferred = time.perf_counter()
            if not torch.isfinite(output).all():
                raise ValueError("모델 출력이 유한하지 않습니다")
            target_tensor = None
            detections = None

            task = self.model.task

            if task == "classify":
                # 결과를 CPU로 가져와서 numpy 변환
                probs = F.softmax(output.detach(), dim=1)[0].cpu().numpy()
                names = self.class_names if self.class_names else [
                    f"Class {i}" for i in range(len(probs))
                ]
                top = int(output[0].argmax().item())
                prediction = InferenceResult(image_path, "ok", task,
                                             f"{names[top]} {probs[top]:.1%}",
                                             details={"class_names": list(names), "probabilities": probs.tolist()})
                if hasattr(self.model, "checkpoint_config"):
                    prediction.details["model_config"] = self.model.checkpoint_config()

            elif task == "segment":
                pred_mask = output.detach().argmax(dim=1)[0].cpu().numpy()
                unique, counts = np.unique(pred_mask, return_counts=True)
                pixel_counts = dict(zip(unique.tolist(), counts.tolist()))
                prediction = InferenceResult(image_path, "ok", task, "세그멘테이션", "#A78BFA",
                                             details={"num_classes": self.model.num_classes,
                                                      "pixel_counts": pixel_counts})

            elif task == "detect":
                from detection import postprocess_detections
                detections = postprocess_detections(output[0].detach().float().cpu().numpy(),
                                                     confidence=getattr(self, "detection_confidence", 0.25),
                                                     iou_threshold=getattr(self, "detection_iou", 0.5))
                prediction = InferenceResult(image_path, "ok", task, f"{len(detections)}개 탐지",
                                             details={"num_detections": len(detections),
                                                      "detections": detections,
                                                      "class_names": list(self.class_names)})

            elif task == "anomaly":
                recon = output.detach()
                # ── 비정규화 원본 [0,1] 텐서 생성 ──────────────
                # 모델은 MSE(model(normalized), original_[0,1])로
                # 학습됨. 추론 시에도 output을 원본 [0,1]과 비교해야
                # 정확한 이상 스코어가 산출됨.
                # normalized input과 비교하면 스케일이 맞지 않아
                # 이상 맵이 왜곡됨.
                unnorm_transform = transforms.Compose([
                    OpenCVResize(input_size),
                    transforms.ToTensor(),  # [0,1] 범위, Normalize 없음
                ])
                target_tensor = unnorm_transform(img).unsqueeze(0).to(device)
                anomaly_map = (target_tensor - recon).pow(2).mean(dim=1)
                score = anomaly_map.mean().item()
                prediction = make_anomaly_result(image_path, score, self._anomaly_threshold)
            else:
                raise ValueError(f"미지원 태스크: {task}")
            prediction = replace(prediction, details=dict(prediction.details or {}))
            prediction.details.update(runtime="onnxruntime" if onnx is not None else "pytorch", device=str(device))
            if str(device) == "cpu":
                finished = time.perf_counter()
                prediction.details.update(runtime="onnxruntime" if onnx is not None else "pytorch", device="cpu",
                    timing_ms={"decode": (decoded-decode_started)*1000,
                               "preprocess": ((prepared-phase_started)-(decoded-decode_started))*1000,
                               "model": (inferred-prepared)*1000, "postprocess": (finished-inferred)*1000})
                if onnx is not None:
                    prediction.details.update(runtime_version=onnx.version, runtime_threads=onnx.threads,
                                              runtime_setup_sec=onnx.setup_sec,
                                              runtime_optimization=onnx.optimization_level,
                                              runtime_validation_attempts=onnx.validation_attempts)

        # 화면과 컬러 히트맵용 변환은 모델 추론 후에만 수행한다.
        from core.image_display import display_rgb
        original_np = display_rgb(original)

        from core.spatial_preview import draw_detection_boxes, segmentation_preview
        base_preview = original_np
        if crop:
            box = center_crop_box((original_np.shape[1], original_np.shape[0]), crop)
            prediction = replace(prediction, details=dict(prediction.details or {}))
            prediction.details.update(center_crop=crop, crop_box=list(box),
                                      original_size=[original_np.shape[1], original_np.shape[0]])
        if task == "detect":
            if crop:
                detections = restore_detections(detections, (original_np.shape[1], original_np.shape[0]), crop)
                prediction.details["detections"] = detections
            base_preview = draw_detection_boxes(original_np, detections)
        elif task == "segment":
            region = original_np[box[1]:box[3], box[0]:box[2]] if crop else original_np
            base_preview, counts = segmentation_preview(pred_mask, region, self.model.num_classes)
            if crop:
                base_preview = paste_crop_preview(original_np, base_preview, crop)
            prediction.details.update(pixel_counts=counts, class_names=list(self.class_names))

        # ── Grad-CAM 히트맵 생성 & 오버레이 ──
        if use_gradcam:
            try:
                with self._measure_inference_stage("gradcam"):
                    activation = self._gradcam.generate_activation(
                        input_tensor, target_tensor=target_tensor
                    )
                valid_mask = None
                if crop:
                    from core.heatmap import reproject_center_crop
                    activation, valid_mask = reproject_center_crop(activation, original_np.shape[:2], crop)
                self._set_heatmap_preview(
                    image_path, original_np, activation, "Grad-CAM",
                    "타겟: backbone.stage4" + (" | 체크 해제: 클래스 분할 마스크" if task == "segment" else ""),
                    base_image=base_preview, detections=detections, valid_mask=valid_mask
                )
            except Exception as e:
                self._clear_heatmap_preview()
                self._set_info(f"Grad-CAM 실패: {e}")
                # Grad-CAM 실패 시 원본 이미지 표시
                self._display_numpy_image(base_preview)
        else:
            self._set_info("")
            self._display_numpy_image(base_preview)

        if crop:
            self._set_info((getattr(self, "info", "") or "") + f" | 판정 범위: 중앙 {crop['width']}×{crop['height']} px")
        return prediction

    def _run_patchcore_inference(self, image_path: str):
        """
        PatchCore 이상 탐지 추론 — 자체 XAI 히트맵 포함

        PatchCore 추론 흐름:
        ┌──────────────────────────────────────────────────────────┐
        │ 이미지 → 백본 특징 추출 → 메모리 뱅크와 kNN 거리 계산    │
        │       → 이상 스코어 + 공간 이상 히트맵 생성               │
        │       → 히트맵을 원본에 오버레이하여 이상 영역 시각화      │
        │                                                          │
        │ ※ 히트맵 자체가 XAI: "어디가 왜 이상한지" 직접 표현      │
        │   → 별도 Grad-CAM 불필요 (PatchCore에는 역전파 없음)     │
        └──────────────────────────────────────────────────────────┘
        """
        from patchcore_data import input_image
        from center_crop import center_crop_box
        from core.heatmap import reproject_center_crop

        crop = getattr(self, "_center_crop", self._patchcore_model.center_crop)
        with self._measure_inference_stage("inference"):
            scores, anomaly_maps = self._patchcore_model.predict_from_path(image_path, center_crop=crop)
            from patchcore_scores import normalize_score, normalized_details
            raw_score = float(scores[0])
            normalization = self._patchcore_model.get_score_normalization()
            score = normalize_score(raw_score, normalization)
            amap = anomaly_maps[0]
            prediction = replace(make_anomaly_result(image_path, score, self._anomaly_threshold),
                                 details=normalized_details(raw_score, self._patchcore_model.anomaly_threshold, normalization))

        with input_image(image_path, self._patchcore_model.preprocessing) as source:
            original_np = np.array(source.convert("RGB"))

        activation = normalize_activation_map(amap)
        valid_mask = None
        region = "전체 이미지"
        if crop is not None:
            activation, valid_mask = reproject_center_crop(activation, original_np.shape[:2], crop)
            box = center_crop_box((original_np.shape[1], original_np.shape[0]), crop)
            prediction.details.update(center_crop=crop, crop_box=list(box),
                                      original_size=[original_np.shape[1], original_np.shape[0]])
            region = f"중앙 {crop['width']}×{crop['height']} px"
        # 표시 범위는 이미지별 상대 강도에만 적용하며 원래 판정 점수는 유지한다.
        self._set_heatmap_preview(
            image_path, original_np, activation,
            "PatchCore", f"스코어 (0~1): {score:.4f} | 판정 범위: {region}", valid_mask=valid_mask
        )

        return prediction




class InferenceEngine(InferenceOperations):
    """한 워커가 모델과 훅을 단독 사용하고 결과만 화면에 전달한다."""
    STATE_FIELDS = ("model", "_patchcore_model", "_gradcam",
                    "_infer_device", "_input_size", "_normalization", "class_names",
                    "_anomaly_threshold", "_active_checkpoint", "_center_crop", "_onnx_runtime")

    def __init__(self, state, *, gradcam=True, input_region=None, runtime="auto", threads=4):
        for name in self.STATE_FIELDS:
            setattr(self, name, state.get(name))
        self.gradcam_enabled = gradcam
        if runtime not in ("auto", "pytorch", "onnx"):
            raise ValueError("런타임은 auto, pytorch, onnx 중에서 선택하세요")
        self.runtime_requested, self.runtime_threads = runtime, threads
        self.runtime = "pytorch"
        self.runtime_warning = ""
        self._prepared = False
        if runtime == "pytorch":
            self._onnx_runtime = None
        self._heatmap_cache = None
        self._current_preview_rgb = None
        self.info = ""
        from center_crop import validate_center_crop
        self._saved_center_crop = validate_center_crop(
            self._patchcore_model.center_crop if self._patchcore_model is not None else self._center_crop)
        self.configure_input_region(input_region)

    def prepare(self):
        """One-time export/validation/warmup runs on the worker before per-image timing."""
        if self._prepared:
            return
        from efficientnet import EfficientNet
        eligible = (isinstance(self.model, EfficientNet) and str(self._infer_device) == "cpu")
        if self.runtime_requested == "onnx" and not eligible:
            raise ValueError("이 ONNX 경로는 CPU EfficientNet 분류 모델을 지원합니다")
        if eligible and self.runtime_requested != "pytorch":
            try:
                from core.efficientnet_onnx import EfficientNetOnnx
                if self._onnx_runtime is None or not self._onnx_runtime.matches(self.model, self._input_size, self.runtime_threads):
                    self._onnx_runtime = EfficientNetOnnx(self.model, self._input_size, threads=self.runtime_threads)
            except Exception as exc:
                self._onnx_runtime = None
                if self.runtime_requested == "onnx":
                    raise
                # An optional accelerator must not disable the existing model.
                # Keep the failed validation visible; never use an unverified session.
                self.runtime = "pytorch"
                self.runtime_warning = f"ONNX 준비 실패로 PyTorch에서 실행합니다.\n{type(exc).__name__}: {exc}"
            else:
                self.runtime = "onnxruntime"
        else:
            self._onnx_runtime = None
        self._prepared = True

    def configure_input_region(self, request=None):
        """Change this engine's input without modifying model/checkpoint metadata."""
        from core.inference_region import resolve_input_region
        region = resolve_input_region(request, self._saved_center_crop, self._input_size)
        self.input_region = region
        self._center_crop = region["center_crop"]

    def _set_info(self, text):
        self.info = text

    def _display_numpy_image(self, image):
        from core.image_display import display_rgb
        self._current_preview_rgb = display_rgb(image)

    def _clear_heatmap_preview(self):
        self._heatmap_cache = None

    def _set_heatmap_preview(self, image_path, original, activation, kind, info,
                            base_image=None, valid_mask=None, detections=None):
        self._heatmap_cache = dict(image_path=image_path, original=original,
                                  activation=activation, kind=kind, info=info,
                                  base_image=base_image, valid_mask=valid_mask, detections=detections)
        self._current_preview_rgb = original if base_image is None else base_image
        self.info = info

    def _measure_inference_stage(self, stage):
        timer = StageTimer(self._infer_device)
        self._stage_timings[stage] = timer
        return timer

    def infer(self, image_path):
        started = time.perf_counter()
        self._stage_timings = {}
        self._heatmap_cache = self._current_preview_rgb = None
        self.info = ""
        try:
            self.prepare()
            if self._patchcore_model is not None:
                result = self._run_patchcore_inference(image_path)
            elif self.model is not None:
                result = self._run_custom_inference(image_path)
            else:
                raise ValueError("모델 미로드")
        except Exception as exc:
            self._heatmap_cache = None
            result = InferenceResult(image_path, "error", "", "ERROR", "#E05555", error=str(exc))
        from copy import deepcopy
        from core.inference_region import input_region_label
        result = replace(result, details={**(result.details or {}), "input_region": deepcopy(self.input_region)})
        if self.runtime_warning:
            result.details.update(runtime=self.runtime, device=str(self._infer_device),
                                  runtime_requested=self.runtime_requested, runtime_warning=self.runtime_warning)
        self.info = " | ".join(filter(None, (self.info, input_region_label(self.input_region))))
        if self._heatmap_cache is not None:
            self._heatmap_cache["info"] = self.info
        timings = {}
        for name, timer in self._stage_timings.items():
            timings[name + "_sec"] = timer.elapsed_sec
            timings[name + "_status"] = timer.status
        if "gradcam_status" not in timings:
            timings["gradcam_status"] = "not_run" if not self.gradcam_enabled else "unsupported"
        return replace(result, elapsed_sec=time.perf_counter() - started, **timings)
