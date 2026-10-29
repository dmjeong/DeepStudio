"""Qt 객체를 참조하지 않는 모델 추론과 Grad-CAM 계산."""
import time
from dataclasses import replace
import numpy as np
import torch
import torch.nn.functional as F
from core.paths import ensure_python_path
from core.heatmap import normalize_activation_map
from core.inference_timing import StageTimer
from core.inference_types import InferenceResult, make_anomaly_result
ensure_python_path()
from center_crop import center_crop_box, restore_detections, paste_crop_preview


def _out_of_memory(exc):
    """PyTorch CPU·CUDA 메모리 부족 예외인지 (메시지가 버전마다 달라 여러 형태를 본다)."""
    if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError":
        return True
    message = str(exc).lower()
    return "out of memory" in message or "can't allocate memory" in message or "not enough memory" in message


class InferenceOperations:
    def _run_upstream_inference(self, image_path: str):
        """Adapt public LibreYOLO Results to Deep Vision Studio's result contract."""
        from PIL import Image
        from core.image_display import display_rgb
        from core.spatial_preview import draw_detection_boxes

        def as_numpy(value):
            value = value.detach().cpu().numpy() if hasattr(value, "detach") else value
            return np.asarray(value)

        model = self._upstream_model
        task = self._upstream_task
        with self._measure_inference_stage("inference"):
            result = model.predict(image_path, imgsz=int(self._input_size[0]),
                                   conf=float(getattr(self, "detection_confidence", .25)),
                                   iou=float(getattr(self, "detection_iou", .5)),
                                   device=str(self._infer_device))
        if isinstance(result, (list, tuple)):
            if len(result) != 1:
                raise ValueError("LibreYOLO 단일 이미지 추론 결과 형식 오류")
            result = result[0]
        with Image.open(image_path) as source:
            original = display_rgb(np.asarray(source.convert("RGB")))
        if task == "classify":
            probs = as_numpy(result.probs.data).reshape(-1).astype(np.float64)
            if not len(probs) or not np.isfinite(probs).all():
                raise ValueError("LibreYOLO 분류 확률 출력 오류")
            top = int(np.argmax(probs))
            names = self.class_names or [f"Class {index}" for index in range(len(probs))]
            if top >= len(names):
                raise ValueError("LibreYOLO 분류 클래스 목록 불일치")
            prediction = InferenceResult(image_path, "ok", task, f"{names[top]} {probs[top]:.1%}",
                                         details={"class_names": list(names), "probabilities": probs.tolist(),
                                                  "runtime": "libreyolo-pytorch", "device": str(self._infer_device)})
            self._current_preview_rgb = original
        elif task == "detect":
            if result.boxes is None:
                boxes = np.empty((0, 4), dtype=np.float32)
                scores = classes = np.empty(0, dtype=np.float32)
            else:
                boxes = as_numpy(result.boxes.xyxyn).reshape(-1, 4)
                scores = as_numpy(result.boxes.conf).reshape(-1)
                classes = as_numpy(result.boxes.cls).reshape(-1)
            if not (len(boxes) == len(scores) == len(classes)):
                raise ValueError("LibreYOLO 검출 출력 길이 불일치")
            detections = [{"class_id": int(cls), "confidence": float(score),
                           "bbox": np.clip(box, 0, 1).astype(float).tolist()}
                          for box, score, cls in zip(boxes, scores, classes)]
            prediction = InferenceResult(image_path, "ok", task, f"{len(detections)}개 탐지",
                                         details={"num_detections": len(detections), "detections": detections,
                                                  "class_names": list(self.class_names),
                                                  "runtime": "libreyolo-pytorch", "device": str(self._infer_device)})
            self._current_preview_rgb = draw_detection_boxes(original, detections)
        else:
            raise ValueError(f"LibreYOLO 태스크 미지원: {task}")
        self._heatmap_cache = None
        self.info = "LibreYOLO native PyTorch 추론 | Grad-CAM 미지원"
        return prediction

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
            # 분할·탐지는 결과 자체(마스크·박스)가 위치를 보여주므로 Grad-CAM을 계산하지 않는다.
            use_gradcam = (self._gradcam is not None and getattr(self, "gradcam_enabled", True)
                           and getattr(self.model, "task", "") not in ("segment", "detect"))

            # 채널 수에 따라 RGB 또는 Grayscale 로드
            crop = getattr(self, "_center_crop", None)
            decode_started = time.perf_counter()
            original = read_image(image_path, in_channels)
            decoded = time.perf_counter()
            box = center_crop_box((original.shape[1], original.shape[0]), crop)
            img = original[box[1]:box[3], box[0]:box[2]]

            onnx = getattr(self, "_onnx_runtime", None)
            # 패치 학습 모델: 줄이지 않고 원본 해상도 타일로 나눠 추론한다 (checkpoint의 tiling 규칙).
            tiling = getattr(self, "_tiling", None) if getattr(self.model, "task", "") == "segment" else None
            if tiling:
                input_tensor = None
                prepared = time.perf_counter()
                output = self._patch_probabilities(img, tiling, mean, std, device).unsqueeze(0)
            else:
                # 입력 텐서를 선택된 디바이스로 전송
                input_tensor = transform(img).unsqueeze(0).to(device)
                prepared = time.perf_counter()
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
                from core.segment_defects import _histogram
                pred_mask = output.detach().argmax(dim=1)[0].cpu().numpy()
                if self.model.num_classes <= 256:
                    pred_mask = pred_mask.astype(np.uint8)  # 원본 해상도 타일 결과도 가볍게 다룬다
                review = getattr(self, "segment_review", None)
                # 결함 점수용 클래스 확률 (모델 해상도, 타일이면 원본 해상도). 결함 분석을 켠 경우에만 만든다.
                # 타일 추론 출력은 이미 확률(겹친 타일 softmax 평균)이다.
                probabilities = None
                if review is not None:
                    probabilities = (output[0].float().cpu().numpy() if tiling
                                     else F.softmax(output.detach().float(), dim=1)[0].cpu().numpy())
                histogram = _histogram(pred_mask)
                pixel_counts = {int(i): int(histogram[i]) for i in np.flatnonzero(histogram)}
                prediction = InferenceResult(image_path, "ok", task, "세그멘테이션", "#A78BFA",
                                             details={"num_classes": self.model.num_classes,
                                                      "pixel_counts": pixel_counts})
                if tiling:
                    prediction.details["tiling"] = {**tiling, "tiles": self._tile_count, "run": self._patch_run,
                                                    "inference_size": list(pred_mask.shape[::-1])}

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
                    prediction.details.update(runtime_version=onnx.version, runtime_threads=onnx.runtime_threads,
                                              runtime_setup_sec=onnx.setup_sec,
                                              runtime_optimization=onnx.optimization_level,
                                              runtime_export_graph=onnx.export_graph,
                                              runtime_compute_precision=onnx.export_optimization.get("compute_precision", "float32"),
                                              runtime_verification_tolerance=onnx.verification_tolerance,
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
            from core.spatial_preview import segmentation_labels
            region = original_np[box[1]:box[3], box[0]:box[2]] if crop else original_np
            labels = segmentation_labels(pred_mask, region.shape[1], region.shape[0])
            # 결함 판정 화면은 배경을 칠하지 않아 결함만 눈에 띄게 한다 (결함 분석이 없는 요청은 기존 표시).
            base_preview, counts = segmentation_preview(pred_mask, region, self.model.num_classes, labels=labels,
                                                        tint_background=review is None)
            if crop:
                base_preview = paste_crop_preview(original_np, base_preview, crop)
            prediction.details.update(pixel_counts=counts, class_names=list(self.class_names))
            if review is not None:
                prediction = self._review_segmentation(prediction, image_path, pred_mask, labels, probabilities,
                                                       original_np.shape[1::-1], box if crop else None, review)

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
        if task == "segment" and tiling:
            scale = float(tiling.get("scale", 1.0))
            size = "×".join(str(v) for v in pred_mask.shape[::-1])
            how = ("이미지 한 번에" if self._patch_run == "whole" else
                   f"타일 {tiling['tile']}px × {self._tile_count}개"
                   + (" (메모리 부족으로 전환)" if self._patch_run == "tiles_after_oom" else ""))
            # 보간이 기록되지 않은 배율 모델 = 9.33 첫 빌드의 선형 축소로 학습 (1px 크랙이 위치에 따라 사라졌다)
            legacy = scale != 1 and "interpolation" not in tiling
            self._set_info((getattr(self, "info", "") or "") + " | 패치 모델: "
                           + ("원본 해상도" if scale == 1 else f"배율 {scale:.0%}") + f" {size} · {how}"
                           + (" · 이전 축소 방식(선형: 1px 결함이 자리에 따라 지워짐)으로 학습됨, 다시 학습 권장"
                              if legacy else ""))
        return prediction

    def _forward_probabilities(self, inputs, device):
        """모델 → softmax 확률 (CPU). GPU는 FP16 autocast로 계산하고, 결과가 유한하지 않으면 FP32로 다시 한다."""
        inputs = inputs.to(device)
        if torch.device(device).type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                probabilities = F.softmax(self.model(inputs).float(), dim=1)
            if torch.isfinite(probabilities).all():
                return probabilities.cpu()
        probabilities = F.softmax(self.model(inputs).float(), dim=1)
        if not torch.isfinite(probabilities).all():
            raise ValueError("모델 출력이 유한하지 않습니다")
        return probabilities.cpu()

    @staticmethod
    def _normalized_pixels(image, mean, std):
        pixels = torch.from_numpy(np.ascontiguousarray(image)).float().div_(255)
        pixels = pixels.unsqueeze(0) if pixels.ndim == 2 else pixels.permute(2, 0, 1)
        return (pixels - torch.tensor(mean).view(-1, 1, 1)) / torch.tensor(std).view(-1, 1, 1)

    def _patch_probabilities(self, image, tiling, mean, std, device):
        """패치 학습 모델 추론 → (C, h, w) 확률. h, w는 배율을 적용한 판정 영역 크기.

        분할 모델 셋(U-Net·DeepLabV3+·Custom CSP)은 전역 풀링이 없는 완전 합성곱이라, 패치로 학습해도
        판정 영역 전체를 한 번에 넣을 수 있다. 타일보다 빠르다(겹친 부분을 다시 계산하지 않음: 3500×1504를
        512 타일 36개로 나누면 9.4M 픽셀을 계산한다, 한 번에는 5.3M). 이음매도 없다.
        메모리가 모자라면(큰 이미지 · GPU) 타일로 바꿔 다시 한다.

            배율 s: 판정 영역을 s배로 줄인 뒤 추론 (학습도 같은 배율로 줄인 뒤 잘랐다) → 계산량 ∝ s²
        """
        from dataset import scale_image, scaled_size
        scale = float(tiling.get("scale", 1.0))
        height, width = image.shape[:2]
        if scale != 1:
            target_w, target_h = scaled_size(width, height, scale)
            # 학습 때와 같은 보간. 규칙에 보간이 없는 모델(9.33 첫 빌드)은 그때 학습한 INTER_LINEAR_EXACT.
            image = scale_image(image, (target_h, target_w), tiling.get("interpolation", "INTER_LINEAR_EXACT"))
        if getattr(self, "_patch_mode", "auto") != "tiles":
            try:
                probabilities = self._whole_probabilities(image, mean, std, device)
                self._patch_run, self._tile_count = "whole", 1
                return probabilities
            except (RuntimeError, MemoryError) as exc:
                if not _out_of_memory(exc):
                    raise
                if torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()
                self._patch_run = "tiles_after_oom"
        else:
            self._patch_run = "tiles"
        return self._tiled_probabilities(image, tiling, mean, std, device)

    def _whole_probabilities(self, image, mean, std, device, multiple=32):
        """판정 영역 전체를 한 번에. 백본의 최대 보폭(32)의 배수가 되게 오른쪽·아래를 0으로 채우고 잘라 낸다
        (학습 패치가 이미지 밖으로 나갈 때와 같은 채움)."""
        height, width = image.shape[:2]
        pixels = self._normalized_pixels(image, mean, std)
        padded_h, padded_w = -(-height // multiple) * multiple, -(-width // multiple) * multiple
        if (padded_h, padded_w) != (height, width):
            filled = torch.zeros(pixels.shape[0], padded_h, padded_w)
            filled[:, :height, :width] = pixels
            pixels = filled
        with torch.inference_mode():
            probabilities = self._forward_probabilities(pixels.unsqueeze(0), device)[0]
        return probabilities[:, :height, :width]

    def _tiled_probabilities(self, image, tiling, mean, std, device, batch=4):
        """원본 해상도에서 tile×tile 창을 overlap만큼 겹쳐 밀며 softmax 확률을 합친다 → (C, H, W) CPU 텐서.

            3500×1504, 타일 512, 겹침 128(보폭 384) → 가로 9 × 세로 4 = 36 타일

        합치기: 합성곱은 타일 가장자리에서 이웃 픽셀을 못 봐 예측이 약하다. 그래서 각 타일은 가장자리
        overlap/2 띠를 거의 믿지 않는다(가중치 1e-4). 이웃 타일의 안쪽이 정확히 그 띠에서 시작하므로
        모든 픽셀이 어떤 타일의 안쪽에서 나온다. 이미지 바깥 경계에 닿은 가장자리는 깎지 않는다.

            타일 A [=========|--]
            타일 B        [--|=========]     ('='가 믿는 안쪽, 두 안쪽이 이음매 없이 맞닿는다)

        이미지가 타일보다 작으면 0으로 채워 한 타일로 추론하고 원래 크기만 돌려준다.
        """
        from dataset import patch_origins
        tile, overlap = int(tiling["tile"]), int(tiling.get("overlap", int(tiling["tile"]) // 4))
        margin = overlap // 2
        height, width = image.shape[:2]
        pixels = self._normalized_pixels(image, mean, std)
        if height < tile or width < tile:
            padded = torch.zeros(pixels.shape[0], max(height, tile), max(width, tile))
            padded[:, :height, :width] = pixels
            pixels = padded
        full_h, full_w = pixels.shape[1:]
        origins = [(y, x) for y in patch_origins(full_h, tile, tile - overlap)
                   for x in patch_origins(full_w, tile, tile - overlap)]

        def edge_weights(start, length):
            """한 축의 타일 가중치: 안쪽 1, 이미지 안쪽으로 난 가장자리 띠 1e-4."""
            weights = torch.ones(tile)
            if margin and start > 0:
                weights[:margin] = 1e-4
            if margin and start + tile < length:
                weights[tile - margin:] = 1e-4
            return weights

        sums = weight_sum = None
        with torch.inference_mode():
            for start in range(0, len(origins), batch):
                group = origins[start:start + batch]
                inputs = torch.stack([pixels[:, y:y + tile, x:x + tile] for y, x in group])
                probabilities = self._forward_probabilities(inputs, device)
                if sums is None:
                    sums = torch.zeros(probabilities.shape[1], full_h, full_w)
                    weight_sum = torch.zeros(1, full_h, full_w)
                for (y, x), probability in zip(group, probabilities):
                    weight = edge_weights(y, full_h)[:, None] * edge_weights(x, full_w)[None, :]
                    sums[:, y:y + tile, x:x + tile] += probability * weight
                    weight_sum[:, y:y + tile, x:x + tile] += weight
        self._tile_count = len(origins)
        return (sums / weight_sum)[:, :height, :width]

    def _review_segmentation(self, prediction, image_path, mask, labels, probabilities, size, box, review,
                             num_classes=None):
        """예측 마스크를 결함 목록(클래스·박스·면적·점수)으로 바꾸고 OK/NG를 판정한다.

        review = {"min_area", "min_score", "split_dirs": [(이미지 폴더, 마스크 폴더), ...]}.
        이미지가 데이터셋에 있고 정답 마스크가 있으면 정답과 비교한다(IoU·놓침·오검출).
        """
        from core.segment_defects import analyze_segmentation, judge, load_truth, summary_text, truth_mask_path
        num_classes = num_classes or self.model.num_classes
        names = list(self.class_names) or [f"Class {i}" for i in range(num_classes)]
        truth, truth_path, note = None, truth_mask_path(image_path, review.get("split_dirs") or ()), ""
        if truth_path:
            try:
                truth = load_truth(truth_path, size, box)
                if truth is None:
                    note = "정답 마스크 크기·형식이 이미지와 달라 비교하지 않았습니다"
            except OSError as exc:
                note = f"정답 마스크를 읽지 못했습니다: {exc}"
        offset = (box[0], box[1]) if box else (0, 0)
        try:
            analysis = analyze_segmentation(mask, probabilities, size=labels.shape[::-1], labels=labels,
                                            class_names=names, offset=offset, truth=truth,
                                            num_classes=num_classes)
        except ValueError as exc:  # 정답 값 범위 오류는 비교만 빼고 결함은 보고한다
            note = f"정답 비교 생략: {exc}"
            analysis = analyze_segmentation(mask, probabilities, size=labels.shape[::-1], labels=labels,
                                            class_names=names, offset=offset, num_classes=num_classes)
        details = dict(prediction.details)
        details.update(analysis, defect_rules={"min_area": int(review.get("min_area", 1)),
                                               "min_score": float(review.get("min_score", 0.0))})
        if truth is not None and "ground_truth" in analysis:
            details["ground_truth"]["mask"] = truth_path
        if note:
            details["ground_truth_note"] = note
        verdict = judge(details)
        return replace(prediction, details=details, summary=summary_text(verdict),
                       color="#E05555" if verdict["decision"] == "NG" else "#34C759")

    def _run_sam2_inference(self, image_path: str):
        """SAM2 미세조정 모델: 격자 점 자동 분할 → 결함 마스크 (C++ SDK Automatic과 같은 점·합집합 규칙).

        SAM2는 클래스를 구분하지 않는다(학습 계약: 모든 전경 클래스 → 이진 객체 마스크).
        결과는 '배경 / 결함' 두 클래스이며, 결함 이름은 프로젝트에 결함 클래스가 하나면 그 이름이다.
        """
        from core.image_display import display_rgb
        from core.sam2_automatic import automatic_mask
        from core.spatial_preview import segmentation_labels, segmentation_preview
        from PIL import Image
        with self._measure_inference_stage("inference"):
            with Image.open(image_path) as source:
                original = display_rgb(np.array(source.convert("RGB")))
            mask, probabilities, info = automatic_mask(self._sam2_predictor, original,
                                                       getattr(self, "_sam2_settings", None))
        names = list(self.class_names or [])
        defect = names[1] if len(names) == 2 else "결함"
        self.class_names = ["background", defect]
        height, width = original.shape[:2]
        review = getattr(self, "segment_review", None)
        labels = segmentation_labels(mask, width, height)
        preview, counts = segmentation_preview(mask, original, 2, labels=labels, tint_background=review is None)
        prediction = InferenceResult(image_path, "ok", "segment", "세그멘테이션", "#A78BFA",
                                     details={"num_classes": 2, "pixel_counts": counts,
                                              "class_names": list(self.class_names),
                                              "sam2": {**info, "model_id": self._sam2_model_id},
                                              "runtime": "pytorch", "device": str(self._infer_device)})
        if review is not None:
            prediction = self._review_segmentation(prediction, image_path, mask, labels, probabilities,
                                                   (width, height), None, review, num_classes=2)
        self._heatmap_cache = None
        self._display_numpy_image(preview)
        self.info = (f"SAM2 자동 분할 | 격자 {info['grid']}×{info['grid']} · 점 {info['candidates']}개 중 "
                     f"{info['decoded']}개 디코딩 · 마스크 {info['accepted']}개 사용 "
                     f"(점수 ≥ {info['min_score']:.2f}, 면적 ≤ {info['max_area_percent']:g}%)")
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
    STATE_FIELDS = ("model", "_patchcore_model", "_upstream_model", "_upstream_task", "_gradcam",
                    "_infer_device", "_input_size", "_normalization", "class_names",
                    "_anomaly_threshold", "_active_checkpoint", "_center_crop", "_onnx_runtime",
                    "_sam2_predictor", "_sam2_model_id", "_tiling")

    def __init__(self, state, *, gradcam=True, input_region=None, runtime="auto", threads=4, segment_review=None,
                 sam2=None):
        for name in self.STATE_FIELDS:
            setattr(self, name, state.get(name))
        self.gradcam_enabled = gradcam
        # 분할 결함 분석(데스크톱에서 켬): None이면 기존처럼 픽셀 분포만 낸다.
        self.segment_review = segment_review
        # SAM2 자동 분할 설정 (격자·최소 점수·최대 면적). None이면 기본값.
        self._sam2_settings = sam2
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
                cached_matches = (self._onnx_runtime is not None and self._onnx_runtime.matches(
                    self.model, self._input_size, self.runtime_threads))
                if (cached_matches and self.runtime_requested == "auto"
                        and self._onnx_runtime.export_graph == "portable_fp64"):
                    raise ValueError("자동 가속에서는 속도가 느린 전체 FP64 ONNX 캐시를 사용하지 않습니다.")
                if not cached_matches:
                    self._onnx_runtime = EfficientNetOnnx(
                        self.model, self._input_size, threads=self.runtime_threads,
                        allow_precision_fallback=self.runtime_requested == "onnx")
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
                if self._onnx_runtime.export_graph == "portable_fp64":
                    self.runtime_warning = ("ONNX 수치 검증을 통과한 전체 FP64 계산을 사용합니다. "
                                            "추론이 느릴 수 있습니다. 자동 실행은 PyTorch를 사용합니다.")
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
            elif self._upstream_model is not None:
                result = self._run_upstream_inference(image_path)
            elif self._sam2_predictor is not None:
                result = self._run_sam2_inference(image_path)
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
