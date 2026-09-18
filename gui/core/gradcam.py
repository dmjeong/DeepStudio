"""
Deep Vision Studio — Grad-CAM (Gradient-weighted Class Activation Mapping)

모델이 "어디를 보고"판단했는지 시각화하는 해석 도구

원리 및 흐름:
┌─────────────────────────────────────────────────────────────────────┐
│                         Grad-CAM Pipeline                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Forward Hook 등록                                               │
│     model.backbone.stage4 (P5 출력) 에 훅을 걸어 activation 캡처     │
│                                                                     │
│  2. Forward Pass (그래디언트 활성화 상태)                              │
│     Input → Backbone → Head → Output (예측 결과)                     │
│                                                                     │
│  3. Target Score 선정                                                │
│     ┌─────────────┬───────────────────────────────────────────────┐ │
│     │ classify    │ 예측 클래스의 logit                            │ │
│     │ segment     │ 예측 마스크의 평균 logit                       │ │
│     │ detect      │ 최고 objectness score                         │ │
│     │ anomaly     │ 재구성 오차(MSE)의 평균                        │ │
│     └─────────────┴───────────────────────────────────────────────┘ │
│                                                                     │
│  4. Backward Pass                                                   │
│     target_score.backward() → stage4 activation의 그래디언트 계산    │
│                                                                     │
│  5. CAM 생성                                                        │
│     weights = GAP(gradients)      ← 각 채널의 중요도                │
│     cam = Σ (weights × activations)  ← 가중 합                      │
│     cam = ReLU(cam)               ← 양의 기여만                     │
│     cam = Normalize(cam)          ← [0, 1] 범위                     │
│                                                                     │
│  6. 히트맵 오버레이                                                  │
│     cam → 원본 이미지 크기로 리사이즈 → JET 컬러맵 적용               │
│     → 원본 이미지와 알파 블렌딩 (α=0.5)                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

태스크별 타겟 레이어:
┌──────────────┬──────────────────────────────────────────────────────┐
│ 태스크        │ 타겟 레이어 & 해석                                   │
├──────────────┼──────────────────────────────────────────────────────┤
│ classify     │ backbone.stage4 (P5) — 최고 수준 시맨틱 특징          │
│ segment      │ backbone.stage4 (P5) — FPN 입력 중 최상위             │
│ detect       │ backbone.stage4 (P5) — FCOS 입력 최상위 스케일        │
│ anomaly      │ backbone.stage4 (P5) — 재구성 전 병목 특징            │
└──────────────┴──────────────────────────────────────────────────────┘
"""

import numpy as np
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Optional, Tuple

from core.heatmap import normalize_activation_map, render_heatmap


class GradCAM:
    """
    Grad-CAM: 모델 해석 시각화 도구

    Usage:
        gradcam = GradCAM(model)
        heatmap, overlay = gradcam.generate(input_tensor, original_image)
        gradcam.release()  # 훅 해제 (메모리 누수 방지)
    """

    def __init__(self, model: torch.nn.Module, target_layer: Optional[str] = None,
                 score_layer: Optional[str] = None):
        """
        Args:
            model: CustomCSP 모델 인스턴스 (eval 모드 권장)
            target_layer: 타겟 레이어 이름 (기본: "backbone.stage4")
                          - 백본의 마지막 스테이지가 가장 풍부한 시맨틱 정보 제공
        """
        self.model = model
        self.task = getattr(model, "task", "classify")
        self.target_layer = target_layer or getattr(model, "gradcam_target_layer", "backbone.stage4")
        self._score_module = self._find_layer(score_layer) if score_layer else None
        if score_layer and self._score_module is None:
            raise ValueError(f"분류 점수 레이어를 찾을 수 없습니다: {score_layer}")
        self._score_hook = None
        self._logits = None
        self.diagnostics = {}

        # ── 타겟 레이어 자동 탐색 ──
        if target_layer is None:
            target_layer = self.target_layer

        self._target_module = self._find_layer(target_layer)
        if self._target_module is None:
            raise ValueError(
                f"타겟 레이어 '{target_layer}'를 찾을 수 없습니다.\n"
                f"사용 가능: {[n for n, _ in model.named_modules()]}"
            )

        # ── 훅 저장소 ──
        self._activations: Optional[torch.Tensor] = None  # forward 출력
        self._gradients: Optional[torch.Tensor] = None     # backward 그래디언트

        # 일반 추론에는 훅을 설치하지 않는다. CAM 계산 중에만 캡처한다.
        self._fwd_hook = None
        self._tensor_hook = None


    def _save_logits(self, module, inputs, output):
        self._logits = output

    def _find_layer(self, layer_name: str) -> Optional[torch.nn.Module]:
        """
        점(.) 으로 구분된 레이어 이름으로 모듈 탐색
        예: "backbone.stage4"→ model.backbone.stage4
        """
        module = self.model
        for part in layer_name.split("."):
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    # ── Hook 콜백 ──────────────────────────────────────

    def _save_activation(self, module, input, output):
        """Forward hook: 타겟 레이어의 출력(activation) 저장"""
        if isinstance(output, (list, tuple)):
            output = output[-1]
        if not torch.is_tensor(output) or output.ndim != 4:
            raise ValueError("Grad-CAM 타겟 레이어는 (B, C, H, W) 특징맵이어야 합니다")
        if not output.requires_grad:
            raise RuntimeError("Grad-CAM 타겟 특징맵에 미분 경로가 없습니다")
        # 뒤쪽 inplace 연산으로 값이 바뀌어도 캡처 당시 특징을 보존한다.
        self._activations = output.detach().clone()
        if self._tensor_hook is not None:
            self._tensor_hook.remove()
        # 모듈 backward 훅의 view는 inplace 활성화 함수와 충돌하므로
        # 실제 출력 텐서에서 해당 시점의 그래디언트를 직접 받는다.
        self._tensor_hook = output.register_hook(self._save_gradient)

    def _save_gradient(self, gradient):
        """타겟 특징맵의 그래디언트를 모델 파라미터와 분리해 저장한다."""
        self._gradients = gradient.detach()

    @contextmanager
    def _differentiable_model_tensors(self):
        """inference 모드에서 fuse된 가중치도 입력 미분에 사용할 수 있게 한다.

        inference 모드에서 준비한 모델은 inference 텐서를 포함할 수 있다.
        입력만 복사해서는 Conv 역전파가 이 가중치를 저장하지 못한다.
        해당 파라미터와 버퍼만 잠시 일반 텐서로 바꾸고 원래 참조를 복원한다.
        """
        originals = []
        clones = {}
        try:
            for module in self.model.modules():
                members = tuple(module._parameters.items()) + tuple(module._buffers.items())
                for name, tensor in members:
                    if tensor is None or not torch.is_inference(tensor):
                        continue
                    if id(tensor) not in clones:
                        clone = tensor.detach().clone()
                        if isinstance(tensor, torch.nn.Parameter):
                            clone = torch.nn.Parameter(clone, requires_grad=tensor.requires_grad)
                        clones[id(tensor)] = clone
                    originals.append((module, name, tensor))
                    setattr(module, name, clones[id(tensor)])
            yield
        finally:
            for module, name, tensor in reversed(originals):
                setattr(module, name, tensor)

    # ── Grad-CAM 생성 ─────────────────────────────────

    def generate(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        alpha: float = 0.5,
        target_tensor: Optional[torch.Tensor] = None,
        *,
        lower: float = 0.0,
        upper: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Grad-CAM 히트맵 + 오버레이 생성

        흐름:
        ┌────────┐   forward    ┌────────┐   backward   ┌──────────┐
        │ Input  │ ──────────► │ Output │ ───────────► │Gradients │
        └────────┘              └────────┘              └──────────┘
                                                              │
        ┌──────────────┐   weighted sum   ┌──────────────────▼──────┐
        │   Heatmap    │ ◄────────────── │ Activations × Weights   │
        └──────┬───────┘                  └────────────────────────┘
               │  overlay
        ┌──────▼───────┐
        │ Original +   │
        │ JET colormap │
        └──────────────┘

        Args:
            input_tensor: 전처리된 입력 (1, C, H, W) — 디바이스 일치 필요
            original_image: 원본 이미지 numpy (H, W, 3) RGB, uint8
            target_class: 분류 시 타겟 클래스 (None이면 예측 클래스 사용)
            alpha: 오버레이 투명도 (0=원본, 1=히트맵)
            lower: 표시할 최소 활성화 강도 (0~1)
            upper: 가장 강한 색상의 시작 강도 (lower 초과, 1 이하)

        Returns:
            heatmap: (H, W, 3) JET 컬러맵 히트맵 uint8
            overlay: (H, W, 3) 원본 + 히트맵 블렌딩 uint8
        """
        cam = self.generate_activation(input_tensor, target_class, target_tensor)
        return render_heatmap(cam, original_image, alpha, lower, upper)

    def generate_activation(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_tensor: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """모델을 한 번 실행해 0~1 범위의 2차원 float32 지도를 반환한다.

        반환된 지도는 원본 크기 변경이나 표시 범위 설정과 독립적이다.
        지도만 캐시한 뒤 render_heatmap으로 표시 설정을 즉시 반영한다.
        """
        if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
            raise ValueError("Grad-CAM 입력은 (1, C, H, W) 텐서여야 합니다")
        training_states = [(module, module.training) for module in self.model.modules()]
        self._activations = self._gradients = None
        self._logits = None
        self.diagnostics = {}
        try:
            self.model.eval()
            self._fwd_hook = self._target_module.register_forward_hook(self._save_activation)
            if self._score_module is not None:
                self._score_hook = self._score_module.register_forward_hook(self._save_logits)
            with (torch.inference_mode(False), torch.enable_grad(),
                  self._differentiable_model_tensors()):
                parameter = next(self.model.parameters(), None)
                input_tensor = input_tensor.detach().clone()
                if parameter is not None:
                    input_tensor = input_tensor.to(device=parameter.device, dtype=parameter.dtype)
                input_tensor.requires_grad_(True)
                output = self.model(input_tensor)
                if self._score_module is not None:
                    if self._logits is None:
                        raise RuntimeError("분류 Linear 점수가 캡처되지 않았습니다")
                    output = self._logits
                target_score = self._compute_target_score(output, target_class, target_tensor)
                if target_score.numel() != 1 or not torch.isfinite(target_score).all():
                    raise ValueError("Grad-CAM 타겟 점수는 유한한 스칼라여야 합니다")
                # 기존 학습 그래디언트를 지우거나 파라미터에 새 값을 쌓지 않는다.
                # 입력까지 미분하면 고정된 백본에서도 텐서 훅이 호출된다.
                torch.autograd.grad(target_score, input_tensor)
                if self._gradients is None or self._activations is None:
                    raise RuntimeError("Grad-CAM 훅이 정상 작동하지 않습니다")
                cam = self._compute_cam()
                self.diagnostics = {
                    "layer": self.target_layer, "feature_shape": tuple(self._activations.shape[-2:]),
                    "input_shape": tuple(input_tensor.shape[-2:]),
                    "positive_fraction": float(np.mean(cam > 0)),
                    "gradient_max": float(self._gradients.detach().float().abs().max().cpu()),
                }
            return cam
        finally:
            for module, was_training in training_states:
                module.training = was_training
            self.release()

    def _compute_target_score(
        self, output, target_class: Optional[int], target_tensor=None
    ) -> torch.Tensor:
        """
        태스크별 Grad-CAM 타겟 스코어 결정

        ┌──────────┬──────────────────────────────────────────────┐
        │ classify │ softmax 이전 타겟 클래스 logit                 │
        │ segment  │ 예측 마스크에서 가장 많은 영역의 클래스 logit  │
        │ detect   │ 최고 objectness 셀의 confidence              │
        │ anomaly  │ 재구성 오차 평균 (높을수록 이상)               │
        └──────────┴──────────────────────────────────────────────┘

        분류 타겟은 softmax 적용 전 점수입니다.
        """
        if not torch.is_tensor(output):
            raise ValueError("Grad-CAM 모델 출력은 텐서여야 합니다")

        if self.task == "classify":
            # 분류: 예측 또는 지정 클래스의 logit
            if output.ndim != 2 or output.shape[0] != 1:
                raise ValueError("분류 Grad-CAM 출력은 (1, 클래스 수)여야 합니다")
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            if not 0 <= target_class < output.shape[1]:
                raise IndexError("Grad-CAM 클래스 인덱스 범위 오류")
            return output[0, target_class]

        elif self.task == "segment":
            # 분할: 가장 넓은 영역을 차지하는 클래스의 평균 logit
            pred_mask = output.argmax(dim=1)  # (B, H, W)
            if target_class is None:
                # 배경(0)을 제외한 가장 넓은 전경 클래스
                unique, counts = torch.unique(pred_mask, return_counts=True)
                if len(unique) > 1:
                    # 배경(0)이 아닌 것 중 최대
                    fg_mask = unique > 0
                    if fg_mask.any():
                        target_class = unique[fg_mask][
                            counts[fg_mask].argmax()
                        ].item()
                    else:
                        target_class = unique[counts.argmax()].item()
                else:
                    target_class = unique[0].item()
            # 해당 클래스의 logit 채널 평균
            return output[0, target_class].mean()

        elif self.task == "detect":
            # 탐지: 최고 objectness 셀의 confidence score
            obj_scores = torch.sigmoid(output[:, :, 4])  # (B, N)
            max_idx = obj_scores[0].argmax()
            return output[0, max_idx, 4]

        elif self.task == "anomaly":
            if target_tensor is None or target_tensor.shape != output.shape:
                raise ValueError("이상 탐지 Grad-CAM에는 출력과 같은 크기의 원본 [0,1] 텐서가 필요합니다")
            target = target_tensor.detach().to(output.device)
            return (output - target).pow(2).mean()

        else:
            raise ValueError(f"지원하지 않는 태스크: {self.task}")

    def _compute_cam(self) -> np.ndarray:
        """
        Grad-CAM 계산 핵심 로직

        수식:
            weights_k = GAP(∂y/∂A^k)     — 채널별 중요도
            CAM = ReLU(Σ_k weights_k × A^k)  — 가중 합
        """
        gradients = self._gradients.float()  # (1, C, H, W)
        activations = self._activations.float()  # (1, C, H, W)
        if gradients.shape != activations.shape:
            raise ValueError("Grad-CAM 특징맵과 그래디언트 크기 불일치")

        # GAP로 채널별 가중치 계산: (1, C, 1, 1)
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # 가중합 계산: (1, 1, H, W)
        cam = (weights * activations).sum(dim=1, keepdim=True)

        # ReLU: 양의 기여만 (음의 기여는 해당 클래스에 반하는 방향)
        cam = F.relu(cam)

        # 정규화: [0, 1] 범위
        cam = cam[0, 0].detach().float().cpu().numpy()  # 크기가 1인 축도 유지
        if np.isfinite(cam).all() and cam.min() == cam.max() and cam.max() > 0:
            # 공간 구분이 없더라도 균일한 양의 기여 자체를 지우지 않는다.
            return np.ones(cam.shape, dtype=np.float32)
        return normalize_activation_map(cam)

    @staticmethod
    def _apply_colormap(cam: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        CAM → JET 컬러맵 변환 (Matplotlib 없이 순수 numpy)

        JET 컬러맵 근사:
        ┌──────────────────────────────────────────────────────┐
        │ 값  : 0.0 ─── 0.25 ─── 0.5 ─── 0.75 ─── 1.0       │
        │ 색  : 파랑 ── 시안 ── 초록 ── 노랑 ── 빨강            │
        │       (관심↓)                        (관심↑)         │
        └──────────────────────────────────────────────────────┘

        Args:
            cam: (H, W) normalized [0, 1]
            size: (width, height) 출력 크기

        Returns:
            heatmap: (height, width, 3) uint8 RGB
        """
        # 리사이즈 (PyTorch bilinear interpolation)
        cam_resized = GradCAM._resize_cam(cam, size)

        # JET 컬러맵 적용 (순수 numpy, OpenCV 불필요)
        heatmap = GradCAM._jet_colormap(cam_resized)

        return heatmap

    @staticmethod
    def _resize_cam(cam: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        CAM을 지정 크기로 리사이즈 (PyTorch interpolate 사용)

        Args:
            cam: (H, W) float array
            size: (width, height)
        Returns:
            resized: (height, width) float array
        """
        w, h = size
        cam_tensor = torch.from_numpy(cam).float().unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(
            cam_tensor, size=(h, w), mode="bilinear", align_corners=False
        )
        return resized.squeeze().numpy()

    @staticmethod
    def _jet_colormap(values: np.ndarray) -> np.ndarray:
        """
        JET 컬러맵 순수 numpy 구현 (OpenCV 의존 제거)

        수학적 정의:
            R = clip(1.5 - |4v - 3|, 0, 1)
            G = clip(1.5 - |4v - 2|, 0, 1)
            B = clip(1.5 - |4v - 1|, 0, 1)

        Args:
            values: (H, W) normalized [0, 1]

        Returns:
            rgb: (H, W, 3) uint8
        """
        v = np.clip(values, 0.0, 1.0)

        # JET 컬러맵 수식
        r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)

        rgb = np.stack([r, g, b], axis=-1)
        return (rgb * 255).astype(np.uint8)

    @staticmethod
    def _overlay(
        original: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        원본 이미지 + 히트맵 알파 블렌딩

        결과 = α × heatmap + (1 - α) × original

        Args:
            original: (H, W, 3) uint8 RGB
            heatmap: (H, W, 3) uint8 RGB JET
            alpha: 히트맵 가중치 (0.0 ~ 1.0)

        Returns:
            blended: (H, W, 3) uint8
        """
        # 크기 불일치 시 히트맵 기준으로 원본 리사이즈
        if original.shape[:2] != heatmap.shape[:2]:
            h, w = heatmap.shape[:2]
            orig_tensor = (
                torch.from_numpy(original)
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            orig_resized = F.interpolate(
                orig_tensor, size=(h, w), mode="bilinear", align_corners=False
            )
            original = (
                orig_resized.squeeze(0)
                .permute(1, 2, 0)
                .clamp(0, 255)
                .byte()
                .numpy()
            )

        # 알파 블렌딩
        blended = (
            alpha * heatmap.astype(np.float32)
            + (1 - alpha) * original.astype(np.float32)
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    # ── 리소스 해제 ──────────────────────────────────

    def release(self):
        """
        훅 해제 — 사용 후 반드시 호출 (메모리 누수 방지)

        GC가 처리하지 못하는 순환 참조를 끊음:
        model → hook → GradCAM → model
        """
        if hasattr(self, "_fwd_hook") and self._fwd_hook is not None:
            self._fwd_hook.remove()
            self._fwd_hook = None
        if hasattr(self, "_tensor_hook") and self._tensor_hook is not None:
            self._tensor_hook.remove()
            self._tensor_hook = None
        if getattr(self, "_score_hook", None) is not None:
            self._score_hook.remove()
            self._score_hook = None
        self._logits = None
        self._activations = None
        self._gradients = None

    def __del__(self):
        """소멸자: 훅 자동 해제"""
        self.release()
