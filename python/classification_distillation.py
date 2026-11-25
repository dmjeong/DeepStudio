"""Online soft-target distillation for native classification models."""

from pathlib import Path
import pickle

import torch
from torch.nn import functional as F


class ClassificationDistiller:
    def __init__(self, checkpoint, class_names, channels, device, *, alpha=0.3, temperature=2.0):
        if not 0 < alpha < 1 or not 0.1 <= temperature <= 20:
            raise ValueError("증류 비중은 0~1 사이, 온도는 0.1~20이어야 합니다.")
        from builtin_assets import file_sha256
        from dataset import get_normalize_params
        from export_onnx import checkpoint_backend, load_custom_model, resolve_checkpoint_spec

        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"분류 교사 체크포인트를 찾을 수 없습니다: {path}")
        # Older EfficientNet best files include NumPy's RNG state alongside the model.
        import numpy as np
        reconstruct = np.empty(0, dtype=np.uint32).__reduce__()[0]
        safe_types = [
            (reconstruct, "numpy.core.multiarray._reconstruct"),
            (reconstruct, "numpy._core.multiarray._reconstruct"),
            np.ndarray, np.dtype, type(np.dtype(np.uint32)),
        ]
        try:
            with torch.serialization.safe_globals(safe_types):
                payload = torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as exc:
            raise ValueError("분류 교사는 Deep Vision Studio에서 저장한 분류 .pt 체크포인트를 선택하세요.") from exc
        if not isinstance(payload, dict) or checkpoint_backend(payload) not in {"builtin", "efficientnet", "custom"}:
            raise ValueError("분류 교사는 기본 분류 모델, EfficientNet 또는 Custom CSP의 체크포인트여야 합니다.")
        spec = resolve_checkpoint_spec(payload)
        if spec["task"] != "classify":
            raise ValueError("분류 증류에는 분류 모델로 학습한 교사 .pt가 필요합니다.")
        teacher_names = spec["class_names"]
        if (len(set(class_names)) != len(class_names) or len(teacher_names) != len(class_names)
                or set(teacher_names) != set(class_names)):
            raise ValueError("교사와 학생에 같은 분류 클래스 이름이 필요합니다.")
        mean, std = get_normalize_params(channels)
        prep = spec["preprocessing"]
        if spec["in_channels"] != channels or prep["normalize_mean"] != mean or prep["normalize_std"] != std:
            raise ValueError("교사와 학생의 입력 채널과 정규화가 같아야 합니다. 흑백/컬러 선택을 확인하세요.")

        # Strip training state before moving the frozen model to the chosen device.
        for key in ("optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "best_state_dict", "rng_state"):
            payload.pop(key, None)
        self.teacher = load_custom_model(payload, spec).to(device).eval().requires_grad_(False)
        self.teacher_size = (spec["input_height"], spec["input_width"])
        order = [teacher_names.index(name) for name in class_names]
        self.class_order = torch.tensor(order, dtype=torch.long, device=device)
        self.alpha, self.temperature = float(alpha), float(temperature)
        self.config = {
            "teacher_file": path.name,
            "teacher_sha256": file_sha256(path),
            "teacher_input_size": list(self.teacher_size),
            "alpha": self.alpha,
            "temperature": self.temperature,
            "views": "same_augmented_view",
            "loss": "classification_kl",
        }

    def __call__(self, images, logits, supervised_loss):
        with torch.no_grad():
            if images.shape[-2:] != self.teacher_size:
                images = F.interpolate(images, size=self.teacher_size, mode="bilinear",
                                       align_corners=False, antialias=True)
            teacher_logits = self.teacher(images)
            if teacher_logits.shape != logits.shape:
                raise ValueError("분류 교사와 학생의 출력 크기가 다릅니다.")
            teacher_logits = teacher_logits.index_select(1, self.class_order)
        temperature = self.temperature
        soft_loss = F.kl_div(
            F.log_softmax(logits.float() / temperature, dim=1),
            F.softmax(teacher_logits.float() / temperature, dim=1),
            reduction="batchmean",
        ) * temperature ** 2
        return (1 - self.alpha) * supervised_loss + self.alpha * soft_loss
