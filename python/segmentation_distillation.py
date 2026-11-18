"""Online pixel distillation: teacher and student see the identical augmented crop."""

from pathlib import Path

import torch
from torch.nn import functional as F


class SegmentationDistiller:
    def __init__(
        self, checkpoint, class_names, channels, device, *, alpha=0.3, temperature=2.0
    ):
        if not 0 < alpha < 1 or not 0.1 <= temperature <= 20:
            raise ValueError("증류 비중은 0~1 사이, 온도는 0.1~20이어야 합니다")
        from builtin_assets import file_sha256
        from export_onnx import load_custom_model, resolve_checkpoint_spec

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        spec = resolve_checkpoint_spec(payload)
        payload.pop("optimizer_state_dict", None)
        teacher_names = spec["class_names"]
        self.focus_class = None
        if (len(teacher_names) == 2 and len(class_names) > 2
                and teacher_names[0] == class_names[0] and teacher_names[1] in class_names[1:]):
            self.focus_class = class_names.index(teacher_names[1])
        if (
            spec["task"] != "segment"
            or (teacher_names != list(class_names) and self.focus_class is None)
            or spec["in_channels"] != channels
        ):
            raise ValueError(
                "교사와 학생은 같은 분할 태스크·입력 채널이어야 합니다. 클래스 순서를 맞추거나 배경과 학생 결함 하나로 학습한 교사를 선택하세요."
            )
        mean, std = (
            ([0.449], [0.226])
            if channels == 1
            else ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
        prep = spec["preprocessing"]
        if (
            prep.get("mean", prep.get("normalize_mean")) != mean
            or prep.get("std", prep.get("normalize_std")) != std
        ):
            raise ValueError("교사의 정규화가 학생과 다릅니다")
        self.teacher = (
            load_custom_model(payload, spec).to(device).eval().requires_grad_(False)
        )
        self.alpha, self.temperature = float(alpha), float(temperature)
        self.config = {
            "teacher_file": Path(checkpoint).name,
            "teacher_sha256": file_sha256(Path(checkpoint)),
            "alpha": self.alpha,
            "temperature": self.temperature,
            "views": "same_augmented_input",
            "loss": "pixel_kl_ignore255",
        }

        if self.focus_class is not None:
            self.config["focus_class"] = teacher_names[1]
            self.config["loss"] = "binary_pixel_kl_ignore255"

    def __call__(self, images, logits, masks, supervised_loss):
        with torch.no_grad():
            teacher = self.teacher(images)
        logits = logits.float()
        teacher = teacher.float()
        valid = masks != 255
        if self.focus_class is not None:
            idx = self.focus_class
            other = torch.cat((logits[:, :idx], logits[:, idx + 1:]), dim=1)
            logits = torch.stack((other.logsumexp(dim=1), logits[:, idx]), dim=1)
            # A specialist never supervises a different labelled defect. Human
            # masks win where the specialist contradicts the annotation.
            valid = ((masks == 0) | (masks == idx)) & (teacher.argmax(1) == (masks == idx).long())
        if teacher.shape != logits.shape:
            raise ValueError("교사와 학생의 분할 출력 크기가 다릅니다")
        t = self.temperature
        # Divide and reduce in FP32, even when the model forwards use AMP.
        kl = F.kl_div(
            F.log_softmax(logits / t, dim=1),
            F.softmax(teacher / t, dim=1),
            reduction="none",
        ).sum(1)
        if not valid.any():
            return supervised_loss
        kd = kl[valid].mean() * t * t
        return (1 - self.alpha) * supervised_loss + self.alpha * kd
