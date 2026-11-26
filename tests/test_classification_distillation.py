import json

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F

from builtin_models import build_builtin_model, make_builtin_checkpoint
from classification_distillation import ClassificationDistiller


@pytest.fixture
def teacher_checkpoint(tmp_path):
    torch.manual_seed(42)
    model = build_builtin_model("resnet18", 2, 1).eval()
    with torch.no_grad():
        model.fc.bias.copy_(torch.tensor([1.0, -1.0]))
    path = tmp_path / "teacher.pt"
    payload = make_builtin_checkpoint("resnet18", model, num_classes=2, input_size=(34, 45),
                                      in_channels=1, class_names=["ok", "ng"])
    payload["rng_state"] = {"numpy": np.random.get_state()}
    torch.save(payload, path)
    return path


def test_classification_distillation_aligns_classes_and_resizes_the_student_view(teacher_checkpoint):
    distill = ClassificationDistiller(teacher_checkpoint, ["ng", "ok"], 1, "cpu")
    images = torch.randn(2, 1, 31, 33)
    student = torch.tensor([[.25, -.25], [.5, -.5]], requires_grad=True)
    hard = F.cross_entropy(student, torch.tensor([0, 1]))
    with torch.no_grad():
        view = F.interpolate(images, size=(34, 45), mode="bilinear", align_corners=False, antialias=True)
        teacher = distill.teacher(view)[:, [1, 0]]
        soft = F.kl_div(F.log_softmax(student / 2, dim=1), F.softmax(teacher / 2, dim=1),
                        reduction="batchmean") * 4
        expected = .7 * hard + .3 * soft
    loss = distill(images, student, hard)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert student.grad is not None
    assert not distill.teacher.training and all(not p.requires_grad and p.grad is None for p in distill.teacher.parameters())
    with pytest.raises(ValueError, match="클래스 이름"):
        ClassificationDistiller(teacher_checkpoint, ["ng", "other"], 1, "cpu")
    with pytest.raises(ValueError, match="입력 채널"):
        ClassificationDistiller(teacher_checkpoint, ["ng", "ok"], 3, "cpu")


def test_classification_distillation_uses_teacher_in_training_only_and_exports_student(tmp_path, teacher_checkpoint):
    from export_onnx import export_checkpoint
    from train_builtin import train_builtin

    data = tmp_path / "data"
    for split in ("train", "val"):
        for name, gray in (("ng", 45), ("ok", 180)):
            folder = data / split / name
            folder.mkdir(parents=True)
            Image.fromarray(np.full((32, 32), gray, dtype=np.uint8)).save(folder / "sample.png")

    student = train_builtin(
        "resnet18", data, in_channels=1, input_size=32, batch_size=2, epochs=1,
        teacher_checkpoint=teacher_checkpoint, output_dir=tmp_path / "student",
        horizontal_flip=0, rotation=0, color_jitter=0, num_workers=0,
    )
    payload = torch.load(student, weights_only=True)
    assert payload["training_config"]["distillation"]["teacher_input_size"] == [34, 45]
    teacher_checkpoint.unlink()
    model = tmp_path / "student-int8.onnx"
    result = export_checkpoint(student, model, precision="int8", calibration_dir=data / "val")
    cfg = json.loads(model.with_suffix(".json").read_text())
    assert result["model_version"] == cfg["model_version"]
    assert result["model_exported_at"] == cfg["model_exported_at"]
    assert result["verification"] == "int8_compared" and cfg["class_names"] == ["ng", "ok"]
    assert cfg["export"]["quantization"]["comparison"]["windows"] == 2
