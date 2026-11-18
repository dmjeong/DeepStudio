"""Regression coverage for desktop teacher training, specialization and deployment."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
from PIL import Image

from builtin_models import build_builtin_model, make_builtin_checkpoint
from segmentation_distillation import SegmentationDistiller


TEACHERS = ("mask2former_swin_small", "segformer_b2", "dinov2_vitb14_seg")


def test_specialist_supervises_only_its_own_defect(tmp_path):
    teacher = build_builtin_model("unet_lite", 2).eval()
    payload = make_builtin_checkpoint("unet_lite", teacher, num_classes=2,
                                      class_names=["background", "tray"])
    path = tmp_path / "teacher.pt"
    torch.save(payload, path)
    distill = SegmentationDistiller(path, ["background", "crack", "tray"], 3, "cpu")
    assert distill.config["focus_class"] == "tray"

    class FixedTeacher(torch.nn.Module):
        def forward(self, images):
            return images[:, :2]

    distill.teacher = FixedTeacher()
    sample = torch.tensor([[[[0., 0., 0., 0.]], [[3., 3., 3., 3.]], [[0., 0., 0., 0.]]]])
    logits = torch.zeros_like(sample, requires_grad=True)
    loss = distill(sample, logits, torch.tensor([[[2, 1, 255, 0]]]), logits.sum() * 0)
    loss.backward()
    assert logits.grad[..., 0].abs().sum() > 0
    assert logits.grad[..., 1:].count_nonzero() == 0
    with pytest.raises(ValueError, match="클래스"):
        SegmentationDistiller(path, ["background", "crack", "flip"], 3, "cpu")


def test_desktop_teacher_initialization_modes(tmp_path):
    from PySide6.QtWidgets import QApplication
    from core.project import ProjectManager
    from widgets.training_widget import TrainingWidget

    app = QApplication.instance() or QApplication([])
    project = ProjectManager.create_new("teacher", "segment", str(tmp_path), ["background", "chip"])
    widget = TrainingWidget()
    try:
        widget.set_project(project)
        for teacher in TEACHERS:
            index = widget.model_id_combo.findData(teacher)
            assert index >= 0
            action = next(a for a in widget.teacher_training_button.menu().actions() if a.data() == teacher)
            widget.teacher_edit.setText("previous-teacher.pt")
            action.trigger()
            assert widget.model_id_combo.currentData() == teacher and not widget.teacher_edit.text()
            assert not widget.teacher_training_button.isHidden()
            available = {widget.mode_combo.itemData(i) for i in range(widget.mode_combo.count())}
            assert {"builtin_scratch", "builtin_transfer", "builtin_resume"} <= available
            assert ("builtin_finetune" in available) == (teacher != "segformer_b2")
            assert not widget.distillation_group.isHidden()
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("model_id", TEACHERS)
@pytest.mark.skipif(os.environ.get("DVS_RUN_TEACHER_INTEGRATION") != "1",
                    reason="Real transformer training and ORT export write large temporary checkpoints")
def test_train_teacher_distill_student_and_run_onnx(model_id):
    import onnxruntime as ort
    from train_builtin import train_builtin
    from export_onnx import export_checkpoint
    from quantize_dense import quantize_dense
    from builtin_models import load_builtin_checkpoint

    with TemporaryDirectory() as directory:
        root = Path(directory)
        data = root / "data"
        for split, count in (("train", 4), ("val", 2)):
            for i in range(count):
                mask = np.zeros((71, 83), np.uint8)
                mask[8:47, 12 + i * 3:18 + i * 3] = 1
                rgb = np.stack([mask * 150 + 35, mask * 45 + 90, mask * 75 + 40], axis=-1)
                for folder, array in (("images", rgb), ("masks", mask)):
                    dest = data / folder / split
                    dest.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(array).save(dest / f"{i}.png")
        args = dict(data_root=data, num_classes=2, class_names=["background", "tray"],
                    input_size=64, in_channels=3, epochs=1, batch_size=2, num_workers=0,
                    optimizer_name="sgd", scheduler_name="none", log=print)
        pretrained = model_id != "segformer_b2" and os.getenv("DVS_TEST_TEACHER_ASSETS") == "1"
        teacher = train_builtin(model_id, output_dir=root / "teacher", pretrained=pretrained, **args)
        payload = torch.load(teacher, map_location="cpu", weights_only=True)
        if pretrained:
            assert payload["weight_provenance"]["source"] not in ("random", "random_init")
        model = load_builtin_checkpoint(payload)
        with torch.no_grad():
            out = model(torch.randn(1, 3, 65, 79))
            assert out.shape == (1, 2, 65, 79) and out.isfinite().all()
        del model, payload
        student = train_builtin("unet_resnet18", teacher_checkpoint=teacher, output_dir=root / "student", **args)
        payload = torch.load(student, weights_only=True)
        assert payload["training_config"]["distillation"]["teacher_sha256"]
        reference = load_builtin_checkpoint(payload)
        sample = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            expected = reference(sample).numpy()
        onnx = root / "student.onnx"
        export_checkpoint(student, onnx, verify=True)
        session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
        actual = session.run(None, {session.get_inputs()[0].name: sample.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=.001, atol=.001)
        quantized = quantize_dense(onnx.with_suffix(".json"), data / "images/train", count=4)
        config = json.loads(quantized.read_text(encoding="utf-8"))
        assert config["verification"] == "int8_compared"
        int8 = ort.InferenceSession(str(quantized.parent / config["model_path"]),
                                    providers=["CPUExecutionProvider"])
        prediction = int8.run(None, {int8.get_inputs()[0].name: sample.numpy()})[0]
        assert prediction.shape == expected.shape and np.isfinite(prediction).all()


@pytest.mark.parametrize("model_id", TEACHERS[:2])
@pytest.mark.skipif(os.environ.get("DVS_RUN_TEACHER_INTEGRATION") != "1",
                    reason="Creates a full-size local pretrained model file")
def test_local_hf_weights_keep_encoder_and_reset_foreign_labels(model_id):
    from safetensors.torch import save_file
    from transformer_segmenters import TASK_HEADS

    with TemporaryDirectory() as directory:
        source = build_builtin_model(model_id, 4)
        tensors = source.net.state_dict()
        key = next(k for k in tensors if k.endswith("projection.weight") or k.endswith("proj.weight"))
        sentinel = tensors[key].flatten()[0].item()
        file = Path(directory) / "model.safetensors"
        save_file(tensors, str(file))
        del source, tensors
        target = build_builtin_model(model_id, 2, backbone_weights=file)
        assert target.weight_provenance["source"] == "local_hf_segmentation"
        assert target.net.state_dict()[key].flatten()[0].item() == sentinel
        assert target.net.state_dict()[TASK_HEADS[model_id][0] + "weight"].shape[0] == (
            3 if model_id == "mask2former_swin_small" else 2)
        with pytest.raises(ValueError, match="가중치"):
            target.initialize_from_weights({"wrong": torch.zeros(1)}, file)
