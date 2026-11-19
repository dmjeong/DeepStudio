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


def test_legacy_query_shape_and_transfer_preserve_matching_slots():
    from transformer_segmenters import checkpoint_query_count, transfer_query_weights

    feature = "net.model.transformer_module.queries_features.weight"
    position = "net.model.transformer_module.queries_embedder.weight"
    rows = torch.arange(800).reshape(100, 8)
    state = {feature: rows, position: -rows, "net.other": torch.tensor(42)}
    assert checkpoint_query_count({"model_state_dict": state}) == 100
    reduced = transfer_query_weights(state, 16, prefix="net.")
    assert torch.equal(reduced[feature], rows[:16]) and torch.equal(reduced[position], -rows[:16])
    assert reduced["net.other"] is state["net.other"] and state[feature].shape[0] == 100
    assert checkpoint_query_count({"model_config": {"num_queries": 16}, "model_state_dict": reduced}) == 16
    with pytest.raises(ValueError, match="설정과 가중치"):
        checkpoint_query_count({"model_config": {"num_queries": 100}, "model_state_dict": reduced})
    with pytest.raises(ValueError, match="후보 텐서"):
        checkpoint_query_count({"model_state_dict": {feature: rows[:16], position: -rows}})
    with pytest.raises(ValueError, match="16 이하"):
        transfer_query_weights(reduced, 100, prefix="net.")


def test_mask2former_matching_samples_preserve_rare_classes_and_ignore_padding():
    from transformer_segmenters import _mask_points

    target = torch.zeros(101, 91, dtype=torch.long)
    target[:3] = 255
    target[50, 20:40] = 1
    target[-1, -1] = 2
    with torch.random.fork_rng():
        torch.manual_seed(42)
        labels, truth, grid = _mask_points(target, 256)
    assert labels.tolist() == [0, 1, 2] and truth.shape == (3, 256)
    x = ((grid[0, :, 0, 0] + 1) * 48).floor().long()
    y = ((grid[0, :, 0, 1] + 1) * 64).floor().long()
    assert torch.equal(truth.bool(), target[y, x] == labels[:, None])
    assert (truth.sum(0) == 1).all() and truth[1].sum() >= 32 and truth[2].sum() >= 32
    small = torch.zeros(9, 17, dtype=torch.long)
    small[-1, -1] = 1
    small_labels, small_truth, _ = _mask_points(small, 12544)
    assert small_labels.tolist() == [0, 1] and small_truth[1].sum() >= small.numel() // 4
    small.zero_()
    background, bg_truth, _ = _mask_points(small, 64)
    assert background.tolist() == [0] and bg_truth.shape == (1, 64) and bg_truth.all()
    with pytest.raises(ValueError, match="클래스 수"):
        _mask_points(target, 2)
    assert _mask_points(torch.full_like(target, 255), 256) is None


def test_mask2former_each_decoder_gets_an_independent_assignment_and_gradient():
    from types import SimpleNamespace
    from transformer_segmenters import _decoder_query_loss

    native = SimpleNamespace(num_points=128, empty_weight=torch.tensor([1., 1., .1]),
                             weight_dict={"loss_cross_entropy": 2., "loss_dice": 5., "loss_mask": 5.})
    target = torch.full((1, 32, 32), 255)
    target[:, :2, :4] = 0
    target[:, :2, 4:8] = 1
    classes = torch.tensor([[[2., 0., -1.], [0., 2., -1.], [-1., -1., 2.]]], requires_grad=True)
    masks = torch.zeros(1, 3, 32, 32, requires_grad=True)
    permutation = [2, 0, 1]
    aux_classes = classes.detach()[:, permutation].requires_grad_()
    aux_masks = masks.detach()[:, permutation].requires_grad_()
    result = SimpleNamespace(class_queries_logits=classes, masks_queries_logits=masks, auxiliary_logits=[])
    with pytest.raises(ValueError, match="4개 이상"):
        _decoder_query_loss(result, torch.tensor([[[0, 1], [2, 3]]]), native)
    with torch.random.fork_rng():
        torch.manual_seed(801)
        final_alone = _decoder_query_loss(result, target, native)
        result.auxiliary_logits = [{"class_queries_logits": aux_classes, "masks_queries_logits": aux_masks}]
        torch.manual_seed(801)
        supervised = _decoder_query_loss(result, target, native)
    torch.testing.assert_close(supervised, final_alone)
    supervised.backward()
    assert aux_classes.grad.abs().sum() > 0 and aux_masks.grad.abs().sum() > 0
    assert masks.grad[:, :, 2:].count_nonzero() == aux_masks.grad[:, :, 2:].count_nonzero() == 0
    torch.testing.assert_close(masks.grad[:, permutation], aux_masks.grad)


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
            if teacher == "mask2former_swin_small":
                assert not widget.mask2former_queries_spin.isHidden()
                assert widget.mask2former_queries_spin.value() == 100
                widget.mask2former_queries_spin.setValue(16)
                assert widget.patience_spin.minimum() == 0
                assert widget.patience_spin.value() == 15
                widget.patience_spin.setValue(0)
                widget._sync_config()
                assert project.training.early_stop_patience == 0
                path = Path(ProjectManager.save(project))
                widget.set_project(ProjectManager.load(path))
                assert widget.patience_spin.value() == 0 and "사용 안 함" in widget.patience_spin.text()
                assert widget.project.training.mask2former_num_queries == widget.mask2former_queries_spin.value() == 16
                widget.mode_combo.setCurrentIndex(widget.mode_combo.findData("builtin_resume"))
                assert not widget.mask2former_resume_queries.isHidden() and widget.mask2former_queries_spin.isHidden()
                widget.mode_combo.setCurrentIndex(widget.mode_combo.findData("builtin_transfer"))
                assert widget.mask2former_resume_queries.isHidden() and not widget.mask2former_queries_spin.isHidden()
                assert widget.mask2former_queries_spin.value() == 16
                legacy = json.loads(path.read_text(encoding="utf-8"))
                del legacy["training"]["mask2former_num_queries"]
                path.write_text(json.dumps(legacy), encoding="utf-8")
                widget.set_project(ProjectManager.load(path))
                assert widget.mask2former_queries_spin.value() == 100
            else:
                assert widget.mask2former_queries_spin.isHidden() and widget.mask2former_resume_queries.isHidden()
                assert widget.patience_spin.minimum() == 1
                assert widget.patience_spin.value() == 15
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
        query_options = {"mask2former_num_queries": 16} if model_id == "mask2former_swin_small" else {}
        teacher = train_builtin(model_id, output_dir=root / "teacher", pretrained=pretrained,
                                **query_options, **args)
        payload = torch.load(teacher, map_location="cpu", weights_only=True)
        if pretrained:
            assert payload["weight_provenance"]["source"] not in ("random", "random_init")
        model = load_builtin_checkpoint(payload)
        if query_options:
            assert payload["model_config"]["num_queries"] == 16
            assert model.net.model.transformer_module.queries_features.num_embeddings == 16
        with torch.no_grad():
            out = model(torch.randn(1, 3, 65, 79))
            assert out.shape == (1, 2, 65, 79) and out.isfinite().all()
            if query_options:
                sample = torch.randn(1, 3, 64, 64)
                expected_teacher = model(sample).numpy()
        del model, payload
        if query_options:
            teacher_onnx = root / "teacher.onnx"
            export_checkpoint(teacher, teacher_onnx, verify=True)
            runtime = ort.InferenceSession(str(teacher_onnx), providers=["CPUExecutionProvider"])
            actual_teacher = runtime.run(None, {runtime.get_inputs()[0].name: sample.numpy()})[0]
            np.testing.assert_allclose(actual_teacher, expected_teacher, rtol=.001, atol=.001)
            del runtime
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
        query_key = "model.transformer_module.queries_features.weight"
        if model_id == "mask2former_swin_small":
            assert .9 < float(tensors[query_key].std()) < 1.1
            tensors[query_key][0, :3] = torch.tensor([.25, -.75, 1.5])
        key = next(k for k in tensors if k.endswith("projection.weight") or k.endswith("proj.weight"))
        sentinel = tensors[key].flatten()[0].item()
        file = Path(directory) / "model.safetensors"
        save_file(tensors, str(file))
        del source, tensors
        target = build_builtin_model(model_id, 2, backbone_weights=file)
        assert target.weight_provenance["source"] == "local_hf_segmentation"
        assert target.net.state_dict()[key].flatten()[0].item() == sentinel
        if model_id == "mask2former_swin_small":
            torch.testing.assert_close(target.net.state_dict()[query_key][0, :3],
                                       torch.tensor([.25, -.75, 1.5]))
        assert target.net.state_dict()[TASK_HEADS[model_id][0] + "weight"].shape[0] == (
            3 if model_id == "mask2former_swin_small" else 2)
        with pytest.raises(ValueError, match="가중치"):
            target.initialize_from_weights({"wrong": torch.zeros(1)}, file)
