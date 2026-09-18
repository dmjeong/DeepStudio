"""공식 모델과의 수치 호환, 저장 계약과 실제 역전파 검증."""

import copy
from pathlib import Path
import random

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import models

from efficientnet import EfficientNet, load_imagenet


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("architecture,count", [("efficientnet_b0", 5288548), ("efficientnet_b1", 7794184)])
def test_reference_layers_and_state_keys(architecture, count):
    reference = getattr(models, architecture)(weights=None).eval()
    actual = EfficientNet(architecture, in_channels=3).eval()
    assert set(actual.state_dict()) == set(reference.state_dict())
    assert actual.get_param_count()["total"] == count
    actual.load_state_dict(reference.state_dict(), strict=True)
    expected_stages, actual_stages = {}, {}
    handles = []
    for index in range(9):
        handles.append(reference.features[index].register_forward_hook(
            lambda m, args, out, i=index: expected_stages.__setitem__(i, out.detach().clone())))
        handles.append(actual.features[index].register_forward_hook(
            lambda m, args, out, i=index: actual_stages.__setitem__(i, out.detach().clone())))
    try:
        for shape in [(1, 3, 64, 80), (1, 3, 65, 79)]:
            image = torch.randn(shape)
            with torch.no_grad():
                expected, result = reference(image), actual(image)
            torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)
            for index in range(9):
                torch.testing.assert_close(actual_stages[index], expected_stages[index], rtol=1e-5, atol=1e-6)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("architecture", ["efficientnet_b0", "efficientnet_b1"])
def test_training_backward_matches_reference_and_inspector(architecture):
    from layer_debug import LayerInspector
    reference = getattr(models, architecture)(weights=None, num_classes=3).train()
    actual = EfficientNet(architecture, num_classes=3, in_channels=3).train()
    actual.load_state_dict(reference.state_dict())
    image = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([0, 2])
    inspector = LayerInspector(["features.0", "classifier.1"]).attach(actual)
    try:
        torch.manual_seed(17)
        expected = reference(image)
        torch.nn.functional.cross_entropy(expected, targets).backward()
        torch.manual_seed(17)
        result = actual(image)
        torch.nn.functional.cross_entropy(result, targets).backward()
        torch.testing.assert_close(result, expected, rtol=2e-4, atol=1e-5)
        for (name, parameter), (other_name, other) in zip(actual.named_parameters(), reference.named_parameters()):
            assert name == other_name
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-3, atol=2e-4)
        assert inspector.records["features.0"]["gradient"]["finite"]
        actual.eval()
        with torch.no_grad():
            actual(image)
        assert "gradient" in inspector.records["features.0"]
    finally:
        inspector.close()
    assert not actual.features[0]._forward_hooks


def test_legacy_grayscale_adapter_keeps_official_three_channel_weights():
    from efficientnet_contract import LEGACY_GRAY_INPUT
    rgb = EfficientNet(num_classes=2, in_channels=3).eval()
    grayscale = EfficientNet(num_classes=2, in_channels=1, input_adapter=LEGACY_GRAY_INPUT).eval()
    grayscale.load_state_dict(rgb.state_dict(), strict=True)
    raw = torch.rand(2, 1, 32, 40)
    with torch.no_grad():
        expected = rgb((raw - grayscale.rgb_mean) / grayscale.rgb_std)
        result = grayscale((raw - .449) / .226)
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)
    assert grayscale.features[0][0].weight.shape[1] == 3


def test_strict_pretrained_load_and_head_replacement(tmp_path):
    source = models.efficientnet_b0(weights=None).state_dict()
    path = tmp_path / "imagenet.pth"
    torch.save(source, path)
    model = EfficientNet(num_classes=2, in_channels=3)
    head = model.classifier[1].weight.detach().clone()
    load_imagenet(model, weights_path=path)
    torch.testing.assert_close(model.features[0][0].weight, source["features.0.0.weight"])
    torch.testing.assert_close(model.classifier[1].weight, head)
    source.pop("features.1.0.block.0.0.weight")
    torch.save(source, path)
    with pytest.raises(RuntimeError):
        load_imagenet(model, weights_path=path)


def make_project(root, data_root):
    from core.project import ProjectManager
    project = ProjectManager.create_new("EfficientNet test", "classify", str(root), ["a", "b"])
    project.data.root = str(data_root)
    project.training.training_mode = "efficientnet_transfer"
    project.training.device = "cpu"
    project.training.input_size = 32
    project.training.batch_size = 2
    project.training.epochs = 2
    project.training.use_amp = False
    project.training.selection_metric = "val_loss"
    project.training.augmentation.horizontal_flip = 0
    project.training.augmentation.rotation = 0
    project.training.augmentation.color_jitter = 0
    return project


def train(project, stop_after_first=False):
    from core.efficientnet_trainer import EfficientNetTrainWorker
    from core.training_engine import TrainingEvents
    worker = EfficientNetTrainWorker(project, signals=TrainingEvents())
    worker.debug_num_workers = 0
    if stop_after_first:
        worker.signals.epoch_finished.connect(lambda epoch, *args: worker.stop() if epoch == 1 else None)
    worker._run_training()
    return worker


def test_actual_training_resume_matches_uninterrupted(tmp_path):
    data = tmp_path / "data"
    for split in ["train", "val"]:
        for index, name in enumerate(["a", "b"]):
            folder = data / split / name
            folder.mkdir(parents=True)
            pixels = np.full((40, 44, 3), 40 + index * 150, dtype=np.uint8)
            Image.fromarray(pixels).save(folder / "sample.png")
    weights = tmp_path / "initial.pth"
    torch.save(models.efficientnet_b0(weights=None).state_dict(), weights)
    complete = make_project(tmp_path / "complete", data)
    complete.model.pretrained_weights = str(weights)
    interrupted = copy.deepcopy(complete)
    interrupted.project_dir = str(tmp_path / "interrupted")
    for project, stopping in [(complete, False), (interrupted, True)]:
        torch.manual_seed(101)
        np.random.seed(101)
        random.seed(101)
        train(project, stopping)
    partial_path = next((Path(interrupted.project_dir) / "runs").rglob("last.pt"))
    resumed = copy.deepcopy(interrupted)
    resumed.project_dir = str(tmp_path / "resumed")
    resumed.training.training_mode = "efficientnet_resume"
    resumed.training.layer_debug_enabled = True
    resumed.training.layer_debug_patterns = "features.0,classifier.1"
    resumed.training.layer_debug_batches = 1
    resumed.model.pretrained_weights = str(partial_path)
    train(resumed)
    expected = torch.load(next((Path(complete.project_dir) / "runs").rglob("last.pt")), weights_only=False)
    actual = torch.load(next((Path(resumed.project_dir) / "runs").rglob("last.pt")), weights_only=False)
    assert actual["epoch"] == 2
    assert not actual["debug_run"]
    assert resumed.runs[-1].config_snapshot["layer_debug"]["samples"] == 1
    assert resumed.training.layer_debug_enabled
    assert len(actual["metrics_history"]["val_loss"]) == 2
    assert actual["best_epoch"] == expected["best_epoch"]
    for key in expected["model_state_dict"]:
        torch.testing.assert_close(actual["model_state_dict"][key], expected["model_state_dict"][key], atol=0, rtol=0)
    from core.inference_loading import load_cpu_engine
    engine = load_cpu_engine(resumed.runs[-1].checkpoint_path, gradcam=True)
    assert engine.model.architecture == "efficientnet_b0"
    bad = copy.deepcopy(resumed)
    Image.new("RGB", (40, 44)).save(data / "train" / "a" / "new.png")
    bad.model.pretrained_weights = str(partial_path)
    with pytest.raises(ValueError, match="데이터 변경"):
        train(bad)


@pytest.mark.parametrize("channels", [1, 3])
def test_export_reload_and_gradcam(tmp_path, channels):
    from checkpoint import make_checkpoint_metadata
    from export_onnx import export_checkpoint, load_custom_model
    from core.gradcam import GradCAM
    model = EfficientNet(num_classes=2, in_channels=channels).eval()
    checkpoint = make_checkpoint_metadata("classify", 2, ["a", "b"], 32, channels,
        model.checkpoint_config(),
        center_crop={"width": 40, "height": 40})
    checkpoint.update(engine="efficientnet", architecture_name="efficientnet_b0", model_state_dict=model.state_dict())
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    restored = load_custom_model(checkpoint)
    sample = torch.randn(1, channels, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(restored(sample), model(sample))
    cam = GradCAM(restored)
    try:
        assert cam.target_layer == "features.8"
        heatmap = cam.generate_activation(sample, target_class=0)
        assert np.isfinite(heatmap).all()
    finally:
        cam.release()
    output = tmp_path / "model.onnx"
    export_checkpoint(path, output, dynamic_batch=True)
    import json
    config = json.loads(output.with_suffix(".json").read_text())
    assert config["backend"] == "efficientnet"
    assert config["architecture"] == "efficientnet_b0"
    assert config["cpp_supported"] and config["verification"] == "passed"
    assert config["preprocessing"]["center_crop"] == {"width": 40, "height": 40}


@pytest.mark.parametrize("mode", ["efficientnet_finetune", "efficientnet_transfer", "efficientnet_resume"])
def test_desktop_val_loss_results_use_efficientnet_policy(tmp_project, mode):
    from PySide6.QtWidgets import QApplication
    from widgets.training_widget import TrainingWidget
    from widgets.training_results import _metric_info_for
    app = QApplication.instance() or QApplication([])
    tmp_project.training.training_mode = mode
    tmp_project.training.selection_metric = "val_loss"
    widget = TrainingWidget()
    try:
        # Restoring the project and rebuilding the start-screen metric cards
        # previously raised before the new worker could run, leaving old results.
        widget.set_project(tmp_project)
        widget._rebuild_metric_cards("classify")
        info = _metric_info_for(tmp_project)
        assert info["primary"] == "val_loss"
        assert "val_loss" in info["display"]
        assert widget.selection_combo.currentData() == "val_loss"
        app.processEvents()
    finally:
        widget.close()
        widget.deleteLater()


def test_desktop_model_selection_and_roundtrip(tmp_project):
    from PySide6.QtWidgets import QApplication
    from core.project import ProjectManager
    from widgets.training_widget import TrainingWidget
    app = QApplication.instance() or QApplication([])
    widget = TrainingWidget()
    try:
        widget.set_project(tmp_project)
        widget.mode_combo.setCurrentIndex(widget.mode_combo.findData("efficientnet_finetune"))
        widget.efficientnet_model_combo.setCurrentIndex(widget.efficientnet_model_combo.findData("efficientnet_b1"))
        assert widget.input_size_spin.value() == 240
        assert widget.selection_combo.findData("val_loss") >= 0
        widget.collect_config()
        restored = ProjectManager.load(ProjectManager.save(tmp_project))
        assert restored.training.efficientnet_model == "efficientnet_b1"
        assert restored.training.input_size == 240
        widget.mode_combo.setCurrentIndex(widget.mode_combo.findData("efficientnet_resume"))
        assert not widget.epochs_spin.isEnabled()
        app.processEvents()
    finally:
        widget.close()
