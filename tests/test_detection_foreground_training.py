import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from core.project import ProjectManager
from core.trainer import TrainWorker
from core.training_modes import validate_training_options
from detection import foreground_supervision_loss
from export_onnx import export_checkpoint


def test_foreground_loss_supervises_even_subcell_boxes():
    logits = torch.zeros((1, 1, 8, 8), requires_grad=True)
    labels = torch.tensor([[[0, .34, .34, .001, .001], [0, 0, 0, 0, 0]]])
    foreground_supervision_loss(logits, labels).backward()
    grad = logits.grad[0, 0]
    assert grad[2, 2] < 0
    assert torch.count_nonzero(grad < 0) == 1
    assert torch.count_nonzero(grad > 0) == 63


def test_foreground_training_exports_a_regular_detector(tmp_path):
    pytest.importorskip("onnxruntime")
    project = ProjectManager.create_new("foreground", "detect", str(tmp_path), ["crack", "material"])
    project.model.model_id = ""
    project.model.backbone_channels = [8, 16, 32, 64, 128]
    project.model.csp_depth = [1, 1, 1, 1]
    cfg = project.training
    cfg.training_mode = "custom"
    cfg.input_size, cfg.epochs, cfg.batch_size, cfg.device = 64, 1, 2, "cpu"
    cfg.detection_foreground_supervision = True
    image = np.full((64, 64, 3), 120, np.uint8)
    image[21:24, 18:23] = 232
    for folder in (project.data.train_dir, project.data.val_dir):
        directory = Path(folder)
        for i in range(2):
            Image.fromarray(image).save(directory / f"{i}.png")
            label_dir = directory.parents[1] / "labels" / directory.name
            (label_dir / f"{i}.txt").write_text(f"{i} .32 .35 .08 .05\n", encoding="utf-8")

    worker = TrainWorker(project)
    worker.debug_num_workers = 0
    worker._run_training()
    saved = project.runs[-1].checkpoint_path
    assert project.runs[-1].status == "completed"
    checkpoint = torch.load(saved, map_location="cpu", weights_only=False)
    assert checkpoint["detection_foreground_supervision"]
    assert not any("foreground_aux" in name for name in checkpoint["model_state_dict"])
    output = tmp_path / "detector.onnx"
    result = export_checkpoint(saved, output, log=lambda _: None)
    assert result["verification"] == "passed"
    if probe := os.environ.get("DETECT_PROBE"):
        result = subprocess.run([probe, result["config_path"], str(directory / "0.png")],
                                capture_output=True, text=True, check=True)
        assert "detections" in json.loads(result.stdout.strip().splitlines()[-1])

    cfg.training_mode = "upstream_scratch"
    project.model.model_id = "re_detr_v4_small"
    with pytest.raises(ValueError, match="전경 보조 학습"):
        validate_training_options(project)


def test_desktop_detect_options_follow_the_selected_model(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from widgets.training_widget import TrainingWidget

    app = QApplication.instance() or QApplication([])
    project = ProjectManager.create_new("detect", "detect", str(tmp_path), ["crack"])
    widget = TrainingWidget()
    try:
        widget.set_project(project)
        assert widget.small_object_coverage_check.isHidden()
        widget.patch_check.click()
        assert not widget.small_object_coverage_check.isHidden()
        assert not widget.small_object_coverage_check.isEnabled()
        widget.patch_count_spin.setValue(2)
        assert widget.small_object_coverage_check.isEnabled()
        widget.small_object_coverage_check.setChecked(True)
        widget.collect_config()
        assert project.training.small_object_coverage
        widget.model_id_combo.setCurrentIndex(widget.model_id_combo.findData(""))
        assert widget.small_object_coverage_check.isHidden()
        assert not widget.detection_foreground_check.isHidden()
        widget.detection_foreground_check.setChecked(True)
        widget.collect_config()
        assert project.model.model_id == ""
        assert project.training.detection_foreground_supervision and not project.training.small_object_coverage
        validate_training_options(project)
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
