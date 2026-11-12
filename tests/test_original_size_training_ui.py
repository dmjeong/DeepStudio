"""The original-size mode stays exclusive through Qt controls and project reloads."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from core.dataset_regions import training_resolution_label, training_scale
from core.project import ProjectManager
from widgets.training_widget import TrainingWidget


def test_original_size_and_patch_modes_do_not_mix(tmp_path):
    app = QApplication.instance() or QApplication([])
    project = ProjectManager.create_new("segment", "segment", str(tmp_path / "native"), ["background", "crack"])
    patch_project = ProjectManager.create_new("patch", "segment", str(tmp_path / "patch"), ["background", "crack"])
    for candidate in (project, patch_project):
        candidate.model.model_id = "unet_resnet18"
        candidate.training.training_mode = "builtin_scratch"
        candidate.training.patch_training = True
        candidate.training.patch_scale = 0.25
    project.training.patchcore_crop_enabled = True
    project.training.patchcore_crop_width = 3500
    project.training.patchcore_crop_height = 1504
    widget = TrainingWidget()
    try:
        widget.set_project(project)
        assert widget.patch_check.isChecked()
        assert widget.input_size_spin.isEnabled()
        assert not widget.original_size_check.isHidden()

        widget.original_size_check.click()
        assert widget.original_size_check.isChecked()
        assert not widget.patch_check.isChecked() and not widget.patch_check.isEnabled()
        assert not widget.input_size_spin.isEnabled()
        assert widget.patch_scale_combo.isHidden() and widget.patch_count_spin.isHidden()
        assert "3500×1504" in widget.pc_crop_info.text()
        widget.collect_config()
        assert project.training.original_size_training and not project.training.patch_training
        assert training_resolution_label(project) == "원본 크기 학습"
        assert training_scale(project, (3840, 2160)) == (1.0, 1.0)
        saved = ProjectManager.save(project)

        widget.set_project(patch_project)
        assert not widget.original_size_check.isChecked()
        assert widget.patch_check.isChecked() and widget.patch_check.isEnabled()
        assert widget.input_size_spin.isEnabled()

        widget.set_project(ProjectManager.load(saved))
        assert widget.original_size_check.isChecked() and not widget.patch_check.isEnabled()
        widget._apply_advice({"patch_training": True})
        assert widget.patch_check.isChecked() and widget.patch_check.isEnabled()
        assert not widget.original_size_check.isChecked()
        assert not widget.project.training.original_size_training and widget.project.training.patch_training

        widget.original_size_check.click()
        widget.model_id_combo.setCurrentIndex(widget.model_id_combo.findData("sam2_hiera_tiny"))
        assert widget.original_size_check.isHidden() and not widget.original_size_check.isChecked()
        assert not widget.patch_check.isEnabled()
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
