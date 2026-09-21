"""The export page must not carry a checkpoint across project boundaries."""

import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from core.project import ProjectManager, RunRecord
from widgets.export_widget import ExportWidget


def test_project_switch_clears_old_checkpoint_and_rejects_mismatched_metadata(tmp_path):
    import torch

    app = QApplication.instance() or QApplication([])
    first = ProjectManager.create_new("first", "classify", str(tmp_path / "first"), ["ok", "ng"])
    checkpoint = tmp_path / "first.pt"
    torch.save({"task": "classify", "model_id": "efficientnet_b0",
                "model_config": {"architecture": "efficientnet_b0"}}, checkpoint)
    first.runs.append(RunRecord(checkpoint_path=str(checkpoint)))
    second = ProjectManager.create_new("second", "segment", str(tmp_path / "second"), ["background", "part"])
    widget = ExportWidget()
    try:
        widget.set_project(first)
        assert os.path.abspath(widget.ckpt_edit.text()) == str(checkpoint)
        widget.set_project(second)
        assert widget.ckpt_edit.text() == ""
        assert Path(widget.output_edit.text()) == tmp_path / "second" / "exports" / "model_segment.onnx"
        with pytest.raises(ValueError, match="태스크"):
            widget._validate_checkpoint_for_project(str(checkpoint))
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


def test_efficientnet_project_binds_local_images_and_never_enables_fp64_by_default(tmp_path):
    import cv2
    import numpy as np
    app = QApplication.instance() or QApplication([])
    project = ProjectManager.create_new("images", "classify", str(tmp_path / "project"), ["ok", "ng"])
    project.model.model_id = "efficientnet_b1"
    for name in project.data.class_names:
        folder = os.path.join(project.data.train_dir, name)
        os.makedirs(folder, exist_ok=True)
        cv2.imwrite(os.path.join(folder, "sample.png"), np.zeros((16, 16), np.uint8))
    widget = ExportWidget()
    try:
        widget.set_project(project)
        assert widget.dataset_verify_check.isChecked()
        assert widget.validation_edit.text() == project.data.train_dir
        assert not widget.precision_check.isChecked()
        other = ProjectManager.create_new("other", "segment", str(tmp_path / "other"), ["background", "part"])
        widget.set_project(other)
        assert not widget.dataset_verify_check.isChecked()
        assert widget.validation_edit.text() == ""
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


def test_export_worker_passes_explicit_validation_options(monkeypatch):
    from widgets import export_widget
    captured = {}
    def export(*args, **kwargs):
        captured.update(kwargs)
        return {}
    monkeypatch.setattr(export_widget, "export_checkpoint", export)
    worker = export_widget.ExportWorker("source.pt", "output.onnx", 17, False, False, True,
                                       validation_dir="local-images", allow_precision_fallback=False)
    worker.run()
    assert captured["validation_dir"] == "local-images"
    assert captured["allow_precision_fallback"] is False
