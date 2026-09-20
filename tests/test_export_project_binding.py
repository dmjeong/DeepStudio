"""The export page must not carry a checkpoint across project boundaries."""

import os

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
        assert widget.output_edit.text().endswith("second/exports/model_segment.onnx")
        with pytest.raises(ValueError, match="태스크"):
            widget._validate_checkpoint_for_project(str(checkpoint))
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()
