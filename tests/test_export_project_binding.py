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


def test_efficientnet_project_accepts_training_folder_with_empty_class(tmp_path):
    import cv2
    import numpy as np
    app = QApplication.instance() or QApplication([])
    project = ProjectManager.create_new("partial", "classify", str(tmp_path / "project"), ["ok", "double"])
    project.model.model_id = "efficientnet_b0"
    os.makedirs(os.path.join(project.data.train_dir, "ok"), exist_ok=True)
    os.makedirs(os.path.join(project.data.train_dir, "double"), exist_ok=True)
    cv2.imwrite(os.path.join(project.data.train_dir, "ok", "sample.png"), np.zeros((16, 16), np.uint8))
    widget = ExportWidget()
    try:
        widget.set_project(project)
        assert widget.dataset_verify_check.isChecked()
        assert widget.validation_edit.text() == project.data.train_dir
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


def test_precision_comparison_uses_shared_worker_and_displays_thread_results(tmp_path, monkeypatch):
    import sys
    import time
    from types import SimpleNamespace
    from widgets.precision_benchmark_dialog import PrecisionBenchmarkDialog

    captured = {}

    def benchmark(**kwargs):
        captured.update(kwargs)
        kwargs["log"]("비교 실행")
        return [
            {"precision": "FP32", "threads": 0, "p50_ms": 10, "p95_ms": 12, "model_p50_ms": 8,
             "comparison": "기준", "status": "완료"},
            {"precision": "INT8", "threads": 0, "p50_ms": 6, "p95_ms": 7, "model_p50_ms": 5,
             "comparison": "판정 100%", "status": "완료"},
        ]

    monkeypatch.setitem(sys.modules, "precision_benchmark", SimpleNamespace(
        parse_thread_counts=lambda raw: tuple(int(value) for value in raw.split(",")),
        benchmark_precisions=benchmark,
    ))
    config = tmp_path / "fp32.json"
    config.write_text("{}", encoding="utf-8")
    app = QApplication.instance() or QApplication([])
    dialog = PrecisionBenchmarkDialog(fp32_path=str(config), image_dir=str(tmp_path), calibration_dir=str(tmp_path))
    try:
        dialog.run_fp16.setChecked(False)
        dialog.fp16.setText("ignored-fp16.json")
        dialog.threads.setText("0, 8")
        dialog.start_button.click()
        assert dialog.isRunning()
        deadline = time.monotonic() + 5
        while dialog.isRunning() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert not dialog.isRunning()
        assert captured["thread_counts"] == (0, 8) and captured["calibration_dir"] == str(tmp_path)
        assert captured["fp16_path"] is None and not captured["run_fp16"]
        assert captured["run_int8"] and dialog.table.rowCount() == 2
        assert dialog.table.item(0, 1).text() == dialog.table.item(1, 1).text() == "자동"
        assert dialog.table.item(1, 3).text() == "7.00"
        assert dialog.table.item(1, 5).text() == "판정 100%"
        assert "비교 실행" in dialog.log.toPlainText()
    finally:
        if dialog.worker is not None:
            dialog.worker.requestInterruption()
            dialog.worker.wait()
            app.processEvents()
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


def test_encryption_extension_and_key_do_not_leak_across_projects(tmp_path):
    app = QApplication.instance() or QApplication([])
    widget = ExportWidget()
    try:
        project = ProjectManager.create_new("one", "classify", str(tmp_path / "one"), ["ok", "ng"])
        widget.set_project(project)
        widget.encrypt_check.setChecked(True)
        widget.key_edit.setText("private.key")
        assert widget.output_edit.text().endswith(".dvsenc")
        widget.encrypt_check.setChecked(False)
        assert widget.output_edit.text().endswith(".onnx")
        widget.encrypt_check.setChecked(True)
        other = ProjectManager.create_new("two", "segment", str(tmp_path / "two"), ["bg", "part"])
        widget.set_project(other)
        assert not widget.encrypt_check.isChecked() and not widget.key_edit.text()
        assert widget.output_edit.text().endswith(".onnx")
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("sam2_model_id", ["", "sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_base_plus", "sam2_hiera_large"])
def test_gui_worker_routes_encryption_and_sam2_through_common_service(monkeypatch, sam2_model_id):
    from widgets import export_widget
    captured = {}
    def export(*args, **kwargs):
        captured.update(kwargs)
        return {"encrypted": True}
    monkeypatch.setattr(export_widget, "export_checkpoint", export)
    worker = export_widget.ExportWorker("source.pt", "output.dvsenc", 17, False, False, True,
                                       encryption_key_path="private.key", sam2_model_id=sam2_model_id)
    worker.run()
    assert captured["encryption_key_path"] == "private.key"
    assert captured["sam2_model_id"] == sam2_model_id
