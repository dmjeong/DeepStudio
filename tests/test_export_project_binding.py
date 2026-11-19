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


@pytest.mark.parametrize("with_images", [False, True])
def test_fp16_export_button_saves_separate_deployment_and_reopens_for_comparison(tmp_path, monkeypatch, with_images):
    import json
    import time

    import torch
    import numpy as np
    from PIL import Image
    from PySide6.QtWidgets import QMessageBox
    from builtin_models import build_builtin_model, make_builtin_checkpoint
    from widgets import export_widget
    from widgets.precision_benchmark_dialog import PrecisionBenchmarkDialog

    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    app = QApplication.instance() or QApplication([])
    project = ProjectManager.create_new("fp16", "segment", str(tmp_path / "project"), ["bg", "crack"])
    project.model.model_id = "unet_lite"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        network = build_builtin_model("unet_lite", 2, 1).eval()
    checkpoint = tmp_path / "best.pt"
    torch.save(make_builtin_checkpoint("unet_lite", network, num_classes=2, input_size=32,
                                       in_channels=1, class_names=["bg", "crack"]), checkpoint)
    project.runs.append(RunRecord(checkpoint_path=str(checkpoint)))
    calls, dialogs, locations = [], [], []
    exporter = export_widget.export_checkpoint

    def export(source, destination, **options):
        calls.append((source, options))
        return exporter(source, destination, **options)

    monkeypatch.setattr(export_widget, "export_checkpoint", export)
    for method in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, method, lambda _, title, msg: dialogs.append((title, msg)))
    monkeypatch.setattr(PrecisionBenchmarkDialog, "exec",
                        lambda dialog: locations.append((dialog.fp32.text(), dialog.fp16.text())))
    widget = ExportWidget()
    try:
        widget.set_project(project)
        baseline = Path(widget.output_edit.text())
        baseline.write_bytes(b"fp32")
        widget.verify_check.setChecked(False)
        widget.precision_check.setChecked(True)
        widget.calibration_edit.clear()
        widget.export_precision.setCurrentIndex(widget.export_precision.findData("float16"))
        destination = Path(widget.output_edit.text())
        assert destination == baseline.with_name("model_segment_fp16.onnx")
        assert widget.verify_check.isChecked() and not widget.verify_check.isEnabled()
        assert not widget.calibration_edit.isEnabled()
        assert widget.dataset_verify_check.isEnabled() and not widget.dataset_verify_check.isChecked()
        validation = tmp_path / "images"
        if with_images:
            validation.mkdir()
            for i in (0, 1):
                Image.fromarray(np.full((32, 32), i * 170, np.uint8)).save(validation / f"{i}.png")
            widget.dataset_verify_check.setChecked(True)
            widget.validation_edit.setText(str(validation))
        widget.export_btn.click()
        deadline = time.monotonic() + 60
        while getattr(widget, "worker", None) is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert dialogs and [title for title, _ in dialogs] == ["완료"], dialogs
        assert "FP16 실행·FP32 출력 비교 완료" in dialogs[0][1]
        assert f"실제 이미지 {2 if with_images else 0}장" in dialogs[0][1] and "픽셀 일치율" in dialogs[0][1]
        assert baseline.read_bytes() == b"fp32"
        manifest = json.loads(destination.with_suffix(".json").read_text(encoding="utf-8"))
        assert manifest["export"]["precision"] == "float16" and manifest["verification"] == "fp16_compared"
        from onnx_session import cpu_session_options
        session = ort.InferenceSession(str(destination.parent / manifest["model_path"]),
                                       cpu_session_options(num_threads=1), providers=["CPUExecutionProvider"])
        output, = session.run(None, {manifest["input_name"]: np.ones((1, 1, 32, 32), np.float32)})
        assert output.shape == (1, 2, 32, 32) and np.isfinite(output).all()
        source, options = calls[0]
        assert source == str(checkpoint) and options["precision"] == "float16"
        assert options["verify"] and not options["allow_precision_fallback"] and not options["calibration_dir"]
        assert options["validation_dir"] == (str(validation) if with_images else None)
        widget.benchmark_btn.click()
        assert locations == [("", str(destination.with_suffix(".json")))]
        widget.encrypt_check.setChecked(True)
        assert Path(widget.output_edit.text()) == destination.with_suffix(".dvsenc")
        widget.export_precision.setCurrentIndex(widget.export_precision.findData("float32"))
        assert Path(widget.output_edit.text()) == baseline.with_suffix(".dvsenc")
        assert widget.verify_check.isEnabled() and not widget.verify_check.isChecked()
    finally:
        if getattr(widget, "worker", None) is not None:
            widget.worker.wait()
            app.processEvents()
        widget.deleteLater()
        app.processEvents()


def _blocked_native_stage(*, progress):
    import time

    progress("워밍업 1/2")
    time.sleep(60)


def _benchmark_events(options, workspace, *, log, on_row, cancelled):
    from precision_benchmark import _run_step

    log("비교 실행")
    rows = [
        {"backend": "onnxruntime", "precision": "FP32", "threads": 0,
         "p50_ms": 10, "p95_ms": 12, "model_p50_ms": 8, "model_p95_ms": 9,
         "comparison": "기준", "status": "완료"},
        {"backend": "openvino", "precision": "FP32", "threads": 0,
         "p50_ms": 6, "p95_ms": 7, "model_p50_ms": 5, "model_p95_ms": 6,
         "comparison": "판정 100%", "status": "완료"},
        {"backend": "openvino", "precision": "INT8", "threads": None,
         "status": "준비 실패", "detail": "지원하지 않는 그래프"},
    ]
    for row in rows:
        on_row(row)
        if options["run_fp16"]:
            # Exercise cancellation while the native worker is still running.
            def signal_stage(stage):
                log(stage)
                if stage.startswith("워밍업"):
                    Path(options["image_dir"], "workspace").write_text(workspace, encoding="utf-8")

            _run_step(_blocked_native_stage, stage="모델 로딩 중", progress=signal_stage,
                      timeout=60, cancelled=cancelled, workspace=workspace)
    return rows


@pytest.mark.parametrize("stop_early", [False, True])
def test_precision_comparison_uses_shared_worker_and_displays_thread_results(tmp_path, monkeypatch, stop_early):
    import time
    import precision_benchmark
    from widgets.precision_benchmark_dialog import PrecisionBenchmarkDialog

    monkeypatch.setattr(precision_benchmark, "run_desktop_benchmark", _benchmark_events)
    config = tmp_path / "fp32.json"
    config.write_text("{}", encoding="utf-8")
    app = QApplication.instance() or QApplication([])
    dialog = PrecisionBenchmarkDialog(fp32_path=str(config), image_dir=str(tmp_path), calibration_dir=str(tmp_path))
    try:
        dialog.run_fp16.setChecked(stop_early)
        if not stop_early:
            dialog.fp16.setText("ignored-fp16.json")
        dialog.threads.setText("0, 8")
        dialog.start_button.click()
        assert dialog.isRunning()
        captured = dialog.worker.options
        deadline = time.monotonic() + 20
        while dialog.isRunning() and time.monotonic() < deadline:
            app.processEvents()
            if stop_early and dialog.table.rowCount() and (tmp_path / "workspace").exists():
                dialog.cancel_button.click()
            time.sleep(.01)
        assert not dialog.isRunning()
        if stop_early:
            assert dialog.table.rowCount() == 1
            assert "중단" in dialog.status.text()
            assert not Path((tmp_path / "workspace").read_text(encoding="utf-8")).exists()
            return
        assert captured["thread_counts"] == (0, 8) and captured["calibration_dir"] == str(tmp_path)
        assert captured["backends"] == ("onnxruntime", "openvino")
        assert captured["fp16_path"] is None and not captured["run_fp16"]
        assert captured["run_int8"] and dialog.table.rowCount() == 3
        assert dialog.table.item(0, 0).text() == "ONNX Runtime"
        assert dialog.table.item(1, 0).text() == "OpenVINO"
        assert dialog.table.item(0, 2).text() == dialog.table.item(1, 2).text() == "자동"
        assert dialog.table.item(1, 4).text() == "7.00"
        assert dialog.table.item(1, 6).text() == "6.00"
        assert dialog.table.item(1, 7).text() == "판정 100%"
        assert dialog.table.item(2, 8).toolTip() == "지원하지 않는 그래프"
        assert "실패 1개" in dialog.status.text()
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


def test_cpu_comparison_builds_missing_fp32_and_measures_matching_original_images(tmp_path):
    import time

    import numpy as np
    from PIL import Image
    import torch

    from builtin_models import build_builtin_model, make_builtin_checkpoint
    from checkpoint import record_original_size_training
    from widgets.precision_benchmark_dialog import PrecisionBenchmarkDialog

    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    app = QApplication.instance() or QApplication([])
    net = build_builtin_model("unet_lite", 2, 1).eval()
    weights = make_builtin_checkpoint(
        "unet_lite", net, num_classes=2, input_size=(32, 64),
        in_channels=1, class_names=["bg", "defect"],
    )
    record_original_size_training(weights, (63, 31))
    checkpoint = tmp_path / "best.pt"
    torch.save(weights, checkpoint)
    images = tmp_path / "시험 영상"
    images.mkdir()
    Image.fromarray(np.zeros((24, 24), np.uint8)).save(images / "다른 해상도.png")
    Image.fromarray(np.zeros((31, 63), np.uint8)).save(images / "원본.png")
    dialog = PrecisionBenchmarkDialog(
        image_dir=str(images), calibration_dir=str(images),
        checkpoint_export={"checkpoint_path": str(checkpoint)},
    )
    try:
        dialog.backend.setCurrentIndex(1)  # ONNX Runtime
        dialog.threads.setText("1")
        dialog.image_count.setValue(1)
        dialog.warmup.setValue(0)
        dialog.repeats.setValue(1)
        dialog.start_button.click()
        deadline = time.monotonic() + 60
        while dialog.isRunning() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert not dialog.isRunning()
        log = dialog.log.toPlainText()
        assert "비교용 FP32 ONNX 생성 중" in log
        assert "1장 제외" in log and "63×31 px" in log
        assert dialog.table.rowCount() == 3, log
        assert all(dialog.table.item(i, 8).text() == "완료" for i in range(3)), log
        assert not list(tmp_path.glob("*.onnx"))
    finally:
        if dialog.worker is not None:
            dialog.worker.requestInterruption()
            dialog.worker.wait()
            app.processEvents()
        dialog.close()
        dialog.deleteLater()
        app.processEvents()
