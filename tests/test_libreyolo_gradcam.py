"""Native LibreYOLO + Qt cache contracts (no pretrained download required)."""
import numpy as np
import pytest
import torch
from PIL import Image

from core.inference_loading import load_inference_engine


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    pytest.importorskip("libreyolo")
    from libreyolo import LibreMobileNetV4
    with torch.random.fork_rng():
        torch.manual_seed(11)
        native = LibreMobileNetV4(size="s", nb_classes=2, device="cpu")
    native.names = {0: "정상", 1: "균열"}
    with torch.no_grad():
        # Positive spatial CAM even with scratch weights.
        native.model.classifier.weight[1].fill_(.001)
        native.model.classifier.bias[1] = 1
    path = tmp_path_factory.mktemp("libre-cam") / "model.pt"
    native.save(str(path))
    return path


def image_at(tmp_path, size):
    w, h = size
    rgb = np.random.default_rng(4).integers(0, 256, (h, w, 3), dtype=np.uint8)
    rgb[:h // 2, :w // 2] = (230, 60, 10)
    path = tmp_path / "컬러.png"
    Image.fromarray(rgb).save(path)
    return path, rgb


@pytest.mark.parametrize("size", [(331, 119), (117, 333), (224, 224)])
def test_native_cam_preserves_prediction_weights_and_crop(checkpoint, tmp_path, size):
    from core.heatmap import render_heatmap
    path, rgb = image_at(tmp_path, size)
    engine = load_inference_engine(checkpoint, gradcam=True)
    model = engine._upstream_model.model
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    before = engine._upstream_model.predict(str(path), imgsz=224).probs.data
    result = engine.infer(str(path))
    assert result.status == "ok", result.error
    assert result.gradcam_status == "completed", engine.info
    np.testing.assert_array_equal(result.details["probabilities"], before.cpu().numpy())
    assert engine._gradcam.cam.diagnostics["feature_shape"] == (7, 7)
    assert engine._gradcam.cam.diagnostics["gradient_max"] > 0
    assert engine._gradcam.cam._fwd_hook is None
    assert engine._gradcam.cam._tensor_hook is None
    for name, value in model.state_dict().items():
        assert torch.equal(value, weights[name]), name
    assert all(p.grad is None for p in model.parameters())
    cache = engine._heatmap_cache
    assert cache["activation"].shape == rgb.shape[:2]
    assert cache["valid_mask"].any() and not cache["valid_mask"].all()
    assert cache["activation"].max() > 0
    _, overlay = render_heatmap(cache["activation"], rgb, 1., valid_mask=cache["valid_mask"])
    np.testing.assert_array_equal(overlay[~cache["valid_mask"]], rgb[~cache["valid_mask"]])
    engine.gradcam_enabled = False
    repeated = engine.infer(str(path))
    assert repeated.gradcam_status == "not_run"
    assert engine._heatmap_cache is None
    assert repeated.details["probabilities"] == result.details["probabilities"]
    np.testing.assert_array_equal(engine._original_preview_rgb, rgb)


def test_cam_uses_identical_native_input_and_recovers_from_error(checkpoint, tmp_path, monkeypatch):
    path, rgb = image_at(tmp_path, (333, 118))
    engine = load_inference_engine(checkpoint, gradcam=True)
    native_tensor = engine._upstream_model._preprocess(str(path), input_size=224)[0]
    seen = []
    generate = engine._gradcam.cam.generate_activation

    def record(tensor, **kwargs):
        seen.append(tensor)
        return generate(tensor, **kwargs)

    monkeypatch.setattr(engine._gradcam.cam, "generate_activation", record)
    first = engine.infer(str(path))
    assert first.gradcam_status == "completed"
    assert torch.equal(seen[0], native_tensor)

    def fail(*args, **kwargs):
        raise RuntimeError("intentional CAM failure")

    monkeypatch.setattr(engine._gradcam, "generate_image", fail)
    result = engine.infer(str(path))
    assert result.status == "ok" and result.gradcam_status == "error"
    assert result.details["probabilities"] == first.details["probabilities"]
    assert "intentional CAM failure" in engine.info
    assert engine._heatmap_cache is None
    np.testing.assert_array_equal(engine._current_preview_rgb, rgb)


def test_native_inspection_and_process_cache(checkpoint, tmp_path):
    from webapp.worker import JobContext, inspect_model, infer
    path, rgb = image_at(tmp_path, (333, 119))
    ctx = JobContext(tmp_path, monitor_parent=False)
    inspection = inspect_model(ctx, {"weights": str(checkpoint), "device": "cpu"})
    assert inspection["output"]["gradcam_supported"] is True
    infer(ctx, {"weights": str(checkpoint), "images": [str(path)], "gradcam": True})
    with np.load(tmp_path / "results/0.npz", allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored["original"], rgb)
        assert stored["activation"].shape == rgb.shape[:2]


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_upstream_cache_compare_survives_missing_source_and_grid(app, tmp_path):
    from core.inference_cache import InferencePreviewCache
    from core.inference_worker import InferenceWorker
    from widgets.inference_widget import InferenceWidget
    from tests.test_upstream_inference import _Classifier, _Detector, _engine
    from PySide6.QtCore import QPointF

    widget = InferenceWidget()
    widget.resize(1200, 900)
    widget.show()
    cache = InferencePreviewCache()
    widget._preview_cache = cache
    try:
        for index, (native, task) in enumerate(((_Classifier(), "classify"), (_Detector(), "detect"))):
            folder = tmp_path / str(index)
            folder.mkdir()
            path, rgb = image_at(folder, (117 + index * 7, 61))
            engine = _engine(native, task)
            worker = InferenceWorker(engine, [str(path)], cache)
            results = []
            worker.result_ready.connect(results.append)
            worker.run()
            assert not worker.error
            path.unlink()
            widget._batch_images = [str(path)]
            widget._inference_results[str(path)] = results[0]
            widget._display_cached_result(str(path))
            widget._switch_view_mode("grid")
            widget.compare_check.setChecked(False)
            widget.compare_check.setChecked(True)
            app.processEvents()
            assert widget._view_mode == "single"
            assert widget.original_label.isVisible()
            np.testing.assert_array_equal(widget.original_label._pixel_source, rgb)
            assert widget.original_label._source_pixmap.size() == widget.image_label._source_pixmap.size()
            widget.image_label.set_view(2., QPointF(12, 15))
            widget.image_label.view_changed.emit(2., QPointF(12, 15))
            assert widget.original_label._zoom == 2.
            assert widget.original_label._pan == QPointF(12, 15)
            worker.deleteLater()
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


def test_ui_respects_process_gradcam_capability(app):
    from widgets.inference_widget import InferenceWidget
    widget = InferenceWidget()
    try:
        for task, supported in (("classify", True), ("detect", False), ("classify", True)):
            widget._model_inspected({"status": "completed", "output": {
                "weights": "libre.pt", "task": task, "class_names": ["OK", "NG"],
                "input_size": [224, 224], "device": "cpu", "upstream": True,
                "gradcam_supported": supported}})
            assert widget.gradcam_checkbox.isEnabled() == supported
            assert widget._gradcam_requested() == supported
    finally:
        widget.close()
        widget.deleteLater()
        app.processEvents()


def test_actual_model_load_button_enables_click_and_worker_renders_cam(app, checkpoint, tmp_path, monkeypatch):
    """No synthetic inspection response: click -> subprocess -> checkbox -> infer."""
    import time
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QPushButton
    from webapp.jobs import JobManager
    from widgets.inference_widget import InferenceWidget

    manager = JobManager(tmp_path / "jobs")
    monkeypatch.setattr("core.job_manager._manager", manager)
    widget = InferenceWidget()
    widget.resize(1280, 900)
    widget.show()
    path, rgb = image_at(tmp_path, (333, 119))
    before = checkpoint.read_bytes()

    def wait_for(field):
        deadline = time.monotonic() + 60
        while getattr(widget, field) is not None:
            app.processEvents()
            assert time.monotonic() < deadline, (field, widget.model_info.text())
            time.sleep(.005)
        app.processEvents()

    try:
        widget.infer_device_combo.setCurrentIndex(widget.infer_device_combo.findData("cpu"))
        widget.ckpt_edit.setText(str(checkpoint))
        button = next(b for b in widget.findChildren(QPushButton) if b.text() == "모델 로드")
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        wait_for("_model_inspection")
        assert widget._process_model, widget.model_info.text()
        assert widget.gradcam_checkbox.isEnabled(), widget.gradcam_checkbox.toolTip()
        assert widget.gradcam_checkbox.isVisible()
        # Click the actual control twice, including explicitly enabling it.
        if widget.gradcam_checkbox.isChecked():
            QTest.mouseClick(widget.gradcam_checkbox, Qt.MouseButton.LeftButton)
        assert not widget.gradcam_checkbox.isChecked()
        QTest.mouseClick(widget.gradcam_checkbox, Qt.MouseButton.LeftButton)
        assert widget.gradcam_checkbox.isChecked()
        assert widget._gradcam_requested()
        widget._current_image = str(path)
        QTest.mouseClick(widget.infer_btn, Qt.MouseButton.LeftButton)
        wait_for("_inference_worker")
        result = widget._inference_results[str(path)]
        assert result.status == "ok", result.error
        assert result.gradcam_status == "completed", widget.gradcam_info.text()
        assert widget._heatmap_cache["kind"] == "Grad-CAM"
        assert "클래스: 균열" in widget.gradcam_info.text()
        QTest.mouseClick(widget.compare_check, Qt.MouseButton.LeftButton)
        app.processEvents()
        assert widget.original_label.isVisible()
        np.testing.assert_array_equal(widget.original_label._pixel_source, rgb)
        assert checkpoint.read_bytes() == before  # An existing model needs no retraining.
        jobs = manager.list(limit=10)
        assert {j["kind"] for j in jobs} == {"inspect_model", "infer"}
        assert all(j["status"] == "completed" for j in jobs)
    finally:
        for field in ("_model_inspection", "_inference_worker"):
            worker = getattr(widget, field)
            if worker is not None:
                worker.stop()
                worker.wait(15000)
                app.processEvents()
        manager.close()
        widget.close()
        widget.deleteLater()
        app.processEvents()
