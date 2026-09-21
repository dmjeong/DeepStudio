"""Exercise the Studio button and its process worker with real, local B0 weights."""
import hashlib
import importlib.util
import json
import gc
import os
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytestmark = pytest.mark.skipif(
    not all(importlib.util.find_spec(name) for name in ("torch", "torchvision", "onnx", "onnxruntime", "cv2")),
    reason="EfficientNet ONNX dependencies required")


@pytest.fixture
def checkpoint_factory(tmp_path):
    import torch
    from checkpoint import make_checkpoint_metadata
    from efficientnet import EfficientNet
    from PIL import Image
    previous = torch.get_num_threads()
    torch.set_num_threads(2)

    def make(channels=1, size=(224, 224)):
        with torch.random.fork_rng():
            torch.manual_seed(17)
            model = EfficientNet(num_classes=2, in_channels=channels).eval()
            with torch.no_grad():
                for layer in model.modules():
                    if isinstance(layer, torch.nn.BatchNorm2d):
                        layer.running_mean.uniform_(-.1, .1)
                        layer.running_var.uniform_(.7, 1.3)
                model.classifier[1].bias.copy_(torch.tensor([-.1, .2]))
        saved = make_checkpoint_metadata("classify", 2, ["OK", "NG"], size, channels)
        saved.update(engine="efficientnet", model_config=model.checkpoint_config(), model_state_dict=model.state_dict())
        weights = tmp_path / f"b0-{channels}.pt"
        torch.save(saved, weights)
        shape = (240, 256) if channels == 1 else (240, 256, 3)
        pixels = np.random.default_rng(17).integers(0, 256, shape, dtype=np.uint8)
        image = tmp_path / f"입력-{channels}.png"
        Image.fromarray(pixels).save(image)
        return str(weights), str(image)

    yield make
    torch.set_num_threads(previous)


@pytest.mark.parametrize("channels,size", [(1, (224, 224)), (3, (40, 56))])
def test_real_onnx_matches_loaded_pytorch_and_keeps_cam(checkpoint_factory, channels, size):
    from core.inference_loading import load_cpu_engine
    from core.efficientnet_onnx import EfficientNetOnnx
    weights, image = checkpoint_factory(channels, size)
    digests = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (weights, image)}
    engine = load_cpu_engine(weights, runtime="auto", gradcam=True, threads=2,
                             input_region={"mode": "json", "center_crop": {"width": 200, "height": 220}})
    reference = load_cpu_engine(weights, runtime="pytorch", input_region=engine.input_region)
    engine.prepare()
    runtime = engine._onnx_runtime
    try:
        with patch.object(EfficientNetOnnx, "__init__", side_effect=AssertionError("prepared twice")):
            engine.prepare()
        engine.gradcam_enabled = False
        expected = reference.infer(image)
        # A classification must succeed even if the retained PyTorch forward cannot run.
        with patch.object(engine.model, "forward", side_effect=AssertionError("classification used PyTorch")):
            actual = engine.infer(image)
        assert actual.status == expected.status == "ok", actual.error
        np.testing.assert_allclose(actual.details["probabilities"], expected.details["probabilities"], rtol=1e-5, atol=1e-6)
        assert actual.summary == expected.summary
        assert actual.details["runtime"] == "onnxruntime"
        assert expected.details["runtime"] == "pytorch"
        stages = actual.details["timing_ms"]
        assert set(stages) == {"decode", "preprocess", "model", "postprocess"}
        assert all(value >= 0 for value in stages.values())
        assert sum(stages.values()) <= actual.inference_sec * 1000 + .001
        assert runtime.input_shape == (1, channels, *size)
        engine.gradcam_enabled = True
        with patch.object(engine.model, "forward", wraps=engine.model.forward) as forward:
            cam = engine.infer(image)
        assert forward.call_count > 0
        assert cam.status == "ok", cam.error
        assert cam.gradcam_status == "completed"
        assert cam.gradcam_sec > 0
        assert cam.details["runtime"] == "onnxruntime"
        np.testing.assert_array_equal(actual.details["probabilities"], cam.details["probabilities"])
        assert engine._heatmap_cache is not None
        assert np.isfinite(engine._heatmap_cache["activation"]).all()
        assert {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in digests} == digests
    finally:
        engine._gradcam.release()


@pytest.mark.parametrize("error_type,message", [(RuntimeError, "export failed"), (ImportError, "onnx unavailable")])
def test_auto_setup_failure_restores_pytorch_and_keeps_gradcam(checkpoint_factory, error_type, message):
    from core.inference_loading import load_cpu_engine
    weights, image = checkpoint_factory(size=(32, 40))
    engine = load_cpu_engine(weights, runtime="auto", gradcam=True)
    reference = load_cpu_engine(weights, runtime="pytorch")
    expected = reference.infer(image)
    # A collected test parameter must not retain live model frames via an exception traceback.
    failure = error_type(message)
    try:
        with patch("export_onnx.export_to_onnx", side_effect=failure) as export:
            engine.prepare()
            result = engine.infer(image)
            again = engine.infer(image)
        assert export.call_count == 1, "A failed accelerator must not be retried per image"
        for actual in (result, again):
            assert actual.status == "ok", actual.error
            assert actual.summary == expected.summary
            assert actual.details["runtime"] == "pytorch"
            assert str(failure) in actual.details["runtime_warning"]
            assert "runtime_optimization" not in actual.details
            assert actual.inference_sec > 0
            assert actual.gradcam_status == "completed"
            assert actual.gradcam_sec > 0
            np.testing.assert_array_equal(actual.details["probabilities"], expected.details["probabilities"])
        assert engine._onnx_runtime is None
        assert engine.runtime == "pytorch"
        assert engine._heatmap_cache is not None
        assert np.isfinite(engine._heatmap_cache["activation"]).all()
    finally:
        engine._gradcam.release()


def test_explicit_onnx_export_failure_stays_strict(checkpoint_factory):
    from core.inference_loading import load_cpu_engine
    weights, image = checkpoint_factory(size=(32, 40))
    engine = load_cpu_engine(weights, runtime="onnx")
    with patch("export_onnx.export_to_onnx", side_effect=RuntimeError("export failed")):
        result = engine.infer(image)
    assert result.status == "error"
    assert "export failed" in result.error
    assert result.inference_sec is None


def test_process_worker_persists_fallback_predictions_and_warning(checkpoint_factory, tmp_path):
    from core.inference_loading import load_cpu_engine
    from webapp.worker import JobContext, infer
    weights, image = checkpoint_factory(size=(32, 40))
    second_image = str(Path(image).with_name("worker-second.png"))
    Path(second_image).write_bytes(Path(image).read_bytes())
    expected = load_cpu_engine(weights, runtime="pytorch").infer(image)
    directory = tmp_path / "fallback-job"
    directory.mkdir()
    context = JobContext(directory, monitor_parent=False)
    with patch("export_onnx.export_to_onnx", side_effect=RuntimeError("worker export failure")) as export:
        outcome = infer(context, {"weights": weights, "images": [image, second_image],
                                  "device": "cpu", "gradcam": False})
    assert export.call_count == 1
    assert outcome["status"] == "completed", outcome
    assert outcome["output"]["completed"] == 2
    assert outcome["output"]["errors"] == 0
    for index in range(2):
        saved = json.loads((directory / "results" / f"{index}.json").read_text(encoding="utf-8"))
        assert saved["status"] == "ok"
        assert saved["cache_ready"]
        assert saved["summary"] == expected.summary
        assert saved["details"]["runtime"] == "pytorch"
        assert saved["details"]["runtime_requested"] == "auto"
        assert "worker export failure" in saved["details"]["runtime_warning"]
        np.testing.assert_array_equal(saved["details"]["probabilities"], expected.details["probabilities"])
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    warnings = [entry["args"][0] for entry in events if entry["event"] == "log_message"]
    assert warnings == [saved["details"]["runtime_warning"]]
    assert sum(entry["event"] == "inference_result" for entry in events) == 2


def test_nonfinite_pytorch_reference_is_not_reported_as_an_onnx_optimizer_failure(checkpoint_factory):
    import torch
    from core.inference_loading import load_cpu_engine
    weights, image = checkpoint_factory(size=(32, 40))
    engine = load_cpu_engine(weights, runtime="auto")
    with torch.no_grad():
        engine.model.classifier[1].bias[0] = float("nan")
    with patch("onnxruntime.InferenceSession", side_effect=AssertionError("invalid reference must stop first")):
        result = engine.infer(image)
    assert result.status == "error"
    assert "모델 출력이 유한하지 않습니다" in result.error
    assert "기준 PyTorch 출력" in engine.runtime_warning
    assert "ONNX 출력 비교 전 중단" in engine.runtime_warning
    assert engine._onnx_runtime is None


@pytest.mark.parametrize("runtime,passing_level", [
    ("auto", "basic"), ("auto", "disabled"), ("auto", None), ("onnx", None),
])
def test_only_a_numerically_verified_onnx_session_is_accepted(checkpoint_factory, runtime, passing_level):
    import onnxruntime as ort
    from core.inference_loading import load_cpu_engine
    from core.inference_timing import format_result_timing
    weights, image = checkpoint_factory(size=(32, 40))
    engine = load_cpu_engine(weights, runtime=runtime, threads=2)
    reference = load_cpu_engine(weights, runtime="pytorch")
    expected = reference.infer(image)
    original_session = ort.InferenceSession
    levels = {ort.GraphOptimizationLevel.ORT_ENABLE_ALL: "all",
              ort.GraphOptimizationLevel.ORT_ENABLE_BASIC: "basic",
              ort.GraphOptimizationLevel.ORT_DISABLE_ALL: "disabled"}
    attempted = []

    class PerturbedSession:
        def __init__(self, *args, **kwargs):
            self.level = levels[args[1].graph_optimization_level]
            attempted.append(self.level)
            self.real = original_session(*args, **kwargs)

        def __getattr__(self, key):
            return getattr(self.real, key)

        def run(self, *args, **kwargs):
            outputs = self.real.run(*args, **kwargs)
            if self.level != passing_level:
                outputs[0][0, 0] += np.float32(2.8339844)
            return outputs

    with patch.object(ort, "InferenceSession", PerturbedSession):
        if passing_level is None:
            result = engine.infer(image)
            if runtime == "onnx":
                assert result.status == "error"
                failure = result.error
                assert result.inference_sec is None
            else:
                assert result.status == "ok", result.error
                assert result.summary == expected.summary
                assert result.details["runtime"] == "pytorch"
                failure = result.details["runtime_warning"]
                assert result.inference_sec > 0
                assert "runtime_optimization" not in result.details
                assert "runtime_validation_attempts" not in result.details
                np.testing.assert_array_equal(result.details["probabilities"], expected.details["probabilities"])
                # Classification now runs the retained model; it never runs the rejected ORT session.
                with patch.object(engine.model, "forward", wraps=engine.model.forward) as forward:
                    again = engine.infer(image)
                assert forward.call_count == 1
                assert again.status == "ok", again.error
                assert again.details["runtime_warning"] == failure
            assert "모든 최적화 설정" in failure
            for part in ("all/seeded", "basic/seeded", "disabled/seeded", "PyTorch=", "ONNX=", "출력 범위"):
                assert part in failure
            assert attempted == ["all", "basic", "disabled"] * 2 + ["disabled", "disabled"]
            assert engine._onnx_runtime is None
        else:
            engine.prepare()
            with patch.object(engine.model, "forward", side_effect=AssertionError("used PyTorch fallback")):
                result = engine.infer(image)
            assert result.status == "ok", result.error
            assert attempted == (["all", "basic"] if passing_level == "basic" else ["all", "basic", "disabled"])
            assert result.details["runtime"] == "onnxruntime"
            assert result.details["runtime_optimization"] == passing_level
            assert result.details["runtime_validation_attempts"][-1]["passed"]
            assert not any(item["passed"] for item in result.details["runtime_validation_attempts"][:-1])
            np.testing.assert_allclose(result.details["probabilities"], expected.details["probabilities"], atol=1e-6)
            assert ("기본 최적화" if passing_level == "basic" else "최적화 꺼짐") in format_result_timing(result)


def wait_until(app, predicate, timeout=60):
    deadline = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        assert time.monotonic() < deadline, "GUI worker timeout"
        time.sleep(.005)
    app.processEvents()


@pytest.fixture(scope="session")
def qt_app():
    # Qt owns process-wide resources; keep one application alive across GUI tests.
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication
    application = QApplication.instance() or QApplication([])
    yield application
    # Fault-injection tracebacks can hold completed QThreads in reference cycles.
    # Collect them while their QApplication and event dispatcher are still alive.
    gc.collect()
    application.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    application.processEvents()


@pytest.mark.skipif(not all(importlib.util.find_spec(name) for name in ("PySide6", "matplotlib", "psutil")),
                    reason="desktop dependencies required")
@pytest.mark.parametrize("process", [False, True])
def test_actual_button_uses_onnx_and_restores_timing(checkpoint_factory, tmp_path, monkeypatch, process, qt_app):
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QMessageBox
    from core import job_manager
    from core.efficientnet_onnx import EfficientNetOnnx
    from widgets.inference_widget import InferenceWidget
    weights, image = checkpoint_factory()
    monkeypatch.setenv("DEEP_STUDIO_DESKTOP_STATE_DIR", str(tmp_path / "jobs"))
    # Isolate the process manager from any running user application or previous test.
    monkeypatch.setattr(job_manager, "_manager", None)
    app = qt_app
    widget = InferenceWidget()
    widget.resize(1280, 900)
    widget.show()
    try:
        widget.infer_device_combo.setCurrentIndex(widget.infer_device_combo.findData("cpu"))
        widget.gradcam_checkbox.setChecked(False)
        with patch.object(QMessageBox, "critical", side_effect=AssertionError("model load failed")):
            if process:
                widget._start_model_inspection(weights)
                wait_until(app, lambda: widget._model_inspection is None)
                assert widget._process_model, widget.model_info.text()
            else:
                assert widget._load_model(weights)
        widget._batch_images = [image]
        widget._current_image = image
        widget.infer_btn.click()
        wait_until(app, lambda: widget._inference_worker is None)
        assert image in widget._inference_results, widget.batch_time_label.text()
        result = widget._inference_results[image]
        assert result.status == "ok", result.error
        assert result.details["runtime"] == "onnxruntime"
        assert result.details["runtime_threads"] == 4
        assert result.details["runtime_optimization"] == "all"
        assert result.details["runtime_verification_tolerance"] == {"atol": 1e-3, "rtol": 5e-4}
        assert "ONNX Runtime cpu" in widget.result_card.time_label.text()
        assert "파일 읽기" in widget.result_card.time_label.text()
        assert "모델" in widget.result_card.time_label.text()
        assert widget.selected_timing_label.isVisible()
        assert "ONNX Runtime cpu" in widget.selected_timing_label.text()
        assert "파일 읽기" in widget.selected_timing_label.text()
        assert "ONNX Runtime" in widget.result_review.model.item(0, 6).toolTip()
        if process:
            from core.desktop_jobs import PersistentInferenceCache, inference_result
            restored = PersistentInferenceCache(widget._preview_cache.root.parent)
            saved = inference_result(restored.entries[image])
            assert saved.details == result.details
            assert saved.inference_sec == result.inference_sec
            assert restored.get(image)["preview"] is not None
        else:
            runtime = widget._onnx_runtime
            with patch.object(EfficientNetOnnx, "__init__", side_effect=AssertionError("second click re-exported")):
                widget.infer_btn.click()
                wait_until(app, lambda: widget._inference_worker is None)
            assert widget._onnx_runtime is runtime
            assert widget._inference_results[image].status == "ok"
        before = widget.result_card.time_label.text()
        visible_before = widget.selected_timing_label.text()
        with patch.object(EfficientNetOnnx, "logits", side_effect=AssertionError("cache view reran inference")):
            widget._display_cached_result(image)
        assert widget.result_card.time_label.text() == before
        assert widget.selected_timing_label.text() == visible_before
    finally:
        if widget._inference_worker is not None:
            widget._inference_worker.stop()
            wait_until(app, lambda: widget._inference_worker is None)
        if widget._model_inspection is not None:
            widget._model_inspection.stop()
            wait_until(app, lambda: widget._model_inspection is None)
        widget._clear_model()
        widget.close()
        widget.deleteLater()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


@pytest.mark.skipif(not all(importlib.util.find_spec(name) for name in ("PySide6", "matplotlib", "psutil")),
                    reason="desktop dependencies required")
def test_actual_button_completes_batch_when_automatic_export_fails(checkpoint_factory, qt_app):
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QMessageBox
    from core.inference_loading import load_cpu_engine
    from widgets.inference_widget import InferenceWidget
    weights, image = checkpoint_factory(size=(32, 40))
    # Two distinct paths exercise a real batch, including model reuse after a failed setup.
    second_image = str(Path(image).with_name("두번째-입력.png"))
    Path(second_image).write_bytes(Path(image).read_bytes())
    reference = load_cpu_engine(weights, runtime="pytorch")
    expected = reference.infer(image)
    widget = InferenceWidget()
    widget.show()
    export_failure = RuntimeError("local export unavailable")
    try:
        widget.infer_device_combo.setCurrentIndex(widget.infer_device_combo.findData("cpu"))
        widget.gradcam_checkbox.setChecked(False)
        with patch.object(QMessageBox, "critical", side_effect=AssertionError("inference must remain available")):
            assert widget._load_model(weights)
            widget._batch_images = [image, second_image]
            widget._current_image = image
            with patch("export_onnx.export_to_onnx", side_effect=export_failure) as export:
                widget.infer_btn.click()
                wait_until(qt_app, lambda: widget._inference_worker is None)
            export_failure.__traceback__ = None
            assert export.call_count == 1
        for path in (image, second_image):
            assert path in widget._inference_results, widget.batch_time_label.text()
            result = widget._inference_results[path]
            assert result.status == "ok", result.error
            assert result.details["runtime"] == "pytorch"
            assert "local export unavailable" in result.details["runtime_warning"]
            assert result.summary == expected.summary
            np.testing.assert_array_equal(result.details["probabilities"], expected.details["probabilities"])
            # Review is served from the completed result and must preserve the fallback explanation.
            widget._display_cached_result(path)
            assert "PyTorch cpu" in widget.selected_timing_label.text()
            assert "ONNX 준비 실패 → PyTorch" in widget.selected_timing_label.text()
            assert "local export unavailable" in widget.selected_timing_label.toolTip()
            assert "PyTorch cpu" in widget.result_card.time_label.text()
            assert "ONNX Runtime cpu" not in widget.result_card.time_label.text()
        assert widget.infer_btn.isEnabled()
        assert widget._onnx_runtime is None
    finally:
        export_failure.__traceback__ = None
        if widget._inference_worker is not None:
            widget._inference_worker.stop()
            wait_until(qt_app, lambda: widget._inference_worker is None)
        widget._clear_model()
        widget.close()
        widget.deleteLater()
        qt_app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qt_app.processEvents()
