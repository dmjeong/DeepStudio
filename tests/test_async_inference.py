"""실제 이벤트 루프가 추론 중 동작하고 중단 후 완료 결과를 보존하는지 검증."""

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT / "python")]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
AVAILABLE = all(importlib.util.find_spec(name) for name in ["torch", "PySide6", "cv2", "matplotlib"])


def wait_for_inference(widget, timeout=45):
    from PySide6.QtWidgets import QApplication
    deadline = time.monotonic() + timeout
    while widget._inference_worker is not None:
        QApplication.processEvents()
        if time.monotonic() > deadline:
            widget._inference_worker.stop()
            widget._inference_worker.wait(10000)
            raise AssertionError("추론 워커 종료 시간 초과")
        time.sleep(.002)
    QApplication.processEvents()


@unittest.skipUnless(AVAILABLE, "Qt/PyTorch runtime required")
class AsyncInferenceTests(unittest.TestCase):
    def test_preview_write_failure_retains_computed_result_and_stops_batch(self):
        from types import SimpleNamespace
        from core.inference_types import InferenceResult
        from core.inference_worker import InferenceWorker
        calls, results = [], []
        def infer(path):
            calls.append(path)
            return InferenceResult(path, "ok", "classify", "NG", inference_sec=.002, inference_status="completed")
        def save(*_args):
            raise OSError("disk full")
        engine = SimpleNamespace(infer=infer, _current_preview_rgb=None, _heatmap_cache=None, info="")
        worker = InferenceWorker(engine, ["first", "second"], SimpleNamespace(put=save))
        worker.result_ready.connect(results.append)
        worker.run()
        self.assertEqual(calls, ["first"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].summary, "NG")
        self.assertEqual(results[0].inference_sec, .002)
        self.assertIn("캐시 저장 실패", worker.error)

    def test_async_batch_reports_per_image_failures(self):
        from PySide6.QtWidgets import QApplication
        from core.inference_types import InferenceResult
        from core.inference_engine import InferenceEngine
        from widgets.inference_widget import InferenceWidget
        from PIL import Image
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            good, bad = [str(Path(folder) / name) for name in ("good.png", "bad.png")]
            Image.new("RGB", (24, 24), "white").save(good)
            Path(bad).write_bytes(b"invalid image")
            widget = InferenceWidget()
            widget.model = object()
            widget._batch_images = [good, bad]
            widget._current_image = good
            def infer(engine, path):
                return (InferenceResult(path, "error", "", "ERROR", error="invalid image") if path == bad else
                        InferenceResult(path, "ok", "classify", "OK"))
            try:
                with patch.object(InferenceEngine, "infer", infer):
                    widget._run_inference()
                    wait_for_inference(widget)
                self.assertIn("오류 포함 완료", widget.batch_time_label.text())
                self.assertIn("오류 1장", widget.batch_time_label.text())
                self.assertEqual(len(widget._inference_results), 2)
            finally:
                if widget._inference_worker is not None:
                    wait_for_inference(widget)
                widget.deleteLater()
                app.processEvents()

    def test_start_returns_and_gui_timer_ticks_while_model_is_busy(self):
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QApplication
        from core.inference_types import InferenceResult
        from core.inference_engine import InferenceEngine
        from widgets.inference_widget import InferenceWidget
        from PIL import Image
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            paths = [str(Path(folder) / f"{index}.png") for index in range(3)]
            for path in paths:
                Image.new("RGB", (24, 24), "white").save(path)
            widget = InferenceWidget()
            widget.model = object()
            widget._batch_images = paths
            widget._current_image = paths[0]
            widget._clear_results()
            ticks, threads = [], []
            entered, release = threading.Event(), threading.Event()
            def infer(engine, path):
                threads.append(threading.get_ident())
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test release timeout")
                engine._current_preview_rgb = np.full((24, 24, 3), 90, np.uint8)
                return InferenceResult(path, "ok", "classify", "OK", inference_sec=.001, inference_status="completed")
            timer = QTimer()
            timer.setInterval(5)
            timer.timeout.connect(lambda: ticks.append(1))
            try:
                with patch.object(InferenceEngine, "infer", infer):
                    timer.start()
                    widget._run_inference()
                    self.assertTrue(entered.wait(2))
                    deadline = time.monotonic() + .08
                    while time.monotonic() < deadline:
                        app.processEvents()
                        time.sleep(.002)
                    self.assertGreater(len(ticks), 2)
                    self.assertNotEqual(threads[0], threading.get_ident())
                    with self.assertRaises(RuntimeError):
                        widget._clear_model()
                    widget._cancel_inference()
                    release.set()
                    wait_for_inference(widget)
                self.assertEqual(len(widget._inference_results), 1)
                self.assertIn("중단", widget.batch_time_label.text())
                with patch.object(InferenceEngine, "infer", side_effect=AssertionError("re-inference")):
                    widget._display_cached_result(paths[0])
                self.assertEqual(widget.image_label._result_text, "OK")
            finally:
                timer.stop()
                release.set()
                if widget._inference_worker is not None:
                    wait_for_inference(widget)
                widget.deleteLater()
                app.processEvents()
