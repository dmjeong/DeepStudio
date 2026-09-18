"""추론과 설명 맵의 계산 시간을 분리하고 저장된 측정값을 복원한다."""

import ast
import builtins
from dataclasses import dataclass, replace
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gui"))
sys.path.insert(0, str(ROOT / "python"))

from core.inference_timing import (
    StageTimer, batch_stage_summary, format_duration, format_result_timing,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
GUI_AVAILABLE = all(importlib.util.find_spec(name) is not None
                    for name in ("PySide6", "torch", "torchvision", "cv2", "matplotlib"))


class ManualClock:
    def __init__(self):
        self.now = 0.0
        self.events = []

    def __call__(self):
        self.events.append("clock")
        return self.now

    def advance(self, seconds):
        self.now += seconds


class StageTimerTests(unittest.TestCase):
    def test_cpu_measures_only_context_and_does_not_synchronize(self):
        clock = ManualClock()
        backend = SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock()),
                                  mps=SimpleNamespace(synchronize=Mock()))
        timer = StageTimer("cpu", torch_module=backend, clock=clock)
        self.assertIsNone(timer.elapsed_sec)
        self.assertEqual(timer.status, "not_run")
        clock.advance(20)
        with timer:
            clock.advance(0.125)
        clock.advance(30)
        self.assertEqual(timer.elapsed_sec, 0.125)
        self.assertFalse(timer.failed)
        self.assertEqual(timer.status, "completed")
        backend.cuda.synchronize.assert_not_called()
        backend.mps.synchronize.assert_not_called()

    def test_cpu_does_not_require_importing_torch(self):
        original_import = builtins.__import__

        def reject_torch(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise AssertionError("CPU 시간 측정에 torch 임포트 불필요")
            return original_import(name, *args, **kwargs)

        clock = ManualClock()
        with patch("builtins.__import__", side_effect=reject_torch):
            timer = StageTimer("cpu", clock=clock)
            with timer:
                clock.advance(0.25)
        self.assertEqual(timer.elapsed_sec, 0.25)

    def test_cuda_excludes_previous_work_and_waits_for_current_work(self):
        clock = ManualClock()
        waits = iter((100.0, 0.375))

        def synchronize(device):
            self.assertEqual(device, "cuda:2")
            clock.events.append("cuda:2")
            clock.advance(next(waits))

        backend = SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock(side_effect=synchronize)))
        timer = StageTimer("cuda:2", torch_module=backend, clock=clock)
        with timer:
            clock.events.append("model")
            clock.advance(0.125)
        self.assertEqual(timer.elapsed_sec, 0.5)
        self.assertEqual(clock.events, ["cuda:2", "clock", "model", "cuda:2", "clock"])
        self.assertEqual(backend.cuda.synchronize.call_count, 2)

    def test_mps_uses_argument_free_synchronization(self):
        clock = ManualClock()

        def synchronize():
            clock.events.append("mps")

        backend = SimpleNamespace(mps=SimpleNamespace(synchronize=Mock(side_effect=synchronize)))
        with StageTimer("mps", torch_module=backend, clock=clock) as timer:
            clock.events.append("model")
            clock.advance(0.0625)
        self.assertEqual(timer.elapsed_sec, 0.0625)
        self.assertEqual(clock.events, ["mps", "clock", "model", "mps", "clock"])
        self.assertEqual(backend.mps.synchronize.call_args_list, [unittest.mock.call(), unittest.mock.call()])

    def test_device_object_is_forwarded_to_cuda_synchronization(self):
        device = SimpleNamespace(type="cuda", index=1)
        sync = Mock()
        with StageTimer(device, torch_module=SimpleNamespace(cuda=SimpleNamespace(synchronize=sync)),
                        clock=ManualClock()):
            pass
        self.assertEqual(sync.call_args_list, [unittest.mock.call(device), unittest.mock.call(device)])

    def test_separate_stages_exclude_startup_render_and_cache_work(self):
        clock = ManualClock()
        clock.advance(16)  # 모델 선택과 화면 준비
        with StageTimer("cpu", clock=clock) as inference:
            clock.advance(0.125)
        clock.advance(8)  # 결과 해석 및 UI 갱신
        with StageTimer("cpu", clock=clock) as gradcam:
            clock.advance(0.5)
        clock.advance(32)  # 컬러맵과 디스크 캐시 저장
        self.assertEqual(inference.elapsed_sec, 0.125)
        self.assertEqual(gradcam.elapsed_sec, 0.5)
        self.assertGreater(clock.now, inference.elapsed_sec + gradcam.elapsed_sec + 50)

    def test_successful_zero_is_distinct_from_unmeasured_and_failed(self):
        timer = StageTimer("cpu", clock=ManualClock())
        self.assertIsNone(timer.elapsed_sec)
        with timer:
            pass
        self.assertEqual(timer.elapsed_sec, 0.0)
        self.assertFalse(timer.failed)

    def test_body_failure_propagates_and_never_reports_zero_duration(self):
        clock = ManualClock()
        timer = StageTimer("cpu", clock=clock)
        error = RuntimeError("model failed")
        with self.assertRaises(RuntimeError) as raised:
            with timer:
                clock.advance(1)
                raise error
        self.assertIs(raised.exception, error)
        self.assertIsNone(timer.elapsed_sec)
        self.assertTrue(timer.failed)
        self.assertEqual(timer.status, "error")

    def test_gpu_completion_failure_is_not_a_successful_timing(self):
        sync = Mock(side_effect=[None, RuntimeError("device failed")])
        timer = StageTimer("cuda:0", clock=ManualClock(),
                           torch_module=SimpleNamespace(cuda=SimpleNamespace(synchronize=sync)))
        with self.assertRaisesRegex(RuntimeError, "device failed"):
            with timer:
                pass
        self.assertIsNone(timer.elapsed_sec)
        self.assertTrue(timer.failed)
        self.assertEqual(timer.status, "error")


class TimingSummaryTests(unittest.TestCase):
    def result(self, inference=None, gradcam=None, inference_status="not_run", gradcam_status="not_run"):
        return SimpleNamespace(inference_sec=inference, gradcam_sec=gradcam,
                               inference_status=inference_status, gradcam_status=gradcam_status,
                               elapsed_sec=9.0)

    def test_missing_failed_unsupported_and_true_zero_are_distinct(self):
        self.assertEqual(format_duration(None), "미실행")
        self.assertEqual(format_duration(None, "error"), "실패")
        self.assertEqual(format_duration(None, "unsupported"), "해당 없음")
        self.assertEqual(format_duration(0.0, "completed"), "0.0 ms")
        self.assertEqual(format_duration(0.125, "completed"), "125.0 ms")
        self.assertEqual(format_duration(1.25, "completed"), "1.25 초")

    def test_result_shows_both_stages_with_original_values(self):
        result = self.result(0.125, 0.5, "completed", "completed")
        text = format_result_timing(result)
        self.assertIn("추론", text)
        self.assertIn("125.0 ms", text)
        self.assertIn("Grad-CAM", text)
        self.assertIn("500.0 ms", text)
        self.assertNotIn("9.00 초", text)
        self.assertIn("9.00 초", format_result_timing(result, include_total=True))

    def test_inference_average_excludes_missing_failed_and_unsupported_stages(self):
        results = [self.result(0.0, inference_status="completed"),
                   self.result(0.1, inference_status="completed"),
                   self.result(inference_status="error"),
                   self.result(inference_status="unsupported"),
                   self.result()]
        summary = batch_stage_summary(results, "inference")
        self.assertIn("추론 평균 50.0 ms", summary)
        self.assertIn("완료 2/5장", summary)

    def test_gradcam_average_has_its_own_denominator_and_values(self):
        results = [self.result(10.0, 0.25, "completed", "completed"),
                   self.result(20.0, 0.5, "completed", "completed"),
                   self.result(30.0, inference_status="completed", gradcam_status="error")]
        summary = batch_stage_summary(results, "gradcam")
        self.assertIn("Grad-CAM 평균 375.0 ms", summary)
        self.assertIn("완료 2/3장", summary)

    def test_unavailable_batch_stages_never_report_zero_average(self):
        cases = [([self.result()], "미실행"),
                 ([self.result(gradcam_status="unsupported")], "해당 없음"),
                 ([self.result(gradcam_status="error")], "완료 없음")]
        for results, expected in cases:
            with self.subTest(expected=expected):
                summary = batch_stage_summary(results, "gradcam")
                self.assertIn(expected, summary)
                self.assertNotIn("0.0 ms", summary)
        self.assertIn("실패 1장", batch_stage_summary([self.result(gradcam_status="error")], "gradcam"))


class ResultTimingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ROOT / "gui/core/inference_types.py"
        node = next(node for node in ast.parse(source.read_text(encoding="utf-8")).body
                    if isinstance(node, ast.ClassDef) and node.name == "InferenceResult")
        cls.module = ModuleType("_timing_result_contract")
        cls.module.dataclass = dataclass
        sys.modules[cls.module.__name__] = cls.module
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                     str(source), "exec"), cls.module.__dict__)
        cls.Result = cls.module.InferenceResult

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop(cls.module.__name__, None)

    def test_legacy_results_have_no_fabricated_stage_durations(self):
        result = self.Result("sample.png", "ok", "classify", "normal")
        self.assertIsNone(result.inference_sec)
        self.assertIsNone(result.gradcam_sec)
        self.assertEqual(result.inference_status, "not_run")
        self.assertEqual(result.gradcam_status, "not_run")

    def test_total_update_preserves_stage_measurements_and_decision(self):
        result = self.Result("sample.png", "ok", "anomaly", "NG", score=0.8, threshold=0.5,
                             inference_sec=0.0, gradcam_sec=0.125,
                             inference_status="completed", gradcam_status="completed")
        completed = replace(result, elapsed_sec=3.5)
        self.assertEqual(completed.inference_sec, 0.0)
        self.assertEqual(completed.gradcam_sec, 0.125)
        self.assertEqual((completed.inference_status, completed.gradcam_status), ("completed", "completed"))
        self.assertEqual((completed.score, completed.threshold, completed.summary), (0.8, 0.5, "NG"))


@unittest.skipUnless(GUI_AVAILABLE, "Qt와 모델 실행 의존성 필요")
class CachedTimingDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from widgets.inference_widget import InferenceResult, InferenceWidget
        self.Result = InferenceResult
        self.widget = InferenceWidget()
        self.widget.show()
        self.application.processEvents()

    def tearDown(self):
        self.widget._preview_cache.clear()
        self.widget._heatmap_timer.stop()
        self.widget.hide()
        self.widget.deleteLater()
        self.application.processEvents()

    def test_cache_selection_restores_stage_times_without_running_any_engine(self):
        paths = ["first.png", "second.png"]
        saved = [self.Result(paths[0], "ok", "classify", "class-0", elapsed_sec=9.0,
                             inference_sec=0.125, gradcam_sec=0.5,
                             inference_status="completed", gradcam_status="completed"),
                 self.Result(paths[1], "ok", "classify", "class-1", elapsed_sec=12.0,
                             inference_sec=0.25, gradcam_sec=None,
                             inference_status="completed", gradcam_status="error")]
        self.widget._batch_images = paths[:]
        self.widget._clear_results()
        for result in saved:
            self.widget._current_preview_rgb = np.full((4, 6, 3), 90, dtype=np.uint8)
            self.widget._cache_inference_result(result)
        self.widget._run_single_inference = Mock(side_effect=AssertionError("캐시 선택 중 재추론"))
        self.widget._on_grid_cell_clicked(0)
        first_text = self.widget.result_card.time_label.text()
        self.assertIn("추론", first_text)
        self.assertIn("Grad-CAM", first_text)
        self.assertIn("125.0", first_text)
        self.assertIn("500.0", first_text)
        self.widget.result_review.select_path(self.widget._batch_images[1])
        second_text = self.widget.result_card.time_label.text()
        self.assertIn("250.0", second_text)
        self.assertIn("실패", second_text)
        self.assertNotIn("500.0", second_text)
        self.assertIsNone(self.widget._current_result.gradcam_sec)
        self.widget._on_grid_cell_clicked(0)
        self.assertEqual(self.widget.result_card.time_label.text(), first_text)
        self.assertIs(self.widget._current_result, saved[0])
        self.widget._run_single_inference.assert_not_called()

    def test_engine_stage_boundaries_exclude_cards_heatmap_rendering_and_disk_cache(self):
        import torch

        clock = ManualClock()
        calls = []

        class TinyClassifier(torch.nn.Module):
            task = "classify"
            in_channels = 3

            def forward(self, tensor):
                calls.append("inference")
                clock.advance(0.125)
                return torch.tensor([[0.0, 1.0]], device=tensor.device)

        def generate_activation(*args, **kwargs):
            calls.append("gradcam")
            clock.advance(0.5)
            return np.array([[0.0, 1.0], [0.25, 0.5]], dtype=np.float32)

        def with_cost(function, seconds):
            def wrapped(*args, **kwargs):
                clock.advance(seconds)
                return function(*args, **kwargs)
            return wrapped

        self.widget.model = TinyClassifier()
        self.widget._gradcam = SimpleNamespace(generate_activation=generate_activation)
        self.widget._input_size = (8, 8)
        self.widget._normalization = ([0.0] * 3, [1.0] * 3)
        self.widget.class_names = ["first", "second"]
        self.widget._set_heatmap_preview = with_cost(self.widget._set_heatmap_preview, 2.0)
        self.widget._cache_inference_result = with_cost(self.widget._cache_inference_result, 4.0)
        self.widget.result_card.show_classification_result = with_cost(
            self.widget.result_card.show_classification_result, 1.0)

        with tempfile.TemporaryDirectory() as temp:
            image_path = str(Path(temp) / "sample.png")
            Image.new("RGB", (8, 8), (90, 90, 90)).save(image_path)
            with patch("widgets.inference_widget.StageTimer",
                       side_effect=lambda device: StageTimer(device, clock=clock)), \
                    patch("widgets.inference_widget.time.perf_counter", clock):
                result = self.widget._run_single_inference(image_path)

        self.assertEqual(result.status, "ok")
        self.assertEqual(calls, ["inference", "gradcam"])
        self.assertEqual(result.inference_sec, 0.125)
        self.assertEqual(result.gradcam_sec, 0.5)
        self.assertEqual((result.inference_status, result.gradcam_status), ("completed", "completed"))
        self.assertEqual(result.elapsed_sec, 7.625)
        self.assertEqual(self.widget._inference_results[image_path].elapsed_sec, 7.625)
        self.assertEqual(clock.now, 7.625)


if __name__ == "__main__":
    unittest.main()
