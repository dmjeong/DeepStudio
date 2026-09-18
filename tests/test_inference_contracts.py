"""실제 GUI 메서드의 결과 전달 계약을 Qt/모델 대체 객체로 검사한다.

전체 GUI/학습 통합 검증은 별도 환경이 필요하다. 여기서는 원본 AST 메서드를
실행해 실패 결과, 후보 모델 교체, 배치 선택, 시간 표시의 연결을 검증한다.
"""
import ast
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
_WIDGET = _ROOT / "gui/widgets/inference_widget.py"


def extracted(path, class_name, names, namespace):
    if class_name is None and path == _WIDGET:
        path = _ROOT / "gui/core/inference_types.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = tree.body if class_name is None else next(
        item.body for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name)
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []
                node.returns = None
                for arg in node.args.args:
                    arg.annotation = None
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                         str(path), "exec"), namespace)


class Label:
    def __init__(self):
        self.value = ""
        self.enabled = False

    def setText(self, text):
        self.value = text

    def set_result(self, text="", color="#5590F0"):
        self.result_text = text
        self.result_color = color

    def text(self):
        return self.value

    def clear(self):
        self.value = ""

    def setStyleSheet(self, text):
        pass

    def setToolTip(self, text):
        self.tooltip = text

    def setEnabled(self, enabled):
        self.enabled = enabled

    def hide(self):
        pass

    def show(self):
        pass


class InferenceContractTests(unittest.TestCase):
    def setUp(self):
        module = ModuleType("_inference_contract_source")
        sys.modules[module.__name__] = module
        self.ns = module.__dict__
        self.ns.update(np=np, dataclass=dataclass, time=time, os=__import__("os"))
        timing_spec = importlib.util.spec_from_file_location(
            "_contract_timing", _ROOT / "gui/core/inference_timing.py")
        timing = importlib.util.module_from_spec(timing_spec)
        timing_spec.loader.exec_module(timing)
        spec = importlib.util.spec_from_file_location("_contract_review", _ROOT / "gui/core/inference_review.py")
        review = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(review)
        self.ns["effective_result"] = review.effective_result
        self.ns.update(format_result_timing=timing.format_result_timing,
                       format_runtime_stages=timing.format_runtime_stages,
                       batch_stage_summary=timing.batch_stage_summary)
        extracted(_WIDGET, None, {"InferenceResult", "make_anomaly_result"}, self.ns)
        self.Result = self.ns["InferenceResult"]

    def tearDown(self):
        sys.modules.pop("_inference_contract_source", None)

    def methods(self, *names):
        extracted(_WIDGET, "InferenceWidget", set(names), self.ns)
        return type("WidgetMethods", (), {name: self.ns[name] for name in names})

    def test_desktop_worker_receives_gradcam_checkbox_in_both_states(self):
        created = []

        class Engine:
            STATE_FIELDS = ()

            def __init__(self, state, *, gradcam=True, input_region=None):
                self.gradcam_enabled = gradcam
                created.append(self)

        jobs = ModuleType("core.desktop_jobs")
        jobs.desktop_manager = lambda: SimpleNamespace(require_idle=lambda: None)
        worker_module = ModuleType("core.inference_worker")
        worker_module.InferenceWorker = Mock(return_value=Mock())
        self.ns.update(InferenceEngine=Engine, QPushButton=object)
        widget_type = self.methods("_run_inference")
        for checked in (False, True):
            with self.subTest(checked=checked):
                widget = widget_type()
                widget._inference_worker, widget._process_model = None, False
                widget.model, widget._patchcore_model = object(), None
                widget._batch_images = ["gray.png"]
                widget._input_region_request = lambda: {"mode": "model"}
                widget._clear_results = lambda: None
                widget.gradcam_checkbox = SimpleNamespace(isChecked=lambda: checked)
                widget._preview_cache = object()
                widget.findChildren = lambda _: []
                for name in ("infer_device_combo", "ckpt_edit", "crop_mode_combo", "crop_json_edit",
                             "cancel_infer_btn", "inference_progress", "_on_async_result",
                             "_on_inference_progress", "_on_inference_failed", "_on_inference_finished"):
                    setattr(widget, name, Mock())
                with patch.dict(sys.modules, {"core.desktop_jobs": jobs, "core.inference_worker": worker_module}):
                    widget._run_inference()
                self.assertIs(created[-1].gradcam_enabled, checked)
                self.assertIs(worker_module.InferenceWorker.call_args.args[0], created[-1])

    def test_single_image_refreshes_gradcam_setting_before_each_run(self):
        widget = self.methods("_run_single_inference")()
        widget.model, widget._patchcore_model = object(), None
        widget._gradcam = object()
        widget._inference_results = {}
        for name in ("image_label", "gradcam_info", "result_card", "_clear_heatmap_preview",
                     "_cache_inference_result", "_render_result_card", "_present_result", "_show_image"):
            setattr(widget, name, Mock())
        observed = []

        def infer(path):
            observed.append(getattr(widget, "gradcam_enabled", None))
            return self.Result(path, "ok", "classify", "OK")

        widget._run_custom_inference = infer
        for checked in (False, True, False):
            widget.gradcam_checkbox = SimpleNamespace(isChecked=lambda: checked)
            result = widget._run_single_inference("gray.png")
            self.assertEqual(result.status, "ok")
            self.assertIs(observed[-1], checked)

    def test_async_result_preserves_selected_pending_image(self):
        widget = self.methods("_on_async_result")()
        widget._inference_results = {}
        widget._batch_images = ["first", "pending"]
        widget._batch_results, widget._batch_times = {}, {}
        widget._grid_cells = []
        widget.result_review = SimpleNamespace(threshold=None, update_result=lambda _: None)
        widget._current_image = "pending"
        displayed = []
        widget._display_cached_result = displayed.append
        widget._on_async_result(self.Result("first", "ok", "classify", "OK"))
        self.assertEqual(displayed, [])
        widget._on_async_result(self.Result("pending", "ok", "classify", "NG"))
        self.assertEqual(displayed, ["pending"])

    def test_uncalibrated_and_nonfinite_anomalies_never_pass(self):
        result = self.ns["make_anomaly_result"]("sample", 0.01, None)
        self.assertEqual(result.status, "uncalibrated")
        self.assertNotIn("OK", result.summary)
        self.assertIn("NG", self.ns["make_anomaly_result"]("sample", 0.1, 0.1).summary)
        for invalid in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                self.ns["make_anomaly_result"]("sample", invalid, 0.1)

    def test_failed_batch_image_is_error_and_preserves_timing(self):
        cls = self.methods("_run_single_inference", "_run_batch_inference", "_present_result")
        widget = cls()
        widget.image_label = Label()
        widget.image_path_label = Label()
        widget.selected_timing_label = Label()
        widget._clear_heatmap_preview = lambda: None
        widget._cache_inference_result = lambda _: None
        widget._render_result_card = lambda _: None
        widget._grid_cells = []
        widget.model, widget._patchcore_model = object(), None
        errors = []
        widget.result_card = SimpleNamespace(show_error=errors.append, set_inference_times=lambda _: None)
        widget.gradcam_info = Label()
        widget.gradcam_checkbox = SimpleNamespace(isChecked=lambda: True)
        widget._show_image = lambda _: None
        widget._refresh_grid = lambda: None
        widget._batch_results, widget._batch_times = {}, {}
        widget._inference_results = {}
        items = []
        widget.project = None
        widget.result_review = SimpleNamespace(threshold=None, set_context=lambda *_: items.clear(),
                                               update_result=items.append)
        widget.batch_time_label = Label()
        with tempfile.TemporaryDirectory() as temp:
            good, missing = str(Path(temp) / "good.png"), str(Path(temp) / "missing.png")
            Image.new("RGB", (2, 2)).save(good)
            def backend(filename):
                with Image.open(filename):
                    return self.Result(filename, "ok", "anomaly", "OK 0.0100")
            widget._run_custom_inference = backend
            widget._batch_images = [good, missing]
            widget._run_batch_inference()
        self.assertEqual(widget._batch_results[0].status, "ok")
        failed = widget._batch_results[1]
        self.assertEqual(failed.status, "error")
        self.assertEqual(failed.image_path, missing)
        self.assertEqual(failed.summary, "ERROR")
        self.assertEqual(items[1].summary, "ERROR")
        self.assertEqual(items[1].elapsed_sec, failed.elapsed_sec)
        self.assertEqual(items[1].inference_status, failed.inference_status)
        self.assertIn("오류 1장", widget.batch_time_label.text())
        self.assertGreaterEqual(failed.elapsed_sec, 0)
        self.assertTrue(errors)
        self.assertEqual(widget.image_label.result_text, "ERROR")
        self.assertEqual(widget._current_image, missing)

    def test_unreadable_image_clears_previous_preview(self):
        self.ns["QPixmap"] = lambda _: SimpleNamespace(isNull=lambda: True)
        widget = self.methods("_show_image")()
        widget.image_label = Label()
        widget._current_result = None
        widget.image_label.setText("previous image")
        widget.image_path_label = Label()
        widget._show_image("missing.png")
        self.assertEqual(widget.image_label.text(), "이미지 표시 불가")
        self.assertEqual(widget.image_path_label.text(), "missing.png")

    def test_patchcore_classmethod_return_and_selected_device_are_used(self):
        candidate = SimpleNamespace(input_size=320, anomaly_threshold=4.2, normalized_threshold=.5,
                                    get_info=lambda: {"memory_bank_size": 123})
        calls, activated = [], []
        def load(path, device):
            calls.append((path, device))
            return candidate
        self.ns["PatchCore"] = SimpleNamespace(load=load)
        widget = self.methods("_load_patchcore_model")()
        widget._device_manager = SimpleNamespace(get_device=lambda x: x, get_device_label=str)
        widget.infer_device_combo = SimpleNamespace(currentData=lambda: "cuda:1")
        widget._activate_model = lambda **kwargs: activated.append(kwargs)
        widget._load_patchcore_model({}, "memory.pt")
        self.assertEqual(calls, [("memory.pt", "cuda:1")])
        self.assertIs(activated[0]["patchcore"], candidate)
        self.assertEqual(activated[0]["input_size"], (320, 320))
        self.assertEqual(activated[0]["threshold"], .5)
        self.assertTrue(activated[0]["score_normalized"])

    def test_failed_custom_candidate_does_not_replace_active_model(self):
        helper = ModuleType("export_onnx")
        helper.resolve_checkpoint_spec = lambda _: {"task": "classify"}
        def fail_load(*args):
            raise ValueError("incompatible weights")
        helper.load_custom_model = fail_load
        self.ns["torch"] = SimpleNamespace(load=lambda *a, **k: {}, nn=SimpleNamespace(Module=type))
        self.ns["QMessageBox"] = SimpleNamespace(critical=lambda *a: None)
        widget = self.methods("_load_model")()
        old = object()
        widget.model = old
        widget._active_checkpoint = "old.pt"
        widget._device_manager = SimpleNamespace(get_device=lambda x: x)
        widget.infer_device_combo = SimpleNamespace(currentData=lambda: "cpu")
        with tempfile.NamedTemporaryFile() as checkpoint, patch.dict(sys.modules, {"export_onnx": helper}):
            self.assertFalse(widget._load_model(checkpoint.name))
        self.assertIs(widget.model, old)
        self.assertEqual(widget._active_checkpoint, "old.pt")

    def test_single_image_selection_clears_old_batch(self):
        self.ns["QFileDialog"] = SimpleNamespace(getOpenFileName=lambda *a: ("new.png", ""))
        widget = self.methods("_select_image")()
        widget._batch_images = ["old.png"]
        widget._clear_results = lambda: None
        widget._show_image = lambda _: None
        widget._switch_view_mode = lambda _: None
        widget._select_image()
        self.assertEqual(widget._batch_images, [])
        self.assertEqual(widget._current_image, "new.png")

    def test_anomaly_cam_uses_reconstruction_error(self):
        extracted(_ROOT / "gui/core/gradcam.py", "GradCAM", {"_compute_target_score"}, self.ns)
        class Tensor:
            def __init__(self, values):
                self.values = np.asarray(values)
                self.shape = self.values.shape
                self.device = "cpu"
            def detach(self):
                return self
            def to(self, _):
                return self
            def __sub__(self, other):
                return Tensor(self.values - other.values)
            def pow(self, exponent):
                return Tensor(self.values ** exponent)
            def mean(self):
                return self.values.mean()
        self.ns["torch"] = SimpleNamespace(is_tensor=lambda value: isinstance(value, Tensor))
        obj = SimpleNamespace(task="anomaly")
        score = self.ns["_compute_target_score"](obj, Tensor([1.0]), None, Tensor([1.0]))
        self.assertEqual(score, 0)
        with self.assertRaises(ValueError):
            self.ns["_compute_target_score"](obj, Tensor([1.0]), None)

    def test_patchcore_entry_cancellation_stops_before_any_image_or_bank_save(self):
        source = _ROOT / "python/patchcore.py"
        extracted(source, None, {"PatchCoreCancelled", "_check_cancel"}, self.ns)
        extracted(source, "PatchCore", {"fit"}, self.ns)
        stopped, processed = [True], []
        feature = SimpleNamespace(cpu=lambda: object())
        image = SimpleNamespace(to=lambda _: image)
        def features(batch):
            processed.append(batch)
            return feature
        obj = SimpleNamespace(backbone=SimpleNamespace(eval=lambda: None), device="cpu",
                              _extract_features=features, memory_bank=None)
        def progress(*args):
            stopped[0] = True
        with self.assertRaises(self.ns["PatchCoreCancelled"]):
            self.ns["fit"](obj, [image, image], progress_callback=progress,
                           cancel_callback=lambda: stopped[0])
        self.assertEqual(len(processed), 0)
        self.assertIsNone(obj.memory_bank)

    def test_metrics_do_not_accept_reconstruction_images_as_labels(self):
        spec = importlib.util.spec_from_file_location("_contract_metrics", _ROOT / "gui/core/metrics.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaises(ValueError):
            module.AnomalyMetrics().update(np.array([1.0, 2.0]), np.zeros((2, 3, 4, 4)))


if __name__ == "__main__":
    unittest.main()
