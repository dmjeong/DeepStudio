"""Real persistence/chart methods with substitute controls; not a Qt render test."""
import ast
import copy
from dataclasses import asdict
import math
from numbers import Real
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "gui")]
from core.project import ProjectManager, RunRecord
from core.training_progress import record_epoch, remaining_seconds, run_description


def source_method(path, class_name, method_name, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace[method_name]


class ResultIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.original = ProjectManager.create_new("Review", "classify", self.temp.name, ["OK", "NG"])
        ProjectManager.save(self.original)
        self.snapshot = copy.deepcopy(self.original)
        self.snapshot.training.training_mode = "efficientnet_resume"
        self.snapshot.training.efficientnet_model = "efficientnet_b1"
        self.config = {"task": "classify", "training": asdict(self.snapshot.training),
                       "model": asdict(self.snapshot.model), "data": asdict(self.snapshot.data),
                       "runtime": {"name": "CPU"}, "job_id": "job-current", "checkpoint_sha256": "abc",
                       "effective_model_source": "actual.pt", "future_field": {"keep": True}}
        self.snapshot.runs.append(RunRecord(run_id="run-current", status="completed", config_snapshot=copy.deepcopy(self.config)))
        self.dialog = MagicMock()
        self.finish = source_method("gui/widgets/training_widget.py", "TrainingWidget", "_on_worker_finished",
            {"copy": copy, "ProjectManager": ProjectManager, "QMessageBox": self.dialog, "run_description": run_description})
        self.ui = SimpleNamespace(_run_project=self.original, _run_snapshot=self.snapshot, _initial_run_count=0,
            _run_config={"training": {"efficientnet_model": "efficientnet_b0"}}, _run_file_digest=None,
            _pending_result=None, _pending_error=None, _cancel_requested=False, worker=MagicMock(),
            _display_finished_run=MagicMock(), collect_config=MagicMock(),
            **{name: MagicMock() for name in ("start_btn", "stop_btn", "settings_panel", "eta_label",
               "status_label", "compare_widget", "progress_bar", "run_identity_label", "log_text")})

    def test_saved_execution_config_retains_actual_model_and_unknown_fields(self):
        self.finish(self.ui)
        loaded = ProjectManager.load(ProjectManager.get_active_filepath(self.original)).runs[-1]
        for key, value in self.config.items():
            self.assertEqual(loaded.config_snapshot[key], value)
        self.assertEqual(loaded.config_snapshot["requested_config"]["training"]["efficientnet_model"], "efficientnet_b0")
        self.assertIn("efficientnet_b1", self.ui.run_identity_label.setText.call_args.args[0])
        self.snapshot.runs[-1].config_snapshot["future_field"]["keep"] = False
        self.assertTrue(self.original.runs[-1].config_snapshot["future_field"]["keep"])

    def test_save_failure_does_not_report_success_and_unlocks(self):
        self.ui._pending_result = (.9, 1, "checkpoint.pt")
        with patch.object(ProjectManager, "save", side_effect=PermissionError("locked")):
            self.finish(self.ui)
        self.dialog.information.assert_not_called()
        self.dialog.critical.assert_called_once()
        self.ui.status_label.setText.assert_called_with("결과 저장 실패")
        self.ui.start_btn.setEnabled.assert_called_with(True)
        self.ui.settings_panel.setEnabled.assert_called_with(True)
        self.assertIsNone(self.ui.worker)

    def test_render_failure_still_saves_and_unlocks(self):
        self.ui._display_finished_run.side_effect = RuntimeError("broken canvas")
        self.finish(self.ui)
        self.assertEqual(len(ProjectManager.load(ProjectManager.get_active_filepath(self.original)).runs), 1)
        self.assertIsNone(self.ui._run_project)
        self.ui.start_btn.setEnabled.assert_called_with(True)
        self.ui.log_text.append.assert_called_once()

    def test_missing_result_cannot_be_reported_as_completed(self):
        self.snapshot.runs.clear()
        self.finish(self.ui)
        self.ui.status_label.setText.assert_called_with("학습 실패")
        self.dialog.information.assert_not_called()

    def test_resume_eta_uses_only_work_completed_after_resume(self):
        clock = MagicMock()
        clock.monotonic.side_effect = [100., 160., 700.]
        progress = source_method("gui/widgets/training_widget.py", "TrainingWidget", "_on_progress",
                                {"time": clock, "remaining_seconds": remaining_seconds})
        self.ui._progress_start_epoch = None
        progress(self.ui, 40, 50)
        progress(self.ui, 41, 50)
        self.ui.eta_label.setText.assert_called_with("ETA: 9m 0s")
        self.ui.progress_bar.setValue.assert_called_with(82)
        progress(self.ui, 50, 50)
        self.ui.progress_bar.setValue.assert_called_with(99)

    def test_unavailable_losses_stay_nan_at_the_qt_signal_boundary(self):
        event = source_method("gui/core/qt_training.py", "QtTrainingWorker", "_event", {})
        signal = MagicMock()
        worker = SimpleNamespace(signals=SimpleNamespace(epoch_finished=signal))
        event(worker, "epoch_finished", [11, .4, None, {"accuracy": .8}])
        args = signal.emit.call_args.args
        self.assertEqual(args[0:2], (11, .4))
        self.assertTrue(math.isnan(args[2]))
        self.assertEqual(args[3], {"accuracy": .8})


class ChartCoordinateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from matplotlib.figure import Figure
        class AggBase:
            def __init__(self, *args):
                self.figure = Figure()
                self.ax = self.figure.add_subplot(111)
            def _style_axes(self, *args):
                pass
            def _draw(self):
                pass
        tree = ast.parse((ROOT / "gui/widgets/training_results.py").read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if
                 (isinstance(n, ast.ClassDef) and n.name in {"LossChart", "MetricChart", "LRChart"}) or
                 (isinstance(n, ast.FunctionDef) and n.name == "_finite_metric") or
                 (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CHART_COLORS" for t in n.targets))]
        cls.charts = {"_BaseChart": AggBase, "HAS_MATPLOTLIB": True, "Real": Real, "math": math, "record_epoch": record_epoch}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "actual_chart_classes", "exec"), cls.charts)

    def test_resumed_epochs_duplicates_and_missing_metrics_use_actual_coordinates(self):
        loss, metric, lr = (self.charts[name]() for name in ("LossChart", "MetricChart", "LRChart"))
        for epoch in (11, 12, 12):
            loss.update_chart(epoch, .4, None)
            lr.update_lr(epoch, .001)
        metric.update_metrics(11, {"accuracy": .8})
        metric.update_metrics(12, {"accuracy": .9, "f1": .85})
        metric.update_metrics(13, {"accuracy": float("nan"), "f1": .9})
        self.assertEqual(loss.ax.lines[0].get_xdata().tolist(), [11, 12])
        self.assertEqual(lr.ax.lines[0].get_xdata().tolist(), [11, 12])
        self.assertEqual(metric.metrics_history["f1"], [None, .85, .9])
        self.assertEqual(metric.metrics_history["accuracy"], [.8, .9, None])
        for chart in (loss, metric, lr):
            chart.clear()
            self.assertEqual(chart.epochs, [])
            self.assertEqual(len(chart.ax.lines), 0)


if __name__ == "__main__":
    unittest.main()
