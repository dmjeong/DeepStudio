"""베스트 체크포인트의 에폭과 손실을 함께 표시하는 학습 카드 회귀 검증."""

import copy
import importlib.util
import math
from numbers import Real
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tests.test_training_contracts import source_method, source_objects


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gui"))
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.metrics import TASK_METRIC_NAMES


class Card:
    def __init__(self):
        self.value = "—"
        self.style = ""

    def setText(self, value):
        self.value = value

    def setStyleSheet(self, style):
        self.style = style

    def text(self):
        return self.value


def project_for(task="classify", mode="custom", anomaly_method="patchcore"):
    return SimpleNamespace(task=task, training=SimpleNamespace(
        training_mode=mode, anomaly_method=anomaly_method))


class BestLossCardContracts(unittest.TestCase):
    def setUp(self):
        namespace = source_objects("gui/widgets/training_results.py",
            {"_finite_metric", "_format_metric", "_metric_info_for"},
            {"math": math, "Real": Real, "copy": copy, "TASK_METRIC_NAMES": TASK_METRIC_NAMES})
        self.on_epoch = source_method("gui/widgets/training_widget.py", "TrainingWidget",
                                      "_on_epoch_finished", namespace)
        self.on_best = source_method("gui/widgets/training_widget.py", "TrainingWidget",
                                     "_on_best_epoch_updated", namespace)
        self.ui = SimpleNamespace(project=project_for(), _run_snapshot=None,
                                  loss_chart=Mock(), metric_chart=Mock())
        self.ui.metric_cards = {name: Card() for name in (
            "Best Accuracy", "Best Epoch", "Train Loss", "Val Loss")}
        render = source_method("gui/widgets/training_widget.py", "TrainingWidget", "_render_best_metrics", namespace)
        self.ui._render_best_metrics = lambda *args: render(self.ui, *args)
        self.ui.eval_widget = Mock()
        def rebuild(task, project=None, metrics=None):
            from core.best_metrics import metric_label
            self.ui._metric_card_keys = {key: (metric_label(key, task) if key in {"train_loss", "val_loss"}
                else f"Best {metric_label(key, task)}") for key in metrics}
            self.ui.metric_cards = {name: Card() for name in ["Best Epoch", *self.ui._metric_card_keys.values()]}
        self.ui._rebuild_metric_cards = rebuild

    def values(self):
        return {name: card.value for name, card in self.ui.metric_cards.items()}

    def test_intermediate_best_keeps_its_loss_pair_while_all_epochs_reach_chart(self):
        epochs = [(1, .8, .2, {"accuracy": .7}),
                  (2, .5, .4, {"accuracy": .9}),
                  (3, .1, .1, {"accuracy": .8})]
        for result in epochs:
            self.on_epoch(self.ui, *result)
            if result[0] in (1, 2):
                self.on_best(self.ui, *result)
        self.assertEqual(self.values(), {"Best Accuracy": "0.9000", "Best Epoch": "2",
                                        "Train Loss": "0.5000", "Val Loss": "0.4000"})
        self.assertEqual(self.ui.loss_chart.update_chart.call_args_list,
                         [unittest.mock.call(*row[:3]) for row in epochs])
        self.assertEqual(self.ui.metric_chart.update_metrics.call_count, 3)

    def test_backend_tie_selection_updates_both_losses_without_ui_reselection(self):
        self.on_best(self.ui, 3, .6, .5, {"accuracy": .9})
        self.on_epoch(self.ui, 4, .3, .4, {"accuracy": .9})
        self.assertEqual(self.values()["Best Epoch"], "3")
        self.on_best(self.ui, 4, .3, .4, {"accuracy": .9})
        self.assertEqual(self.values(), {"Best Accuracy": "0.9000", "Best Epoch": "4",
                                        "Train Loss": "0.3000", "Val Loss": "0.4000"})

    def test_backend_selection_is_not_rejected_by_rounded_displayed_metric(self):
        self.on_best(self.ui, 5, .4, .5, {"accuracy": .900049})
        self.on_best(self.ui, 6, .3, .6, {"accuracy": .900048})
        self.assertEqual(self.values(), {"Best Accuracy": "0.9000", "Best Epoch": "6",
                                        "Train Loss": "0.3000", "Val Loss": "0.6000"})

    def test_metric_identity_comes_from_running_snapshot(self):
        cases = [
            (project_for("segment", "custom"),
             "Best mIoU", {"mIoU": .45}, "0.4500"),
            (project_for("anomaly", anomaly_method="reconstruction"),
             "Best Reconstruction Loss", {"recon_loss": .015, "auroc": .99}, "0.0150"),
        ]
        for snapshot, card_name, metrics, expected in cases:
            with self.subTest(card_name=card_name):
                self.ui._run_snapshot = snapshot
                self.ui.metric_cards[card_name] = Card()
                self.on_best(self.ui, 7, .3, .2, metrics)
                self.assertEqual(self.ui.metric_cards[card_name].value, expected)
                self.assertNotIn("Best Accuracy", self.ui.metric_cards)

    def test_unavailable_best_values_replace_previous_numbers_and_highlight(self):
        self.on_best(self.ui, 2, .3, .4, {"accuracy": .8})
        self.assertEqual(self.ui.metric_cards["Best Accuracy"].value, "0.8000")
        self.on_best(self.ui, 3, float("nan"), None, {})
        self.assertEqual(self.values(), {"Best Epoch": "3",
                                        "Train Loss": "N/A", "Val Loss": "N/A"})
        self.assertNotIn("Best Accuracy", self.ui.metric_cards)


GUI_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in (
    "PySide6", "torch", "torchvision", "cv2", "matplotlib", "unsupported"))


@unittest.skipUnless(GUI_AVAILABLE, "Qt와 모델 실행 의존성 필요")
class BestLossCardQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        from PySide6.QtCore import QObject, Signal
        from core.project import ProjectData
        from core.signals import TrainingSignals
        from widgets.training_widget import TrainingWidget

        class ControlledWorker(QObject):
            finished = Signal()

            def __init__(self, project):
                super().__init__()
                self.project = project
                self.signals = TrainingSignals()
                self.started = False
                self.stopped = False

            def start(self):
                self.started = True

            def isRunning(self):
                return self.started and not self.stopped

            def stop(self):
                self.stopped = True

        self.worker_patch = patch("widgets.training_widget.TrainWorker", ControlledWorker)
        self.worker_patch.start()
        self.addCleanup(self.worker_patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.widget = TrainingWidget()
        project = ProjectData()
        project.project_dir = self.temp.name
        project.data.root = self.temp.name
        project.training.training_mode = "custom"
        project.training.device = "cpu"
        self.widget.set_project(project)

    def tearDown(self):
        self.widget.hide()
        self.widget.deleteLater()
        self.application.processEvents()

    def loss_values(self):
        return tuple(self.widget.metric_cards[name].text() for name in (
            "Best Epoch", "Train Loss", "Val Loss"))

    def start_with_intermediate_best(self):
        self.widget._start_training()
        worker = self.widget.worker
        self.assertIsNotNone(worker)
        self.assertTrue(worker.started)
        worker.signals.epoch_finished.emit(1, .6, .5, {"accuracy": .9})
        worker.signals.best_epoch_updated.emit(1, .6, .5, {"accuracy": .9})
        worker.signals.epoch_finished.emit(2, .1, .2, {"accuracy": .8})
        self.application.processEvents()
        self.assertEqual(self.loss_values(), ("1", "0.6000", "0.5000"))
        return worker

    def test_start_connects_real_signals_and_successful_next_run_resets_cards(self):
        from PySide6.QtWidgets import QLabel
        worker = self.start_with_intermediate_best()
        for name in ("Train Loss", "Val Loss"):
            frame = self.widget.metric_cards[name].parentWidget()
            self.assertEqual(frame.findChild(QLabel, "metric_label").text(), f"{name} (Best)")
            self.assertIn("베스트", frame.toolTip())
        with patch("widgets.training_widget.ProjectManager.save"), \
                patch("widgets.training_widget.QMessageBox.information"):
            worker.signals.training_finished.emit(.9, 1, "best.pt")
            worker.finished.emit()
        self.assertEqual(self.loss_values(), ("1", "0.6000", "0.5000"))
        self.widget._start_training()
        self.assertEqual(self.loss_values(), ("—", "—", "—"))
        self.assertTrue(all(not card.styleSheet() for card in self.widget.metric_cards.values()))

    def test_cancelled_run_keeps_last_saved_best_and_missing_loss_is_na(self):
        worker = self.start_with_intermediate_best()
        worker.signals.best_epoch_updated.emit(3, float("nan"), float("nan"), {})
        self.assertEqual(self.loss_values(), ("3", "N/A", "N/A"))
        self.widget._stop_training()
        self.assertTrue(worker.stopped)
        with patch("widgets.training_widget.ProjectManager.save"):
            worker.finished.emit()
        self.assertEqual(self.loss_values(), ("3", "N/A", "N/A"))
        self.assertEqual(self.widget.status_label.text(), "학습 중단")


if __name__ == "__main__":
    unittest.main()
