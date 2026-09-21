"""Training page transitions using real Qt controls, without model downloads."""
import copy
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox

from core.project import ProjectManager, RunRecord
from widgets.training_widget import TrainingWidget


@pytest.fixture
def page(tmp_project):
    app = QApplication.instance() or QApplication([])
    widget = TrainingWidget()
    widget.set_project(tmp_project)
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


def change_mode(page, mode):
    # Invoke the slot explicitly so Qt cannot swallow its exception in a signal.
    page.mode_combo.blockSignals(True)
    page.mode_combo.setCurrentIndex(page.mode_combo.findData(mode))
    page.mode_combo.blockSignals(False)
    page._on_mode_changed(page.mode_combo.currentIndex())


@pytest.mark.parametrize("task,source,metric,target", [
    ("classify", "efficientnet_finetune", "val_loss", "efficientnet_scratch"),
    ("classify", "efficientnet_resume", "val_loss", "efficientnet_scratch"),
])
def test_model_switch_preserves_supported_val_loss_selection(page, task, source, metric, target):
    project = page.project
    project.task = task
    project.training.training_mode = source
    project.training.selection_metric = metric
    page.set_project(project)
    assert page.selection_combo.currentData() == metric
    change_mode(page, target)
    assert page.selection_combo.currentData() == metric
    assert page.selection_info.text()
    assert page.epochs_spin.isEnabled() == (target != "efficientnet_resume")


def test_project_switch_clears_previous_weights_and_monitor(page):
    page.resume_edit.setText("previous-project/best.pt")
    page.progress_bar.setValue(100)
    page.status_label.setText("학습 완료")
    page.eta_label.setText("ETA: 5m 0s")
    page.best_selection_label.setText("Best epoch 5")
    page.log_text.append("previous project result")
    project = copy.deepcopy(page.project)
    project.training.training_mode = "efficientnet_finetune"
    project.model.pretrained_weights = ""
    page.set_project(project)
    change_mode(page, "efficientnet_transfer")
    assert page.resume_edit.text() == ""
    assert page.progress_bar.value() == 0
    assert page.status_label.text() == "대기 중"
    assert page.best_selection_label.text() == "Best 선정 결과: 대기"
    assert page.log_text.toPlainText() == ""


@pytest.mark.parametrize("task,model_id", [
    ("classify", "resnet18"), ("classify", "resnet50"), ("classify", "convnext_v1_tiny"),
    ("segment", "deeplabv3plus_resnet34"), ("segment", "unet_resnet18"),
])
def test_builtin_model_picker_uses_its_own_weights_and_roundtrips(page, task, model_id):
    project = copy.deepcopy(page.project)
    project.task = task
    project.model.model_id = model_id
    project.training.training_mode = "builtin_finetune"
    page.set_project(project)
    assert page.mode_combo.currentData() == "builtin_finetune"
    assert all("EfficientNet" not in page.mode_combo.itemText(i) for i in range(page.mode_combo.count()))
    assert page.efficientnet_frame.isHidden()
    page.collect_config()
    assert project.model.model_id == model_id
    assert project.model.pretrained_weights == ""
    change_mode(page, "builtin_transfer")
    page.resume_edit.setText("C:/weights/local.pth")
    page.collect_config()
    assert project.model.pretrained_weights == "C:/weights/local.pth"
    page.set_project(copy.deepcopy(project))
    assert page.mode_combo.currentData() == "builtin_transfer"
    assert page.resume_edit.text() == "C:/weights/local.pth"


def test_catalog_and_efficientnet_variant_stay_in_sync(page):
    page.model_id_combo.setCurrentIndex(page.model_id_combo.findData("efficientnet_b1"))
    page.collect_config()
    assert page.project.model.model_id == "efficientnet_b1"
    assert page.project.training.efficientnet_model == "efficientnet_b1"
    assert page.input_size_spin.value() == 240
    page.efficientnet_model_combo.setCurrentIndex(page.efficientnet_model_combo.findData("efficientnet_b0"))
    page.collect_config()
    assert page.project.model.model_id == "efficientnet_b0"
    assert page.project.training.efficientnet_model == "efficientnet_b0"
    page.resume_edit.setText("previous-efficientnet.pt")
    page.model_id_combo.setCurrentIndex(page.model_id_combo.findData("resnet18"))
    assert page.mode_combo.currentData() == "builtin_finetune"
    assert page.resume_edit.text() == ""


def test_builtin_run_description_uses_the_recorded_model_id():
    from core.training_progress import run_description
    run = RunRecord(run_id="resnet-run", config_snapshot={"engine": "builtin", "model_id": "resnet50"})
    assert "모델: resnet50" in run_description(run)


def test_all_native_training_events_show_epoch_timing_without_polluting_metrics_chart(page):
    page._rebuild_metric_cards("classify")
    page._on_epoch_finished(1, .4, .3, {
        "accuracy": .8, "epoch_time_sec": 12.9, "elapsed_time_sec": 34.2,
    })
    assert page.metric_cards["최근 에폭 시간"].text() == "00:00:12"
    assert page.metric_cards["누적 시간"].text() == "00:00:34"
    assert "epoch_time_sec" not in page.metric_chart.metrics_history
    assert "elapsed_time_sec" not in page.metric_chart.metrics_history


@pytest.mark.parametrize("task,metric,values", [
    ("classify", "val_loss", {"accuracy": .8, "top5_accuracy": .95}),
    ("detect", "mAP_50_95", {"mAP_50": .7, "precision": .8}),
    ("segment", "prompt_dice", {"prompt_iou": .65}),
    ("segment", "mIoU", {"dice_score": .75}),
    ("anomaly", "auroc", {"f1": .85, "recall": .9}),
])
def test_dashboard_and_summary_render_all_saved_best_metrics_without_full_eval(page, task, metric, values):
    from core.best_metrics import best_epoch_metrics, metric_label
    snapshot = copy.deepcopy(page.project)
    snapshot.task = task
    snapshot.training.selection_metric = "engine_default"
    history = {"epoch": [11, 12], "train_loss": [.4, .1], "val_loss": [.3, .05],
               **{key: [value, .01] for key, value in values.items()}}
    run = RunRecord(best_epoch=11, best_metric_name=metric, best_metric=.75, metrics_history=history)
    page._display_finished_run(snapshot, run)
    expected = best_epoch_metrics(run)
    table = page.eval_widget.summary_table
    summary = {table.item(row, 0).text(): table.item(row, 1).text() for row in range(table.rowCount())}
    assert summary == {metric_label(key, task): f"{value:.4f}" for key, value in expected.items()}
    assert set(page._metric_card_keys) == set(expected)
    for key, name in page._metric_card_keys.items():
        assert page.metric_cards[name].text() == summary[metric_label(key, task)]
    page._on_epoch_finished(13, .001, .001, {metric: .01})
    assert page.metric_cards["Best Epoch"].text() == "11"
    assert page.metric_cards["Train Loss"].text() == "0.4000"


def test_live_best_event_updates_summary_and_replaces_previous_model_metrics(page):
    page._on_best_epoch_updated(2, .4, .3, {"accuracy": .8, "recall_macro": .7})
    old_card = page.metric_cards["Best Accuracy"].parentWidget()
    page._on_best_epoch_updated(3, .2, .25, {"prompt_dice": .9, "prompt_iou": .8})
    assert old_card.isHidden()
    assert "accuracy" not in page._metric_card_keys
    assert page.metric_cards["Best Prompt Dice"].text() == "0.9000"
    assert page.eval_widget.summary_table.rowCount() == 4


@pytest.mark.parametrize("task,model_id", [("segment", "sam2_hiera_tiny")])
def test_sam2_shipped_model_uses_packaged_pretrained_asset_without_a_docker_pack(page, task, model_id):
    project = copy.deepcopy(page.project)
    project.task = task
    project.model.model_id = model_id
    project.training.training_mode = "sam2_finetune"
    page.set_project(project)
    assert {page.mode_combo.itemData(index) for index in range(page.mode_combo.count())} == {
        "sam2_finetune", "sam2_transfer",
    }
    assert "prompt mask 미세조정" in page.mode_desc.text()
    assert "모델 팩" not in page.mode_desc.text()
    assert "EfficientNet" not in page.mode_combo.currentText()


@pytest.mark.parametrize("task,model_id", [("detect", "re_detr_v4_small"),
                                         ("classify", "libreyolo_classify_mobilenetv4_small")])
def test_upstream_shipped_models_offer_native_initialization_modes(page, task, model_id):
    project = copy.deepcopy(page.project)
    project.task = task
    project.model.model_id = model_id
    project.training.training_mode = "upstream_scratch"
    page.set_project(project)
    choices = {page.mode_combo.itemData(index) for index in range(page.mode_combo.count())}
    assert choices == {"upstream_finetune", "upstream_transfer", "upstream_resume", "upstream_scratch"}
    assert "Custom CSP" not in page.mode_desc.text()


def test_confusion_matrix_clear_removes_previous_color_scale(page):
    import numpy as np
    chart = page.cm_chart
    for matrix in ([[7, 1], [2, 6]], [[5, 3], [1, 7]]):
        chart.update_matrix(np.array(matrix), ["OK", "NG"])
        assert len(chart.figure.axes) == 2
        np.testing.assert_array_equal(chart.ax.images[0].get_array(), matrix)
        chart.clear()
        assert len(chart.figure.axes) == 1
        assert not chart.ax.images
        assert not chart.ax.texts


@pytest.fixture
def prepared_page(page, monkeypatch, tmp_path):
    from core import job_manager
    from webapp import storage
    monkeypatch.setattr(job_manager, "desktop_manager",
                        lambda: SimpleNamespace(require_idle=lambda: None))
    for name in ("critical", "information", "warning"):
        monkeypatch.setattr(QMessageBox, name, MagicMock())
    page.project.data.root = str(tmp_path)
    change_mode(page, "efficientnet_scratch")
    monkeypatch.setattr(ProjectManager, "save", lambda *args, **kwargs: "project.dvproj")
    monkeypatch.setattr(storage, "digest", lambda path: "unchanged")
    return page


@pytest.mark.parametrize("failure", ["save", "digest"])
def test_start_preparation_failure_preserves_results_and_buttons(prepared_page, monkeypatch, failure):
    from webapp import storage
    page = prepared_page
    page.log_text.append("previous result")
    page.progress_bar.setValue(100)
    def fail(*args, **kwargs):
        raise PermissionError("project storage unavailable")
    monkeypatch.setattr(ProjectManager if failure == "save" else storage, failure, fail)
    page._start_training()
    assert page.start_btn.isEnabled()
    assert not page.stop_btn.isEnabled()
    assert page.settings_panel.isEnabled()
    assert page._run_project is None
    assert page.log_text.toPlainText() == "previous result"
    assert page.progress_bar.value() == 100
    QMessageBox.critical.assert_called_once()


@pytest.mark.parametrize("failure", ["constructor", "start", None])
def test_worker_lifecycle_locks_settings_and_recovers(prepared_page, monkeypatch, failure):
    from widgets import training_widget
    page = prepared_page
    worker = MagicMock()
    if failure == "constructor":
        factory = MagicMock(side_effect=RuntimeError("cannot construct worker"))
    else:
        factory = MagicMock(return_value=worker)
        if failure == "start":
            worker.start.side_effect = RuntimeError("cannot start worker")
    monkeypatch.setattr(training_widget, "TrainWorker", factory)
    page._start_training()
    if failure is None:
        assert not page.settings_panel.isEnabled()
        assert not page.start_btn.isEnabled()
        assert page.stop_btn.isEnabled()
        page._stop_training()
        worker.stop.assert_called_once()
        assert not page.stop_btn.isEnabled()
        page._on_worker_finished()
        assert page.status_label.text() == "학습 중단"
    else:
        assert page.status_label.text() == "학습 실패"
        QMessageBox.critical.assert_called_once()
    assert page.settings_panel.isEnabled()
    assert page.start_btn.isEnabled()
    assert not page.stop_btn.isEnabled()
    assert page._run_project is None
    assert page.worker is None


def test_completed_resume_displays_worker_config_and_class_order(prepared_page, monkeypatch):
    from dataclasses import asdict
    from widgets import training_widget
    page = prepared_page
    monkeypatch.setattr(training_widget, "TrainWorker", MagicMock(return_value=MagicMock()))
    page._start_training()
    snapshot = page._run_snapshot
    snapshot.training.training_mode = "efficientnet_resume"
    snapshot.training.efficientnet_model = "efficientnet_b1"
    snapshot.training.selection_metric = "accuracy"
    snapshot.data.class_names = ["NG", "OK"]
    snapshot.data.num_classes = 2
    record = RunRecord(run_id="resumed-b1", status="completed", best_epoch=12, best_metric=.8,
        best_metric_name="accuracy", metrics_history={"epoch": [11, 12], "train_loss": [.5, .4],
        "val_loss": [.6, .5], "accuracy": [.7, .8],
        "epoch_time_sec": [11.2, 12.9], "elapsed_time_sec": [23.1, 36.0]}, lr_history=[.001, .0009],
        eval_results={"task": "classify", "accuracy": .8, "confusion_matrix": [[4, 1], [1, 4]]},
        config_snapshot={"training": asdict(snapshot.training), "runtime": {"name": "CPU"},
                         "checkpoint_sha256": "abc", "job_id": "job-b1"})
    snapshot.runs.append(record)
    page._on_training_finished(.8, 12, "b1.pt")
    page._on_worker_finished()
    assert page.project.runs[-1].config_snapshot["training"]["efficientnet_model"] == "efficientnet_b1"
    assert "efficientnet_b1" in page.run_identity_label.text()
    assert page.loss_chart.epochs == [11, 12]
    assert page.metric_cards["Best Epoch"].text() == "12"
    assert page.metric_cards["최근 에폭 시간"].text() == "00:00:12"
    assert page.metric_cards["누적 시간"].text() == "00:00:36"
    assert [label.get_text() for label in page.cm_chart.ax.get_xticklabels()] == ["NG", "OK"]
    assert page.progress_bar.value() == 100
    assert page.settings_panel.isEnabled()


def test_installed_container_pack_actions_start_dvw1_job(prepared_page, monkeypatch, tmp_path):
    """A Settings-added Docker model exposes the same pack lifecycle as the web API."""
    from core.desktop_jobs import DesktopJob
    from core.project import ProjectManager

    page = prepared_page
    project = copy.deepcopy(page.project)
    project.task = "detect"
    project.training.training_mode = "custom"
    project.model.model_id = "vendor.external-detector"
    data_root = tmp_path / "detect-data"
    pack_root = tmp_path / "re-detr-pack"
    pack_root.mkdir()
    project.data.root = str(data_root)
    project.model.pack_path = str(pack_root)
    page.set_project(project)
    # The registry normally supplies this path after a real .dvmodel install;
    # the fixture supplies the already-validated directory directly.
    page.project.model.pack_path = str(pack_root)
    external = SimpleNamespace(model_id="vendor.external-detector", display_name="External Detector")
    monkeypatch.setattr(page, "_selected_container_spec", lambda: (external, pack_root.resolve()))
    page._update_pack_controls()
    assert page.model_pack_train_button.isHidden()
    data_root.mkdir()
    page._update_pack_controls()

    assert not page.model_pack_train_button.isHidden()
    assert not page.model_pack_infer_button.isHidden()
    assert not page.model_pack_export_button.isHidden()

    start = MagicMock()
    monkeypatch.setattr(DesktopJob, "start", start)
    monkeypatch.setattr(ProjectManager, "get_active_filepath",
                        lambda current: str(tmp_path / "detect.dvproj"))
    page._start_model_pack_operation("train")

    assert start.call_count == 1
    assert page._pack_job is not None
    assert page._pack_job.kind == "pack_train"
    assert page._pack_job.payload["pack_dir"] == str(pack_root.resolve())
    assert page._pack_job.payload["data_dir"] == str(data_root.resolve())
    assert page._pack_job.payload["request"]["model"]["model_id"] == "vendor.external-detector"
    assert not page.start_btn.isEnabled()
    assert not page.settings_panel.isEnabled()

    page._on_pack_completed({"status": "completed", "output": {"result": "ok"}})
    assert page._pack_job is None
    assert page.start_btn.isEnabled()
    assert page.settings_panel.isEnabled()
    assert not page.model_pack_train_button.isHidden()


def test_layer_observation_controls_and_result_reset(page):
    change_mode(page, "efficientnet_finetune")
    page.debug_check.setChecked(True)
    page._default_debug_patterns()
    page.debug_batches.setValue(2)
    page._sync_config()
    assert page.project.training.layer_debug_enabled
    assert page.project.training.layer_debug_batches == 2
    page._on_layer_debug({"run_id": "selected-run", "epoch": 2, "batch": 1,
        "layers": {"classifier.1": {"input_shapes": [[2, 1280]],
            "output": {"shape": [2, 2], "mean": None, "finite": False},
            "gradient": {"mean": 0.0, "finite": True}}}})
    assert page.debug_table.rowCount() == 1
    assert page.debug_table.item(0, 0).text() == "classifier.1"
    assert page.debug_table.item(0, 5).text() == "N/A"
    assert page.debug_table.item(0, 7).text() == "0"
    page.project.training.training_mode = "unsupported_finetune"
    page.set_project(page.project)
    assert not page.debug_check.isEnabled()
    page._sync_config()
    assert not page.project.training.layer_debug_enabled
    page.set_project(page.project)
    assert page.debug_table.rowCount() == 0
    change_mode(page, "efficientnet_resume")
    assert page.debug_check.isEnabled()
    assert not page.lr_spin.isEnabled()
