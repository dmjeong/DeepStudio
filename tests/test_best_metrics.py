"""One saved best epoch feeds CSV, dashboard and summary across model families."""
import csv
import json

import pytest

from core.best_metrics import best_epoch_metrics, scalar_metrics
from core.project import RunRecord
from core.training_artifacts import publish_best


@pytest.mark.parametrize("engine,task,metric,extra", [
    ("efficientnet", "classify", "accuracy", {"f1_macro": .7}),
    ("builtin", "classify", "val_loss", {"accuracy": .7, "recall_macro": .6}),
    ("upstream", "classify", "accuracy", {"top5_accuracy": .9}),
    ("builtin", "segment", "mIoU", {"dice_score": .7, "pixel_accuracy": .8}),
    ("upstream", "detect", "mAP_50_95", {"mAP_50": .7, "precision": .8}),
    ("sam2", "segment", "prompt_dice", {"prompt_iou": .7}),
    ("patchcore", "anomaly", "auroc", {"f1": .7, "precision": .8, "recall": .9}),
])
def test_csv_and_reopened_record_share_exact_best_metrics(tmp_path, engine, task, metric, extra):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"selected checkpoint")
    score = .523456789
    history = {"epoch": [11, 12], "train_loss": [.4, .1], "val_loss": [.3, .05],
               metric: [score, .1], "epoch_time_sec": [8., 9.]}
    evaluation = {"task": task, **extra, "sample_count": 100, "evaluable": True}
    named, selection = publish_best(checkpoint, tmp_path, history, epoch=11, metric=metric, value=score,
        direction="min" if metric == "val_loss" else "max", engine=engine, task=task,
        policy="saved", evaluation_metrics=evaluation)
    run = RunRecord(best_epoch=11, best_metric_name=metric, best_metric=score,
        checkpoint_path=named, metrics_history=history, eval_results=evaluation,
        config_snapshot={"best_selection": selection})
    # Same serialization used by project files, including the full-precision snapshot.
    from dataclasses import asdict
    restored = RunRecord(**json.loads(json.dumps(asdict(run))))
    displayed = best_epoch_metrics(restored)
    with (tmp_path / "best_result.csv").open(encoding="utf-8-sig", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert displayed == selection["metrics"]
    assert displayed[metric] == score
    assert displayed["train_loss"] == .4
    assert all(float(row[key]) == value for key, value in displayed.items())
    assert "sample_count" not in displayed and "epoch_time_sec" not in displayed
    assert set(extra) <= set(displayed)


def test_resume_best_without_its_history_never_borrows_latest_losses():
    run = RunRecord(best_epoch=3, best_metric_name="accuracy", best_metric=.8,
                    metrics_history={"epoch": [11, 12], "val_loss": [.2, .1]})
    assert best_epoch_metrics(run) == {"accuracy": .8}
    run.best_epoch = 1
    run.metrics_history.pop("epoch")
    run.config_snapshot = {"training": {"training_mode": "upstream_resume"}}
    assert best_epoch_metrics(run) == {"accuracy": .8}


def test_missing_and_nonfinite_values_are_not_zero_or_previous_scores():
    assert scalar_metrics({"f1": None, "accuracy": float("nan"), "valid": True}) == {"f1": None, "accuracy": None}
    assert best_epoch_metrics(RunRecord(best_epoch=0, best_metric_name="accuracy")) == {}


def test_worker_retains_a_metric_present_only_at_the_best_epoch(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import torch
    import train_builtin
    from core.project import ProjectManager
    from webapp.worker import _train_builtin_project
    project = ProjectManager.create_new("metrics", "classify", str(tmp_path / "project"), ["a", "b"])
    project.model.model_id = "resnet18"
    project.training.training_mode = "builtin_scratch"
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"best epoch 12")

    def fake_train(*args, log, **kwargs):
        for epoch in (11, 12, 13):
            log({"event": "epoch_finished", "epoch": epoch, "total_epochs": 13,
                 "train_loss": .4, "val_loss": .3, "metric": .8,
                 "metrics": {"accuracy": .8, **({"top5_accuracy": .95} if epoch == 12 else {})}})
        return checkpoint

    monkeypatch.setattr(train_builtin, "train_builtin", fake_train)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"epoch": 11, "metric": .8})
    record = _train_builtin_project(SimpleNamespace(emit=lambda *args: None, cancelled=lambda: False), project, "cpu")
    assert record.metrics_history["epoch"] == [11, 12, 13]
    assert record.metrics_history["top5_accuracy"] == [None, .95, None]
    assert best_epoch_metrics(record)["top5_accuracy"] == .95
    assert record.config_snapshot["best_selection"]["metrics"]["top5_accuracy"] == .95
