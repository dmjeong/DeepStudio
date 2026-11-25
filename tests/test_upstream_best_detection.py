import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.project import ProjectManager
from webapp.worker import _train_upstream_project


@pytest.mark.parametrize("inherited", [False, True])
def test_upstream_best_checkpoint_and_csv_use_selected_detection_epoch(tmp_path, monkeypatch, inherited):
    import upstream_models

    project = ProjectManager.create_new("detect", "detect", str(tmp_path), ["crack"])
    best_epoch = 1
    per_class = [{"class_id": 0, "name": "crack", "support": 2, "tp": 1, "fp": 0, "fn": 1}]
    best_metrics = {"mAP_50_95": .6, "mAP_50": .8, "precision": 1., "recall": .5}
    events = []

    def train(model_id, *, output_dir, emit, **settings):
        best = output_dir / "weights" / "best.pt"
        best.parent.mkdir(parents=True)
        best.write_bytes(b"best model")
        for epoch in ([2] if inherited else [1, 2]):
            selected = epoch == best_epoch
            metrics = best_metrics if selected else {"mAP_50_95": .1, "precision": .2, "recall": .3}
            emit({"event": "epoch_finished", "epoch": epoch, "total_epochs": 2,
                  "train_loss": .2, "val_loss": .4, "metric": metrics["mAP_50_95"],
                  "epoch_time_sec": 1.2, "elapsed_time_sec": epoch * 1.2,
                  "metric_name": "mAP_50_95", "metrics": metrics, "is_best": selected,
                  "best_epoch": best_epoch, "detection_per_class": per_class if selected else []})
        return {"best_checkpoint": str(best), "best_epoch": best_epoch, "best_mAP50_95": .6,
                "best_validation": {"epoch": best_epoch, "metrics": best_metrics, "per_class": per_class}}

    monkeypatch.setattr(upstream_models, "train_upstream_model", train)
    record = _train_upstream_project(SimpleNamespace(emit=lambda *args: events.append(args)), project, "cpu")
    assert (record.best_epoch, record.best_metric_name, record.best_metric) == (1, "mAP_50_95", .6)
    assert record.eval_results["per_class"] == per_class
    assert record.eval_results["detection_evaluation"]["unit"] == "image"
    assert record.eval_results["precision"] == 1.
    checkpoint = Path(record.checkpoint_path)
    assert checkpoint.name == "best_mAP_50_95_0.600000_epoch_1.pt" and checkpoint.read_bytes() == b"best model"
    with (checkpoint.parents[1] / "best_result.csv").open(encoding="utf-8", newline="") as stream:
        best, = csv.DictReader(stream)
    assert (float(best["mAP_50_95"]), float(best["precision"])) == (.6, 1.)
    with (checkpoint.parents[1] / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
        selected, = (row for row in csv.DictReader(stream) if row["is_best"] == "1")
    assert (int(selected["epoch"]), float(selected["precision"])) == (1, 1.)
    assert any(event[0] == "eval_finished" for event in events)


def test_upstream_detection_does_not_publish_last_checkpoint_as_best(tmp_path, monkeypatch):
    import upstream_models

    project = ProjectManager.create_new("detect", "detect", str(tmp_path), ["crack"])
    last = tmp_path / "last.pt"
    last.write_bytes(b"last model")
    monkeypatch.setattr(upstream_models, "train_upstream_model", lambda *args, **kwargs: {
        "last_checkpoint": str(last), "best_epoch": 1, "best_mAP50_95": .6,
    })
    with pytest.raises(RuntimeError, match="Best 에폭의 체크포인트"):
        _train_upstream_project(SimpleNamespace(emit=lambda *_: None), project, "cpu")
    assert not project.runs
