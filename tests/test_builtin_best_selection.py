"""Best checkpoint and stopping follow the chosen metric, not a fixed accuracy."""
import json
from types import SimpleNamespace

import pytest

from core.model_selection import available_metrics, selection_policy


@pytest.mark.parametrize("task", ["classify", "detect", "segment"])
def test_all_supervised_tasks_offer_val_loss(task):
    assert "val_loss" in available_metrics("builtin", task)
    assert selection_policy(SimpleNamespace(selection_metric="val_loss"), "builtin", task).direction == "min"


@pytest.mark.parametrize("model_id,task", [("resnet18", "classify"), ("unet_resnet18", "segment")])
def test_minimum_loss_ties_early_stop_and_resume_guard(tmp_path, monkeypatch, model_id, task):
    import torch
    from torch import nn
    import train_builtin as trainer

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc, self.head = nn.Linear(1, 2), nn.Linear(1, 2)
            self.weight_provenance = {"source": "test"}

    monkeypatch.setattr(trainer, "build_builtin_model", lambda *a, **kw: Tiny())
    monkeypatch.setattr(trainer, "create_classification_loaders", lambda *a, **kw: ([], [], ["ok", "ng"]))
    monkeypatch.setattr(trainer, "create_segmentation_loaders", lambda *a, **kw: ([], []))
    losses = iter([.4, .2, .2, .3, .1])
    counter = []

    def epoch(model, loader, criterion, optimizer=None, device="cpu", metrics_out=None):
        if metrics_out is None:
            return 1., 0.
        loss = next(losses)
        counter.append(loss)
        score = len(counter) / 10  # Accuracy/mIoU keep increasing while loss worsens.
        metrics_out.update(val_loss=loss, accuracy=score, mIoU=score)
        return loss, score

    monkeypatch.setattr(trainer, "_classification_epoch" if task == "classify" else "_segmentation_epoch", epoch)
    log = []
    best = trainer.train_builtin(model_id, tmp_path, num_classes=2, input_size=32,
        epochs=10, scheduler_name="none", selection_metric="val_loss", early_stop_patience=2,
        output_dir=tmp_path / "run", log=log.append)
    state = json.loads((best.parent / "training.json").read_text())
    saved = torch.load(best, weights_only=False)
    last = torch.load(best.parent / "last.pt", weights_only=False)
    assert saved["epoch"] == 1 and saved["metric"] == .2
    assert last["epoch"] == 3 and last["metric"] == .3
    assert state["best_epoch"] == 2 and state["completed_epochs"] == 4
    assert state["best_metric_name"] == "val_loss" and state["best_metric"] == .2
    assert all(item["epoch_time_sec"] >= 0 and item["elapsed_time_sec"] >= 0
               for item in state["metrics_history"])
    epoch_events = [item for item in log if isinstance(item, dict) and item["event"] == "epoch_finished"]
    assert all(event["metrics"]["epoch_time_sec"] >= 0 for event in epoch_events)
    assert any(isinstance(item, str) and "총 학습시간:" in item for item in log)
    assert [item["epoch"] for item in log if isinstance(item, dict) and item["event"] == "best_epoch_updated"] == [1, 2]
    with pytest.raises(ValueError, match="Best 기준"):
        trainer.train_builtin(model_id, tmp_path, num_classes=2, input_size=32,
            resume=best.parent / "last.pt", output_dir=tmp_path / "resume", log=lambda _: None)
