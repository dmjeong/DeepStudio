"""Best checkpoint and stopping follow the chosen metric, not a fixed accuracy."""
import json
from types import SimpleNamespace

import pytest

from core.model_selection import available_metrics, selection_policy


def test_mask2former_transfer_and_resume_preserve_learned_query_slots(tmp_path, monkeypatch):
    import torch
    from torch import nn
    import train_builtin as trainer

    model_id = "mask2former_swin_small"
    def build(_model_id, classes, channels, *, mask2former_num_queries, **kwargs):
        model = nn.Module()
        model.weight_provenance = {}
        model.net = nn.Module()
        model.net.model = nn.Module()
        decoder = model.net.model.transformer_module = nn.Module()
        decoder.queries_features = nn.Embedding(mask2former_num_queries, 4)
        decoder.queries_embedder = nn.Embedding(mask2former_num_queries, 4)
        return model

    starts = []
    def epoch(model, loader, criterion, optimizer=None, device="cpu", metrics_out=None, **kwargs):
        if optimizer is not None:
            starts.append({key: value.clone() for key, value in model.state_dict().items()})
            optimizer.zero_grad()
            sum(param.square().mean() for param in model.parameters()).backward()
            optimizer.step()
        else:
            metrics_out.update(val_loss=1., mIoU=.5)
        return 1., .5

    monkeypatch.setattr(trainer, "build_builtin_model", build)
    monkeypatch.setattr(trainer, "create_segmentation_loaders", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(trainer, "_segmentation_epoch", epoch)
    def run(name, **kwargs):
        trainer.train_builtin(model_id, tmp_path, num_classes=2, input_size=32, output_dir=tmp_path / name,
                              scheduler_name="none", log=lambda event: None, **kwargs)
        return tmp_path / name / "last.pt"

    path = run("legacy")
    legacy = torch.load(path, weights_only=True)
    assert legacy["model_config"].pop("num_queries") == 100
    legacy["training_config"].pop("mask2former_num_queries")
    torch.save(legacy, path)
    resumed_legacy = run("resume100", resume=path, epochs=2, mask2former_num_queries=16)
    assert torch.load(resumed_legacy, weights_only=True)["model_config"]["num_queries"] == 100

    small = run("transfer16", initial_weights=path, mask2former_num_queries=16)
    for key, value in starts[-1].items():
        torch.testing.assert_close(value, legacy["model_state_dict"][key][:16])
    resumed = run("resume16", resume=small, epochs=2, mask2former_num_queries=100)
    checkpoint = torch.load(resumed, weights_only=True)
    assert checkpoint["training_config"]["mask2former_num_queries"] == checkpoint["model_config"]["num_queries"] == 16
    for state in checkpoint["optimizer_state_dict"]["state"].values():
        assert state["step"] == 2 and state["exp_avg"].shape == (16, 4)
    with pytest.raises(ValueError, match="16 이하"):
        run("invalid_transfer", initial_weights=small, mask2former_num_queries=100)


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
    losses = iter([.4, .2, .2, .3, .4])
    counter = []

    def epoch(model, loader, criterion, optimizer=None, device="cpu", metrics_out=None,
              pixel_counts_out=None):
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
    # Desktop publication removes best.pt. Resume must preserve the actual best,
    # not silently replace it with last.pt (whose validation loss is worse).
    from core.training_artifacts import publish_best
    publish_best(best, best.parent, {}, epoch=2, metric="val_loss", value=.2,
                 direction="min", engine="builtin", task=task, policy="val_loss")
    resumed = trainer.train_builtin(model_id, tmp_path, num_classes=2, input_size=32,
        epochs=5, scheduler_name="none", selection_metric="val_loss",
        resume=best.parent / "last.pt", output_dir=tmp_path / "resumed", log=lambda _: None)
    resumed_state = torch.load(resumed, weights_only=True)
    assert resumed_state["epoch"] == 1 and resumed_state["metric"] == .2
    with pytest.raises(ValueError, match="Best 기준"):
        trainer.train_builtin(model_id, tmp_path, num_classes=2, input_size=32,
            resume=best.parent / "last.pt", output_dir=tmp_path / "resume", log=lambda _: None)
