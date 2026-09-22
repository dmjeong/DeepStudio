"""Different checkpoint predictions must retain different P/R through best CSV."""
import csv

import numpy as np
import pytest
import torch

from core.best_metrics import best_epoch_metrics
from core.metrics import ClassificationMetrics
from core.project import RunRecord
from core.training_artifacts import publish_best
from train_builtin import _classification_epoch


def test_separate_predictions_produce_separate_metrics_and_saved_best_values(tmp_path):
    targets = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])
    predictions = ([0, 0, 0, 1, 1, 0, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1])
    expected = ((.625, (3 / 5 + 2 / 3) / 2), (.875, .9))
    saved = []
    for index, (preds, (precision, recall)) in enumerate(zip(predictions, expected)):
        logits = torch.nn.functional.one_hot(torch.tensor(preds), 2).float() * 4
        meter = ClassificationMetrics(2, ["a", "b"])
        meter.update(np.array(preds), targets.numpy())
        reference = meter.compute()
        actual = {}
        # Split batches so the metric must aggregate the entire validation set.
        loader = [(logits[:3], targets[:3]), (logits[3:], targets[3:])]
        _classification_epoch(torch.nn.Identity(), loader, torch.nn.CrossEntropyLoss(), metrics_out=actual)
        assert actual["precision_macro"] == pytest.approx(precision)
        assert actual["recall_macro"] == pytest.approx(recall)
        for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            assert actual[key] == pytest.approx(reference[key])
        directory = tmp_path / str(index)
        directory.mkdir()
        checkpoint = directory / "best.pt"
        checkpoint.write_bytes(b"fixture checkpoint")
        history = {"epoch": [2, 3], **{key: [value, 0.] for key, value in actual.items()}}
        _, selection = publish_best(checkpoint, directory, history, epoch=2, metric="accuracy",
            value=actual["accuracy"], direction="max", engine="builtin", task="classify", policy="saved")
        displayed = best_epoch_metrics(RunRecord(best_epoch=2, best_metric=actual["accuracy"],
            best_metric_name="accuracy", metrics_history=history, config_snapshot={"best_selection": selection}))
        with (directory / "best_result.csv").open(encoding="utf-8-sig", newline="") as stream:
            row = next(csv.DictReader(stream))
        assert float(row["precision_macro"]) == displayed["precision_macro"]
        assert float(row["recall_macro"]) == displayed["recall_macro"]
        saved.append(displayed)
    assert saved[0]["precision_macro"] != saved[1]["precision_macro"]
    assert saved[0]["recall_macro"] != saved[1]["recall_macro"]


def test_builtin_epoch_excludes_class_without_validation_samples():
    targets = torch.tensor([0, 0, 1, 1])
    predictions = torch.tensor([0, 0, 0, 1])
    logits = torch.nn.functional.one_hot(predictions, 3).float() * 4
    actual = {}
    _classification_epoch(torch.nn.Identity(), [(logits, targets)],
                          torch.nn.CrossEntropyLoss(), metrics_out=actual)
    assert actual["accuracy"] == pytest.approx(.75)
    assert actual["precision_macro"] == pytest.approx((2 / 3 + 1) / 2)
    assert actual["recall_macro"] == pytest.approx(.75)
    assert actual["f1_macro"] == pytest.approx((.8 + 2 / 3) / 2)
