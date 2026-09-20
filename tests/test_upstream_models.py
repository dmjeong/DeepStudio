import json
from types import SimpleNamespace

import pytest

from upstream_models import (_ProgressCallback, get_upstream_spec, upstream_model_ids,
                             write_detection_dataset_yaml)


def test_shipped_upstream_ids_are_explicit_and_use_real_family_sizes():
    assert upstream_model_ids() == {
        "libreyolo_classify_mobilenetv4_small", "libreyolo_detect_9t",
        "re_detr_v4_small", "re_detr_v4_medium", "re_detr_v4_large",
    }
    assert get_upstream_spec("re_detr_v4_small").size == "s"
    assert get_upstream_spec("re_detr_v4_medium").size == "m"
    assert get_upstream_spec("re_detr_v4_large").size == "l"


def test_detection_descriptor_preserves_existing_yolo_layout_and_unicode_names(tmp_path):
    (tmp_path / "images/train").mkdir(parents=True)
    output = write_detection_dataset_yaml(tmp_path, ["정상", "불량"], tmp_path / "run/dataset.yaml")
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data == {"path": str(tmp_path.resolve()), "train": "images/train",
                    "val": "images/train", "nc": 2, "names": ["정상", "불량"]}


def test_detection_descriptor_rejects_missing_train_split_or_invalid_class_names(tmp_path):
    with pytest.raises(ValueError, match="학습 이미지"):
        write_detection_dataset_yaml(tmp_path, ["ok"], tmp_path / "dataset.yaml")
    (tmp_path / "images/train").mkdir(parents=True)
    with pytest.raises(ValueError, match="class names"):
        write_detection_dataset_yaml(tmp_path, [""], tmp_path / "dataset.yaml")


def test_upstream_callback_records_real_validation_loss_instead_of_training_loss():
    records = []
    callback = _ProgressCallback(records.append)
    callback.on_train_start(SimpleNamespace(start_epoch=1, total_epochs=3, model_family="rtdetrv4",
                                             model_size="s", task="detect", save_dir="run"))
    callback.on_train_epoch_end(SimpleNamespace(
        epoch=0, total_epochs=3, train_loss=2.0, epoch_seconds=1.5,
        val_metrics={"metrics/val_loss": 1.25, "metrics/mAP50-95": .6},
        current_metric=.6, current_metric_name="metrics/mAP50-95",
        is_best=True, best_metric=.6, best_epoch=1, lr={"default": .001},
    ))
    assert records == [{"event": "training_started", "epoch": 1, "total_epochs": 3,
                        "family": "rtdetrv4", "size": "s", "task": "detect", "save_dir": "run"},
                       {"event": "epoch_finished", "epoch": 0, "total_epochs": 3,
                        "train_loss": 2.0, "val_loss": 1.25, "metric": .6,
                        "metric_name": "metrics/mAP50-95",
                        "metrics": {"metrics/val_loss": 1.25, "metrics/mAP50-95": .6,
                                    "epoch_time_sec": 1.5, "elapsed_time_sec": 1.5},
                        "epoch_time_sec": 1.5, "elapsed_time_sec": 1.5,
                        "learning_rate": .001, "is_best": True, "best_metric": .6,
                        "best_epoch": 1}]
