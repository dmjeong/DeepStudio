import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import upstream_models
from upstream_models import (
    _ProgressCallback,
    get_upstream_spec,
    prepare_classification_dataset,
    train_upstream_model,
    upstream_model_ids,
    write_detection_dataset_yaml,
)


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


def test_classification_dataset_fills_empty_val_per_class_without_touching_sources(tmp_path):
    root = tmp_path / "data"
    for class_name in ("double", "single"):
        for index in range(4):
            path = root / "train" / class_name / f"nested/{index}.JPG"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"image")
        (root / "val" / class_name).mkdir(parents=True)

    with prepare_classification_dataset(
        root, ["double", "single"], tmp_path / "runs", val_split=0.25, seed=7,
    ) as (prepared, summary):
        prepared_path = prepared
        assert summary == {"automatic_classes": ["double", "single"],
                           "train_images": 6, "val_images": 2}
        for class_name in ("double", "single"):
            assert len(list((prepared / "train" / class_name).glob("*.jpg"))) == 3
            assert len(list((prepared / "val" / class_name).glob("*.jpg"))) == 1
    assert not prepared_path.exists()
    assert len(list((root / "train" / "double").rglob("*.JPG"))) == 4


def test_classification_dataset_reports_class_that_cannot_be_split(tmp_path):
    image = tmp_path / "data/train/double/only.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    with (
        pytest.raises(ValueError, match="클래스 'double'.*최소 2장"),
        prepare_classification_dataset(
            tmp_path / "data", ["double"], tmp_path / "runs",
        ),
    ):
        pass


def test_libreyolo_classification_trains_with_prepared_dataset(tmp_path, monkeypatch):
    root = tmp_path / "data"
    for class_name in ("double", "single"):
        for index in range(3):
            path = root / "train" / class_name / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"image")
        (root / "val" / class_name).mkdir(parents=True)

    observed = {}

    class FakeModel:
        def train(self, **kwargs):
            prepared = Path(kwargs["data"])
            observed["root"] = prepared
            observed["counts"] = {
                (split, name): len(list((prepared / split / name).iterdir()))
                for split in ("train", "val") for name in ("double", "single")
            }
            return {"best_checkpoint": "best.pt"}

    monkeypatch.setattr(upstream_models, "build_upstream_model", lambda *args, **kwargs: FakeModel())
    events = []
    result = train_upstream_model(
        "libreyolo_classify_mobilenetv4_small", data_root=root,
        class_names=["double", "single"], output_dir=tmp_path / "runs/run-1",
        epochs=1, batch_size=2, learning_rate=1e-3, device="cpu", weights=None,
        resume=False, use_amp=False, patience=1, val_split=0.34, seed=3, emit=events.append,
    )
    assert result == {"best_checkpoint": "best.pt"}
    assert observed["counts"] == {
        ("train", "double"): 2, ("train", "single"): 2,
        ("val", "double"): 1, ("val", "single"): 1,
    }
    assert not observed["root"].exists()
    assert events[0]["event"] == "dataset_prepared"


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
