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
        "libreyolo_classify_mobilenetv4_small", "libreyolo_classify_mobilenetv4_medium",
        "libreyolo_classify_mobilenetv4_large",
        "libreyolo_detect_9t", "libreyolo_detect_9s", "libreyolo_detect_9m", "libreyolo_detect_9c",
        "re_detr_v4_small", "re_detr_v4_medium", "re_detr_v4_large",
    }
    # 상위 변형은 LibreYOLO의 실제 크기 이름과 기본 입력을 쓴다 (가중치 파일 이름으로 내려받는다)
    assert [(get_upstream_spec(f"libreyolo_detect_9{size}").size,
             get_upstream_spec(f"libreyolo_detect_9{size}").pretrained_name) for size in "smc"] == [
        ("s", "LibreYOLO9s.pt"), ("m", "LibreYOLO9m.pt"), ("c", "LibreYOLO9c.pt")]
    assert get_upstream_spec("libreyolo_classify_mobilenetv4_medium").input_size == 224
    assert get_upstream_spec("libreyolo_classify_mobilenetv4_large").input_size == 256
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


def test_detection_augmentation_maps_screen_values_to_libreyolo_arguments():
    from upstream_models import detection_augmentation_arguments
    assert detection_augmentation_arguments(None) == {}                 # 모델 기본 증강: 아무것도 넘기지 않는다
    assert detection_augmentation_arguments(
        {"horizontal_flip": 0.3, "rotation": 5.0, "color_jitter": 0.0, "mosaic": False}) == {
        "flip_prob": 0.3, "degrees": 5.0, "hsv_prob": 0.0, "mosaic_prob": 0.0}
    assert detection_augmentation_arguments(
        {"horizontal_flip": 0.0, "rotation": 0.0, "color_jitter": 0.2, "mosaic": True}) == {
        "flip_prob": 0.0, "degrees": 0.0, "hsv_prob": 1.0, "mosaic_prob": 1.0}
    with pytest.raises(ValueError, match="증강"):
        detection_augmentation_arguments({"horizontal_flip": 1.5})


def test_detection_training_passes_augmentation_only_when_requested(tmp_path, monkeypatch):
    (tmp_path / "data/images/train").mkdir(parents=True)
    calls = []

    class FakeModel:
        def train(self, **kwargs):
            calls.append(kwargs)
            return {"best_checkpoint": None}

    monkeypatch.setattr(upstream_models, "build_upstream_model", lambda *args, **kwargs: FakeModel())
    common = dict(data_root=tmp_path / "data", class_names=["a"], output_dir=tmp_path / "runs/r", epochs=1,
                  batch_size=2, learning_rate=1e-3, device="cpu", weights=None, resume=False, use_amp=False,
                  patience=1, emit=lambda event: None)
    train_upstream_model("libreyolo_detect_9s", **common)
    train_upstream_model("libreyolo_detect_9s", **common,
                         augmentation={"horizontal_flip": 0.0, "rotation": 0.0, "color_jitter": 0.0, "mosaic": False})
    assert not {"flip_prob", "mosaic_prob", "degrees", "hsv_prob"} & set(calls[0])
    assert {key: calls[1][key] for key in ("flip_prob", "mosaic_prob", "degrees", "hsv_prob")} == {
        "flip_prob": 0.0, "mosaic_prob": 0.0, "degrees": 0.0, "hsv_prob": 0.0}


def test_classification_augmentation_replaces_the_fixed_recipe_and_restores_it():
    pytest.importorskip("libreyolo")
    from torchvision import transforms
    import libreyolo.data.classify_dataset as module
    from upstream_models import classification_augmentation
    original = module.build_classify_transforms
    kinds = lambda compose: [type(step).__name__ for step in compose.transforms]   # noqa: E731
    assert "RandomResizedCrop" in kinds(original(224, True))            # LibreYOLO 기본: 무작위 잘라내기
    with classification_augmentation(None):
        assert module.build_classify_transforms is original             # 모델 기본 증강: 건드리지 않는다
    with classification_augmentation({"horizontal_flip": 0.3, "rotation": 10.0, "color_jitter": 0.1}):
        train = module.build_classify_transforms(224, True)
        assert kinds(train) == ["Resize", "CenterCrop", "RandomHorizontalFlip", "RandomRotation", "ColorJitter",
                                "ToTensor", "Normalize"]
        assert next(step for step in train.transforms if isinstance(step, transforms.RandomHorizontalFlip)).p == 0.3
        assert kinds(module.build_classify_transforms(224, False)) == kinds(original(224, False))   # 검증은 그대로
    with classification_augmentation({"horizontal_flip": 0.0, "rotation": 0.0, "color_jitter": 0.0}):
        assert kinds(module.build_classify_transforms(224, True)) == ["Resize", "CenterCrop", "ToTensor", "Normalize"]
    assert module.build_classify_transforms is original                 # 학습이 끝나면 되돌린다


def test_patch_training_tiles_the_dataset_and_stamps_the_checkpoint(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from PIL import Image
    from upstream_models import checkpoint_tiling
    root = tmp_path / "data"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        Image.fromarray(np.full((960, 1600, 3), 120, np.uint8)).save(root / "images" / split / "a.png")
        (root / "labels" / split / "a.txt").write_text("0 0.1 0.1 0.02 0.02\n", encoding="utf-8")
    seen = {}

    class FakeModel:
        def train(self, **kwargs):
            descriptor = json.loads(Path(kwargs["data"]).read_text(encoding="utf-8"))
            seen["root"] = Path(descriptor["path"])
            seen["tiles"] = sorted(path.name for path in (seen["root"] / "images" / "train").iterdir())
            seen["imgsz"] = kwargs["imgsz"]
            best = tmp_path / "best.pt"
            torch.save({"model_family": "yolo9", "task": "detect", "nc": 1, "names": {0: "a"}, "imgsz": 640}, best)
            return {"best_checkpoint": str(best), "last_checkpoint": None}

    monkeypatch.setattr(upstream_models, "build_upstream_model", lambda *args, **kwargs: FakeModel())
    events = []
    common = dict(data_root=root, class_names=["a"], output_dir=tmp_path / "runs/r", epochs=1, batch_size=2,
                  learning_rate=1e-3, device="cpu", weights=None, resume=False, use_amp=False, patience=1,
                  emit=events.append)
    result = train_upstream_model("libreyolo_detect_9t", **common, tiling={"scale": 1.0, "max_tiles": 0})
    assert seen["imgsz"] == 640 and seen["tiles"][0] == "a_y00000_x00000.png" and len(seen["tiles"]) == 2
    assert not seen["root"].exists()                                    # 타일은 학습 동안만 있는 임시 데이터
    assert events[0]["event"] == "tiles_prepared" and events[0]["splits"]["train"]["object_tiles"] == 1
    contract = {"tile": 640, "overlap": 128, "scale": 1.0, "interpolation": "INTER_AREA", "region": [1600, 960]}
    assert result["tiling"] == contract
    assert checkpoint_tiling(torch.load(result["best_checkpoint"], weights_only=False)) == contract
    assert checkpoint_tiling({"model_family": "yolo9"}) is None         # 전체 축소로 학습한 모델
    with pytest.raises(ValueError, match="dvs_tiling"):
        checkpoint_tiling({"dvs_tiling": {"tile": 640, "overlap": 700, "scale": 1.0}})
    # RT-DETR는 판정 영역 한 번에 추론하는 배포를 만들 수 없어 패치 학습을 받지 않는다
    with pytest.raises(ValueError, match="LibreYOLO9"):
        train_upstream_model("re_detr_v4_small", **common, tiling={"scale": 1.0})


def test_patch_trained_detector_exports_at_the_judgement_region_size(tmp_path):
    """패치 학습한 검출 모델의 내보내기 입력 = 판정 영역 × 배율을 32의 배수로 올린 크기 (타일 크기가 아니다)."""
    torch = pytest.importorskip("torch")
    import export_onnx
    checkpoint = {"model_family": "yolo9", "task": "detect", "nc": 2, "names": {0: "a", 1: "b"}, "imgsz": 640}
    plain = export_onnx._upstream_spec(checkpoint)
    assert (plain["input_height"], plain["input_width"], plain["tiling"]) == (640, 640, None)

    tiled = {**checkpoint, "dvs_tiling": {"tile": 640, "overlap": 128, "scale": 0.5, "interpolation": "INTER_AREA",
                                          "region": [3500, 1500]}}
    spec = export_onnx._upstream_spec(tiled)
    # 3500×1500 → 50% = 1750×750 → 32의 배수 1760×768
    assert (spec["input_width"], spec["input_height"]) == (1760, 768)
    assert spec["preprocessing"]["input_size"] == [768, 1760]
    assert spec["tiling"]["region"] == [3500, 1500] and spec["tiling"]["scaled"] == [1750, 750]
    # 내보내기 화면에서 판정 영역을 바꾸면 그 크기를 쓴다
    assert export_onnx._upstream_spec(tiled, [1600, 960])["input_width"] == 800

    missing = {**checkpoint, "dvs_tiling": {"tile": 640, "overlap": 128, "scale": 1.0, "region": None}}
    with pytest.raises(ValueError, match="판정 영역"):
        export_onnx._upstream_spec(missing)
    assert export_onnx._upstream_spec(missing, [1600, 960])["input_height"] == 960

    path = tmp_path / "best.pt"
    torch.save(tiled, path)
    preview = export_onnx.deployment_preview(path)
    assert preview["patch"] is True and preview["region"] == [3500, 1500] and preview["scale"] == 0.5
    torch.save(checkpoint, path)
    assert export_onnx.deployment_preview(path)["patch"] is False
