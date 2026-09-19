"""Weight-free built-in model construction and generic ONNX deployment tests."""

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


@pytest.mark.parametrize(
    ("model_id", "task", "shape"),
    [
        ("resnet18", "classify", (1, 3, 32, 32)),
        ("resnet50", "classify", (1, 3, 32, 32)),
        ("convnext_v1_tiny", "classify", (1, 3, 32, 32)),
        ("deeplabv3plus_resnet34", "segment", (1, 3, 64, 64)),
        ("unet_resnet18", "segment", (1, 3, 64, 64)),
    ],
)
def test_builtin_models_are_constructible_without_pretrained_download(model_id, task, shape):
    import torch
    from builtin_models import build_builtin_model, get_builtin_spec, make_builtin_checkpoint

    model = build_builtin_model(model_id, 2, shape[1]).eval()
    assert model.task == task
    assert model.in_channels == shape[1]
    assert model.num_classes == 2
    assert isinstance(model.gradcam_target_layer, str)
    with torch.inference_mode():
        output = model(torch.zeros(shape))
    assert output.shape[:2] == (1, 2)
    if task == "segment":
        assert tuple(output.shape[-2:]) == shape[-2:]
    checkpoint = make_builtin_checkpoint(model_id, model, num_classes=2,
                                         input_size=shape[-1], class_names=["ok", "ng"])
    assert checkpoint["backend"] == "builtin"
    assert checkpoint["model_id"] == get_builtin_spec(model_id).model_id


@pytest.mark.parametrize(
    ("model_id", "target_layer"),
    [
        ("resnet18", "layer4"),
        ("resnet50", "layer4"),
        ("convnext_v1_tiny", "features.7"),
        ("deeplabv3plus_resnet34", "layer4"),
        ("unet_resnet18", "enc4"),
    ],
)
def test_builtin_model_runtime_contract_supports_gradcam(model_id, target_layer):
    from builtin_models import build_builtin_model

    model = build_builtin_model(model_id, 2, 3).eval()
    current = model
    for component in target_layer.split("."):
        current = getattr(current, component)
    assert current is not None


@pytest.mark.parametrize(
    ("model_id", "size"),
    [
        ("resnet18", 224),
        ("resnet50", 224),
        ("convnext_v1_tiny", 224),
        ("deeplabv3plus_resnet34", 512),
        ("unet_resnet18", 512),
    ],
)
def test_builtin_checkpoint_exports_and_has_cpp_contract(tmp_path, model_id, size):
    import torch
    from builtin_models import build_builtin_model, make_builtin_checkpoint
    from export_onnx import export_checkpoint

    model = build_builtin_model(model_id, 2, 3).eval()
    checkpoint = make_builtin_checkpoint(model_id, model, num_classes=2,
                                         input_size=size, class_names=["ok", "ng"])
    source = tmp_path / "model.pt"
    output = tmp_path / "model.onnx"
    torch.save(checkpoint, source)
    result = export_checkpoint(source, output, verify=True, log=lambda _: None)
    manifest = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert result["verification"] == "passed"
    assert manifest["backend"] == "builtin"
    assert manifest["cpp_supported"] is True
    assert manifest["schema_version"] == 5
    assert manifest["model_config"]["model_id"] == model_id


def test_builtin_training_checkpoint_roundtrips_for_export(tmp_path):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from builtin_models import build_builtin_model
    from train_builtin import _classification_epoch, save_builtin_checkpoint
    from export_onnx import export_checkpoint

    model = build_builtin_model("resnet18", 2, 3)
    loader = DataLoader(TensorDataset(torch.randn(2, 3, 64, 64), torch.tensor([0, 1])), batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _classification_epoch(model, loader, torch.nn.CrossEntropyLoss(), optimizer)
    checkpoint = tmp_path / "last.pt"
    save_builtin_checkpoint(checkpoint, model.eval(), model_id="resnet18", num_classes=2,
                            input_size=64, in_channels=3, class_names=["ok", "ng"],
                            epoch=0, optimizer=optimizer, metric=0.5)
    result = export_checkpoint(checkpoint, tmp_path / "model.onnx", verify=True, log=lambda _: None)
    assert result["verification"] == "passed"


def test_builtin_trainer_persists_applied_ui_settings(tmp_path):
    from PIL import Image
    import torch
    from train_builtin import train_builtin

    data = tmp_path / "data"
    for split in ("train", "val"):
        for class_name, value in (("ok", 32), ("ng", 224)):
            folder = data / split / class_name
            folder.mkdir(parents=True)
            Image.new("RGB", (32, 32), (value, value, value)).save(folder / "sample.png")
    output = tmp_path / "run"
    best = train_builtin(
        "resnet18", data, input_size=32, epochs=1, batch_size=2,
        optimizer_name="sgd", scheduler_name="none", weight_decay=0.0123,
        horizontal_flip=0.0, rotation=0.0, color_jitter=0.0,
        freeze_backbone=True, output_dir=output,
    )
    state = json.loads((output / "training.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(best, map_location="cpu", weights_only=False)
    assert state["completed_epochs"] == 1
    assert state["best_epoch"] == 1
    assert len(state["metrics_history"]) == 1
    assert state["training_config"]["optimizer"] == "sgd"
    assert state["training_config"]["weight_decay"] == 0.0123
    assert state["training_config"]["augmentation"] == {
        "horizontal_flip": 0.0, "rotation": 0.0, "color_jitter": 0.0,
    }
    assert checkpoint["training_config"] == state["training_config"]


def test_builtin_resume_preserves_best_metric_from_history(tmp_path):
    from PIL import Image
    import torch
    from train_builtin import train_builtin

    data = tmp_path / "data"
    for split in ("train", "val"):
        for class_name, value in (("ok", 32), ("ng", 224)):
            folder = data / split / class_name
            folder.mkdir(parents=True)
            Image.new("RGB", (32, 32), (value, value, value)).save(folder / "sample.png")
    first = tmp_path / "first"
    train_builtin(
        "resnet18", data, input_size=32, epochs=1, batch_size=2,
        scheduler_name="none", horizontal_flip=0.0, rotation=0.0,
        color_jitter=0.0, freeze_backbone=True, output_dir=first,
    )
    resume = torch.load(first / "last.pt", map_location="cpu", weights_only=False)
    resume["metric"] = -1.0
    resume["metrics_history"] = [{
        "epoch": 1, "train_loss": 1.0, "train_metric": 0.0,
        "val_loss": 1.0, "val_metric": 2.0, "learning_rate": 1e-3,
    }]
    torch.save(resume, first / "last.pt")
    previous_best = torch.load(first / "best.pt", map_location="cpu", weights_only=False)
    previous_best["metric"] = 2.0
    previous_best["epoch"] = 0
    torch.save(previous_best, first / "best.pt")
    second = tmp_path / "second"
    train_builtin(
        "resnet18", data, input_size=32, epochs=2, batch_size=2,
        scheduler_name="none", horizontal_flip=0.0, rotation=0.0,
        color_jitter=0.0, freeze_backbone=True, output_dir=second,
        resume=first / "last.pt",
    )
    state = json.loads((second / "training.json").read_text(encoding="utf-8"))
    assert state["best_metric"] == 2.0
    assert state["best_epoch"] == 1
