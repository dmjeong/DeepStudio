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
    ("model_id", "size"),
    [
        ("resnet18", 32),
        ("resnet50", 32),
        ("convnext_v1_tiny", 32),
        ("deeplabv3plus_resnet34", 64),
        ("unet_resnet18", 64),
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
