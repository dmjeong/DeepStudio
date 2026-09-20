"""Verify weight identity and strict loading without network access."""
from pathlib import Path

import pytest
import torch
from torchvision import models

from builtin_models import build_builtin_model, make_builtin_checkpoint, load_builtin_checkpoint


@pytest.mark.parametrize("model_id,base,first_key,channels", [
    ("resnet18", "resnet18", "conv1", 1),
    ("resnet50", "resnet50", "conv1", 3),
    ("convnext_v1_tiny", "convnext_tiny", "features.0.0", 3),
    ("deeplabv3plus_resnet34", "resnet34", "stem.0", 3),
    ("unet_resnet18", "resnet18", "stem.0", 1),
])
def test_imagenet_loads_correct_backbone_and_preserves_first_convolution(monkeypatch, model_id, base, first_key, channels):
    import builtin_assets
    reference = getattr(models, base)(weights=None)
    state = reference.state_dict()
    key = "features.0.0.weight" if base == "convnext_tiny" else "conv1.weight"
    expected = state[key].clone()
    calls = []
    def packaged(model):
        calls.append(model)
        return Path("official.pth")
    monkeypatch.setattr(builtin_assets, "builtin_asset_path", packaged)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: state)
    model = build_builtin_model(model_id, 2, channels, pretrained=True).eval()
    assert calls == [("convnext_v1_tiny" if base == "convnext_tiny" else base)]
    actual = dict(model.named_modules())[first_key].weight
    torch.testing.assert_close(actual, expected.sum(1, keepdim=True) if channels == 1 else expected, rtol=0, atol=0)
    checkpoint = make_builtin_checkpoint(model_id, model, num_classes=2, in_channels=channels)
    monkeypatch.setattr(builtin_assets, "builtin_asset_path", lambda *a: pytest.fail("restore attempted packaged-asset lookup"))
    restored = load_builtin_checkpoint(checkpoint)
    assert restored.weight_provenance["source"] == "imagenet"
    torch.testing.assert_close(dict(restored.named_modules())[first_key].weight, actual, rtol=0, atol=0)


def test_missing_packaged_imagenet_weight_does_not_train_random_weights(monkeypatch):
    import builtin_assets
    def fail(*args):
        raise OSError("offline")
    monkeypatch.setattr(builtin_assets, "builtin_asset_path", fail)
    with pytest.raises(OSError, match="offline"):
        build_builtin_model("resnet18", 2, pretrained=True)


def test_local_backbone_requires_the_selected_architecture(tmp_path):
    source = tmp_path / "resnet18.pth"
    torch.save(models.resnet18(weights=None).state_dict(), source)
    model = build_builtin_model("unet_resnet18", 2, backbone_weights=source)
    assert model.weight_provenance["source"] == "local_backbone"
    with pytest.raises(ValueError, match="가중치 구조가 일치하지"):
        build_builtin_model("resnet50", 2, backbone_weights=source)
