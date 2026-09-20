"""SAM2.1 pretrained asset mapping and download contract."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest


def test_sam2_assets_cover_every_shipped_hiera_variant():
    from sam2_assets import SAM2_ASSETS, get_sam2_asset

    assert set(SAM2_ASSETS) == {
        "sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_base_plus", "sam2_hiera_large",
    }
    for model_id, asset in SAM2_ASSETS.items():
        assert asset.hub_model_id.startswith("facebook/sam2.1-hiera-")
        assert asset.filename.endswith(".pt")
        assert asset.config_name.startswith("configs/sam2.1/")
        assert get_sam2_asset(model_id) is asset
    with pytest.raises(ValueError, match="지원하지 않는"):
        get_sam2_asset("sam2_hiera_xl")


def test_download_uses_the_official_repo_and_selected_absolute_cache(tmp_path, monkeypatch):
    from sam2_assets import download_sam2_pretrained

    calls = []
    hub = ModuleType("huggingface_hub")

    def download(**kwargs):
        calls.append(kwargs)
        destination = Path(kwargs["local_dir"]) / kwargs["filename"]
        destination.write_bytes(b"official-checkpoint")
        return str(destination)

    hub.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    output = download_sam2_pretrained("sam2_hiera_tiny", tmp_path)
    assert output == (tmp_path / "sam2.1_hiera_tiny.pt").resolve()
    assert calls == [{"repo_id": "facebook/sam2.1-hiera-tiny", "filename": "sam2.1_hiera_tiny.pt", "local_dir": str(tmp_path)}]
    with pytest.raises(ValueError, match="절대 경로"):
        download_sam2_pretrained("sam2_hiera_tiny", "relative-cache")


def test_load_uses_official_builder_and_selected_checkpoint(tmp_path, monkeypatch):
    from sam2_assets import load_sam2_pretrained

    checkpoint = tmp_path / "tiny.pt"
    checkpoint.write_bytes(b"checkpoint")
    calls = []
    package, builder = ModuleType("sam2"), ModuleType("sam2.build_sam")

    def build(config, **kwargs):
        calls.append((config, kwargs))
        return "official-model"

    builder.build_sam2 = build
    monkeypatch.setitem(sys.modules, "sam2", package)
    monkeypatch.setitem(sys.modules, "sam2.build_sam", builder)
    assert load_sam2_pretrained("sam2_hiera_tiny", device="cuda:0", checkpoint_path=checkpoint) == "official-model"
    assert calls == [("configs/sam2.1/sam2.1_hiera_t.yaml", {
        "ckpt_path": str(checkpoint), "device": "cuda:0", "mode": "eval",
    })]


def test_finetune_checkpoint_is_layered_on_the_matching_bundled_hiera_model(tmp_path, monkeypatch):
    from unittest.mock import patch
    from sam2_assets import load_sam2_checkpoint

    checkpoint = tmp_path / "fine-tune.pt"
    checkpoint.write_bytes(b"fine-tune")
    payload = {"type": "sam2_finetune", "model_id": "sam2_hiera_tiny",
               "model_state_dict": {"weight": 2.0}}
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    sys.modules["torch"].load = lambda *args, **kwargs: payload

    class FakeModel:
        def __init__(self):
            self.loaded = None

        def load_state_dict(self, state, strict=False):
            self.loaded = (state, strict)
            return [], []

    model = FakeModel()
    with patch("sam2_assets.load_sam2_pretrained", return_value=model) as base:
        assert load_sam2_checkpoint("sam2_hiera_tiny", checkpoint_path=checkpoint) is model
    base.assert_called_once_with("sam2_hiera_tiny", device="cpu")
    assert model.loaded[0]["weight"] == 2.0


def test_finetune_checkpoint_rejects_a_different_hiera_variant(tmp_path, monkeypatch):
    from sam2_assets import load_sam2_checkpoint

    checkpoint = tmp_path / "fine-tune.pt"
    checkpoint.write_bytes(b"fine-tune")
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    sys.modules["torch"].load = lambda *args, **kwargs: {
        "type": "sam2_finetune", "model_id": "sam2_hiera_small", "model_state_dict": {"weight": 2.0},
    }
    with pytest.raises(ValueError, match="sam2_hiera_small"):
        load_sam2_checkpoint("sam2_hiera_tiny", checkpoint_path=checkpoint)
