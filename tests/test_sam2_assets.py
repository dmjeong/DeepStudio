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


def test_download_uses_pinned_official_file_and_selected_absolute_cache(tmp_path, monkeypatch):
    from sam2_assets import download_sam2_pretrained
    import model_download

    calls = []
    def download(url, destination, sha256):
        calls.append((url, destination, sha256))
        destination.write_bytes(b"official-checkpoint")
        return destination

    monkeypatch.setattr(model_download, "cached_https_weights", download)
    output = download_sam2_pretrained("sam2_hiera_tiny", tmp_path)
    assert output == (tmp_path / "sam2.1_hiera_tiny.pt").resolve()
    assert calls == [("https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
                      output, "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69")]
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


def test_sam2_training_preflight_accepts_background_and_foreground(tmp_path):
    import numpy as np
    from PIL import Image
    from sam2_trainer import _validate_prompt_masks

    image = tmp_path / "sample.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (8, 6), "gray").save(image)
    Image.fromarray(np.zeros((6, 8), dtype=np.uint8)).save(mask)

    class Dataset:
        pairs = [(str(image), str(mask))]

    class Loader:
        dataset = Dataset()

    report = _validate_prompt_masks(Loader(), "train", ["background", "part"])
    assert report['background_images'] == 1 and report['annotated_images'] == 0

    pixels = np.zeros((6, 8), dtype=np.uint8)
    pixels[2:4, 3:5] = 1
    Image.fromarray(pixels).save(mask)
    assert _validate_prompt_masks(Loader(), "train", ["background", "part"]) == {
        "images": 1, "annotated_images": 1, "foreground_class_ids": [1],
        "background_images": 0, "automatic_background_images": 0, "excluded_images": 0,
    }


def test_finetune_rejects_incomplete_decoder_weights(tmp_path, monkeypatch):
    import sam2_assets
    import torch

    checkpoint = tmp_path / "partial.pt"
    torch.save(
        {
            "type": "sam2_finetune",
            "model_id": "sam2_hiera_tiny",
            "model_state_dict": {"sam_mask_decoder.some_weight": torch.ones(1)},
        },
        checkpoint,
    )

    class Model:
        def load_state_dict(self, *args, **kwargs):
            return ["sam_mask_decoder.mask_tokens.weight"], []

    monkeypatch.setattr(sam2_assets, "load_sam2_pretrained", lambda *a, **k: Model())
    with pytest.raises(ValueError, match="모델 구조"):
        sam2_assets.load_sam2_checkpoint("sam2_hiera_tiny", checkpoint_path=checkpoint)


def test_finetune_reuses_the_exact_selected_local_encoder_when_bundle_is_absent(
    tmp_path, monkeypatch
):
    import hashlib
    from types import SimpleNamespace

    import builtin_assets
    import sam2_assets
    import torch

    base = tmp_path / "official_tiny.pt"
    base.write_bytes(b"the exact encoder selected for training")
    digest = hashlib.sha256(base.read_bytes()).hexdigest()
    fine = tmp_path / "fine.pt"
    torch.save(
        {
            "type": "sam2_finetune",
            "model_id": "sam2_hiera_tiny",
            "model_state_dict": {"sam_mask_decoder.weight": torch.ones(1)},
            "base_checkpoint": {
                "model_id": "sam2_hiera_tiny",
                "path": str(base),
                "sha256": digest,
            },
        },
        fine,
    )

    def missing(_):
        raise FileNotFoundError("no installed assets in this development checkout")

    monkeypatch.setattr(builtin_assets, "builtin_asset_path", missing)
    calls = []

    class Model:
        def __init__(self):
            self.sam_mask_decoder = SimpleNamespace(dynamic_multimask_via_stability=True)

        def load_state_dict(self, state, strict=False):
            return [], []

    def selected(model_id, **kwargs):
        calls.append(kwargs)
        return Model()

    monkeypatch.setattr(sam2_assets, "load_sam2_pretrained", selected)
    model = sam2_assets.load_sam2_checkpoint("sam2_hiera_tiny", checkpoint_path=fine)
    assert calls == [{"device": "cpu", "checkpoint_path": base.resolve()}]
    assert model.dvs_multimask_output is False
    assert model.sam_mask_decoder.dynamic_multimask_via_stability is False
    base.write_bytes(b"a different encoder at the same path")
    with pytest.raises(ValueError, match="SHA256"):
        sam2_assets.load_sam2_checkpoint("sam2_hiera_tiny", checkpoint_path=fine)
    assert len(calls) == 1
