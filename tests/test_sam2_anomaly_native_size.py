"""Rectangular native-pixel training contracts for anomaly and prompt segmentation."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "gui")]


def test_native_anomaly_uses_pixels_from_same_roi_for_training_and_inference(tmp_path):
    from patchcore_data import native_anomaly_size, input_tensor, IMAGENET_MEAN, IMAGENET_STD
    from dense_deployment import region_pixels
    from dinomaly_runtime import deployment_config

    path = tmp_path / "normal.png"
    rows, cols = np.indices((55, 73), dtype=np.uint8)
    rgb = np.stack((rows, cols, rows + cols), axis=2)
    Image.fromarray(rgb).save(path)
    crop = {"width": 51, "height": 39}
    size = native_anomaly_size([path], crop)
    assert size == (39, 51)
    roi = rgb[8:47, 11:62]
    normalized = (roi.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = input_tensor(path, size, center_crop=crop)
    np.testing.assert_allclose(tensor.numpy(), normalized.transpose(2, 0, 1), atol=3e-7)
    with pytest.raises(ValueError, match="51×39"):
        input_tensor(path, size)

    config = deployment_config({"input_size": size, "in_channels": 3, "center_crop": crop})
    assert config["preprocessing"]["original_size_region"] == [51, 39]
    np.testing.assert_allclose(region_pixels(config, rgb[:, :, ::-1]), tensor.numpy(), atol=3e-7)
    config["preprocessing"]["center_crop"] = None
    with pytest.raises(ValueError, match="51×39"):
        region_pixels(config, rgb[:, :, ::-1])
    assert "original_size_region" not in deployment_config(
        {"input_size": 64, "in_channels": 3})["preprocessing"]


def test_anomaly_headers_preserve_the_decoders_orientation_and_detect_mixed_sets(tmp_path):
    from patchcore_data import native_anomaly_size
    a, b = tmp_path / "first.jpg", tmp_path / "second.png"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (71, 39)).save(a, exif=exif)
    Image.new("RGB", (39, 71)).save(b)
    assert native_anomaly_size([a]) == (39, 71)  # Dinomaly reads unchanged pixel storage
    assert native_anomaly_size([a, b], transpose_exif=True) == (71, 39)  # PatchCore
    with pytest.raises(ValueError, match="second.png"):
        native_anomaly_size([a, b])


@pytest.mark.parametrize("region, padded, mask_size", [
    ((1024, 1024), (1024, 1024), (256, 256)),
    ((269, 145), (160, 288), (37, 68)),
    ((2048, 1024), (1024, 2048), (256, 512)),
])
def test_native_sam_image_prompt_and_mask_grids(region, padded, mask_size):
    import torch
    from sam2_native import configure_native_region, native_mask_size, pad_image, pad_mask_prompt, visible_logits
    from core.sam2_automatic import automatic_mask

    model = SimpleNamespace(backbone_stride=16, sam_prompt_encoder=SimpleNamespace())
    configure_native_region(model, region)
    image = torch.ones(1, 3, region[1], region[0])
    padded_image = pad_image(image, region)
    assert padded_image.shape[-2:] == padded
    assert torch.equal(padded_image[..., :region[1], :region[0]], image)
    assert padded_image.count_nonzero() == image.numel()
    assert model.sam_prompt_encoder.input_image_size == padded
    assert model.sam_prompt_encoder.image_embedding_size == tuple(v // 16 for v in padded)
    prompt = torch.zeros(1, 1, *native_mask_size(region))
    prompt[..., 8:-8, 5:-5] = 1
    assert prompt.shape[-2:] == mask_size
    encoded = pad_mask_prompt(prompt, region, model.sam_prompt_encoder.mask_input_size)
    assert encoded.shape[-2:] == tuple(v // 4 for v in padded)
    restored = visible_logits(encoded, region)
    assert restored.shape == prompt.shape
    assert torch.all(restored[..., 15:-15, 9:-9] > 0.99)
    assert torch.all(restored[..., :3, :] == 0)
    with pytest.raises(ValueError, match="원본 학습 판정 영역"):
        automatic_mask(SimpleNamespace(model=model), np.zeros((4, 4, 3), np.uint8))
