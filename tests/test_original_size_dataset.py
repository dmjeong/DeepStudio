"""Original-resolution segmentation loader and full-frame preprocessing checks."""

from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT / "python")]
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from dataset import OriginalSizeSegmentationDataset, create_segmentation_loaders  # noqa: E402


def _dataset(root, size=(37, 69)):
    height, width = size
    pixels = (np.arange(height * width).reshape(size) % 251).astype(np.uint8)
    mask = np.zeros(size, np.uint8)
    mask[-1, -2] = 1
    for split in ("train", "val"):
        images = root / "images" / split
        masks = root / "masks" / split
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        Image.fromarray(pixels).save(images / "defect.png")
        Image.fromarray(mask).save(masks / "defect.png")
        Image.fromarray(pixels).save(images / "unlabeled.png")
    return pixels


def test_full_frame_keeps_original_pixels_and_ignores_padding(tmp_path):
    from core.inference_engine import InferenceOperations

    pixels = _dataset(tmp_path)
    train, val = create_segmentation_loaders(
        str(tmp_path), input_size=(32, 32), original_size=True,
        in_channels=1, num_classes=2, batch_size=2, flip_prob=1.0,
    )
    assert isinstance(train.dataset, OriginalSizeSegmentationDataset)
    assert train.dataset.original_size == val.dataset.original_size == (37, 69)
    assert train.dataset.padded_size == val.dataset.padded_size == (64, 96)
    assert len(train.dataset) == len(val.dataset) == 2
    image, target = next(iter(val))
    assert image.shape == (2, 1, 64, 96)
    assert target.shape == (2, 64, 96)
    assert image[..., 37:, :].count_nonzero() == image[..., :, 69:].count_nonzero() == 0
    assert (target[..., 37:, :] == 255).all() and (target[..., :, 69:] == 255).all()
    assert target[0, 36, 67] == 1 and (target[0] == 1).sum() == 1
    assert (target[1, :37, :69] == 0).all()
    assert train.dataset.class_pixel_counts(2) == [2 * 37 * 69 - 1, 1]
    flipped, flipped_target = train.dataset[0]
    assert torch.equal(flipped[..., :37, :69], image[0, ..., :37, :69].flip(-1))
    assert flipped_target[36, 1] == 1

    # Existing whole-image inference must see the same padded model input.
    engine = InferenceOperations()
    received = []

    def capture(inputs, device):
        received.append(inputs)
        return inputs

    engine._forward_probabilities = capture
    engine._whole_probabilities(pixels, val.dataset.mean, val.dataset.std, "cpu")
    torch.testing.assert_close(received[0][0], image[0], atol=0, rtol=0)


@pytest.mark.parametrize("split", ["train", "val"])
def test_original_size_rejects_mixed_image_sizes(tmp_path, split):
    _dataset(tmp_path)
    # Both widths round to the same model input width, but deployment ROI differs.
    Image.new("L", (70, 37)).save(tmp_path / "images" / split / "mixed.png")
    with pytest.raises(ValueError, match=r"크기가 모두 같아야 합니다.*mixed\.png"):
        create_segmentation_loaders(str(tmp_path), original_size=True)


def test_original_size_rejects_patch_sampling(tmp_path):
    with pytest.raises(ValueError, match="동시에 사용할 수 없습니다"):
        create_segmentation_loaders(str(tmp_path), original_size=True, patch_size=64)
