"""Segmentation training contracts: class names, transfer safety and label alignment."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

NAMES = ["background", "scratch", "dent"]


def _dataset(root, size=32):
    """Two-defect masks; classes 1 and 2 never overlap."""
    from PIL import Image

    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "masks" / split).mkdir(parents=True, exist_ok=True)
        for index in range(2):
            image = (rng.random((size, size, 3)) * 255).astype(np.uint8)
            mask = np.zeros((size, size), np.uint8)
            mask[4:10, 4:20] = 1
            mask[20:26, 8:14] = 2
            Image.fromarray(image).save(root / "images" / split / f"{index}.png")
            Image.fromarray(mask, "L").save(root / "masks" / split / f"{index}.png")
    return root


def _train(data, output, *, class_names=None, initial=None, resume=None, epochs=1, lr=1e-3):
    from train_builtin import train_builtin

    return train_builtin(
        "unet_resnet18", data, num_classes=3, input_size=32, epochs=epochs, batch_size=2,
        learning_rate=lr, weight_decay=0.0, scheduler_name="none", horizontal_flip=0.0,
        rotation=0.0, color_jitter=0.0, freeze_backbone=True, output_dir=output,
        class_names=class_names, initial_weights=initial, resume=resume, log=lambda *_: None)


def _head(path):
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    return {key: value for key, value in state.items() if key.startswith("head.")}


def _head_change(before, after):
    return max(float((before[key] - after[key]).abs().max()) for key in before)


def test_segmentation_checkpoint_and_deployment_json_keep_project_class_names(tmp_path):
    import torch
    from export_onnx import export_checkpoint

    best = _train(_dataset(tmp_path / "data"), tmp_path / "run", class_names=NAMES)
    assert torch.load(best, map_location="cpu", weights_only=False)["class_names"] == NAMES
    exported = export_checkpoint(str(best), str(tmp_path / "out" / "seg.onnx"), log=lambda *_: None)
    assert json.loads(Path(exported["config_path"]).read_text(encoding="utf-8"))["class_names"] == NAMES


def test_segmentation_class_names_must_match_class_count(tmp_path):
    with pytest.raises(ValueError, match="class_names"):
        _train(_dataset(tmp_path / "data"), tmp_path / "run", class_names=NAMES[:2])


@pytest.mark.parametrize(
    ("source_names", "target_names", "head_reused"),
    [
        (NAMES, NAMES, True),                                       # same project classes
        (NAMES, ["background", "dent", "scratch"], False),          # reordered: head would swap classes
        (None, NAMES, False),                                       # legacy "0".."2": meaning unknown
        (None, None, True),                                         # legacy CLI on both sides: unchanged
    ],
)
def test_segmentation_transfer_reuses_head_only_for_the_same_class_order(
        tmp_path, source_names, target_names, head_reused):
    data = _dataset(tmp_path / "data")
    source = _train(data, tmp_path / "source", class_names=source_names)
    # A negligible step leaves a transferred head unchanged; a fresh head differs.
    target = _train(data, tmp_path / "target", class_names=target_names, initial=source, lr=1e-12)
    change = _head_change(_head(source), _head(target))
    assert (change < 1e-6) is head_reused, change


def test_segmentation_resume_rejects_a_different_class_order(tmp_path):
    data = _dataset(tmp_path / "data")
    _train(data, tmp_path / "first", class_names=NAMES)
    with pytest.raises(ValueError, match="클래스 이름"):
        _train(data, tmp_path / "second", class_names=["background", "dent", "scratch"],
               resume=tmp_path / "first" / "last.pt", epochs=2)


def test_training_mask_resize_is_pixel_centre_aligned_with_the_image(tmp_path):
    """INTER_NEAREST shifted labels ~0.38 output px right/down of the image."""
    from PIL import Image
    from dataset import SegmentationDataset

    width, output = 256, 64  # 4x downscale
    shifts = []
    for edge in range(100, 108):  # every phase of the 4-pixel sampling block
        root = tmp_path / str(edge)
        (root / "images").mkdir(parents=True)
        (root / "masks").mkdir(parents=True)
        image = np.zeros((8, width), np.uint8)
        image[:, edge:] = 255
        mask = (image > 0).astype(np.uint8)
        Image.fromarray(image).save(root / "images" / "a.png")
        Image.fromarray(mask, "L").save(root / "masks" / "a.png")
        dataset = SegmentationDataset(str(root / "images"), str(root / "masks"), input_size=(8, output),
                                      in_channels=1, is_train=False, num_classes=2)
        tensor, labels = dataset[0]
        row = tensor[0, 0].numpy()
        image_edge = np.interp((row.min() + row.max()) / 2, row, np.arange(output))
        label_edge = int(np.argmax(labels[0].numpy() == 1)) - 0.5
        shifts.append(label_edge - image_edge)
    assert abs(np.mean(shifts)) < 0.2 and np.max(np.abs(shifts)) <= 0.5 + 1e-6, shifts


def test_sam2_loss_targets_use_pixel_centre_sampling():
    import torch
    pytest.importorskip("sam2_assets")
    from sam2_trainer import _loss_and_scores

    # 8 → 2 px: centre sampling reads source columns 2 and 6, top-left reads 0 and 4.
    targets = torch.zeros(1, 8, 8, dtype=torch.long)
    targets[:, :, 2] = 1
    logits = torch.full((1, 2, 2), 4.0)  # predicts foreground everywhere
    _, _, iou = _loss_and_scores(logits, targets)
    # Centre sampling: truth = left column → (2 + 1) / (4 + 1). Top-left: empty → 0.2.
    assert iou == pytest.approx(0.6)
