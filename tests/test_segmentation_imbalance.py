"""Small defects in segmentation training: mask downscaling, imbalance-aware loss, mask prefetch."""

import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")


def _components(mask, cls):
    return cv2.connectedComponents((mask == cls).astype(np.uint8), connectivity=8)[0] - 1


# ── resize_label_mask ──────────────────────────────────────────

def test_thin_defects_survive_a_4x_downscale():
    """Plain nearest reads one pixel per 4x4 block, so ~half of 2px scratches vanished."""
    from dataset import resize_label_mask

    mask = np.zeros((512, 512), np.uint8)
    for k in range(20):  # 2px horizontal scratches at every sampling phase
        mask[10 + k * 25: 12 + k * 25, 40:200] = 1
    for k in range(20):  # 3x3 pits
        y, x = 13 + k * 25, 300 + (k % 4) * 30
        mask[y:y + 3, x:x + 3] = 2
    nearest = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST_EXACT)
    assert _components(nearest, 1) + _components(nearest, 2) < 40  # the bug: defects lost
    resized = resize_label_mask(mask, (128, 128))
    assert _components(resized, 1) == 20 and _components(resized, 2) == 20


@pytest.mark.parametrize("width", [1, 2])
def test_oblique_thin_scratches_stay_one_connected_line(width):
    """Nearest sampling cut a 1-2px diagonal scratch into dashes (median 5 pieces at 4x)."""
    from dataset import resize_label_mask

    broken = 0
    for angle in np.linspace(0.1, 3.0, 12):
        mask = np.zeros((512, 512), np.uint8)
        end = (int(100 + 300 * np.cos(angle)), int(250 + 200 * np.sin(angle)))
        cv2.line(mask, (100, 250), end, 1, width)
        nearest = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST_EXACT)
        broken += _components(nearest, 1) > 1
        resized = resize_label_mask(mask, (128, 128))
        assert _components(resized, 1) == 1, angle
        # The restored line stays a line, not a blob: about 1-2 output pixels wide
        # (a diagonal source line straddles two output pixels per row/column).
        rows, cols = np.nonzero(resized)
        assert (resized == 1).sum() <= 2.5 * (max(np.ptp(rows), np.ptp(cols)) + 1)
    assert broken >= 6  # the bug this guards against


def test_large_defects_are_unchanged_and_other_labels_are_never_overwritten():
    from dataset import resize_label_mask

    mask = np.zeros((400, 400), np.uint8)
    mask[50:250, 60:300] = 1                 # large defect: nearest keeps it well
    mask[300:302, 0:400] = 2                 # thin scratch crossing ...
    mask[290:310, 100:140] = 255             # ... a training-exclusion region
    mask[260:262, 0:400] = 1                 # thin scratch next to nothing
    nearest = cv2.resize(mask, (100, 100), interpolation=cv2.INTER_NEAREST_EXACT)
    resized = resize_label_mask(mask, (100, 100))
    changed = resized != nearest
    assert np.all(nearest[changed] == 0)     # only background pixels are restored
    assert np.array_equal(resized[(nearest == 255)], nearest[nearest == 255])
    assert np.array_equal(resized[12:62, 15:75], nearest[12:62, 15:75])  # large blob interior
    assert _components(resized, 2) >= 1 and _components(resized, 1) >= 2


def test_upscale_is_plain_pixel_centre_nearest():
    from dataset import resize_label_mask

    mask = np.random.default_rng(0).integers(0, 3, (17, 23), dtype=np.uint8)
    assert np.array_equal(resize_label_mask(mask, (68, 92)),
                          cv2.resize(mask, (92, 68), interpolation=cv2.INTER_NEAREST_EXACT))


# ── SegmentationDataset cache and pixel counts ─────────────────

def _pair(root, mask, name="a"):
    from PIL import Image

    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full(mask.shape, 90, np.uint8)).save(root / "images" / f"{name}.png")
    Image.fromarray(mask, "L").save(root / "masks" / f"{name}.png")


def test_mask_cache_follows_file_changes_and_counts_skip_ignored_pixels(tmp_path):
    from dataset import SegmentationDataset

    mask = np.zeros((64, 64), np.uint8)
    mask[:8] = 1
    mask[60:] = 255
    _pair(tmp_path, mask)
    dataset = SegmentationDataset(str(tmp_path / "images"), str(tmp_path / "masks"), input_size=(32, 32),
                                  in_channels=1, is_train=False, num_classes=3)
    # 64→32: rows 0-7 → 4 output rows of class 1, rows 60-63 → 2 rows of 255 (not counted)
    assert dataset.class_pixel_counts(3) == [32 * 26, 32 * 4, 0]
    first = dataset[0][1]
    first[:] = 2  # callers get a copy; the cache must not change
    assert int((dataset[0][1] == 1).sum()) == 32 * 4

    mask[:8] = 2  # relabel and push the mtime forward (same-size file)
    _pair(tmp_path, mask)
    stat = os.stat(tmp_path / "masks" / "a.png")
    os.utime(tmp_path / "masks" / "a.png", ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    assert dataset.class_pixel_counts(3) == [32 * 26, 0, 32 * 4]


def test_pixel_counts_reject_labels_outside_the_class_range(tmp_path):
    from dataset import SegmentationDataset

    mask = np.zeros((16, 16), np.uint8)
    mask[:4] = 5
    _pair(tmp_path, mask)
    dataset = SegmentationDataset(str(tmp_path / "images"), str(tmp_path / "masks"), input_size=(16, 16),
                                  in_channels=1, is_train=False, num_classes=3)
    with pytest.raises(ValueError, match="범위"):
        dataset.class_pixel_counts(3)


# ── SegmentationLoss ───────────────────────────────────────────

def test_ce_mode_matches_plain_cross_entropy_with_ignore():
    from torch.nn import functional as F
    from segmentation_loss import SegmentationLoss

    logits = torch.randn(2, 3, 8, 8)
    targets = torch.randint(0, 3, (2, 8, 8))
    targets[0, :2] = 255
    expected = F.cross_entropy(logits, targets, ignore_index=255)
    assert torch.allclose(SegmentationLoss(3, loss="ce")(logits, targets), expected)


def test_dice_term_rewards_finding_a_tiny_defect_that_ce_barely_notices():
    from segmentation_loss import SegmentationLoss

    targets = torch.zeros(1, 32, 32, dtype=torch.long)
    targets[0, 10:12, 10:12] = 1  # 4 of 1024 pixels
    missed = torch.zeros(1, 2, 32, 32)
    missed[:, 0] = 4.0            # predicts background everywhere
    found = missed.clone()
    found[0, 1, 10:12, 10:12] = 8.0
    ce, dice = SegmentationLoss(2, loss="ce"), SegmentationLoss(2, loss="ce_dice")
    ce_gain = float(ce(missed, targets) - ce(found, targets))
    dice_gain = float(dice(missed, targets) - dice(found, targets))
    assert dice_gain > 10 * ce_gain  # the Dice term does not shrink with the defect's area


def test_ignored_pixels_are_excluded_from_the_dice_term():
    from segmentation_loss import SegmentationLoss

    loss = SegmentationLoss(2, loss="ce_dice")
    targets = torch.zeros(1, 8, 8, dtype=torch.long)
    targets[0, :4] = 1
    logits = torch.zeros(1, 2, 8, 8)
    logits[0, 1, :4] = 10.0
    logits[0, 0, 4:] = 10.0
    excluded = targets.clone()
    excluded[0, 6:] = 255
    noisy = logits.clone()
    noisy[0, 1, 6:] = 50.0  # wrong prediction only where the label says "ignore"
    assert torch.allclose(loss(noisy, excluded), loss(logits, excluded), atol=1e-6)


def test_a_batch_without_any_trainable_pixel_gives_a_zero_finite_loss():
    from segmentation_loss import SegmentationLoss

    logits = torch.randn(1, 3, 4, 4, requires_grad=True)
    value = SegmentationLoss(3, loss="ce_dice")(logits, torch.full((1, 4, 4), 255))
    value.backward()
    assert float(value.detach()) == 0.0 and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("kwargs", [
    {"loss": "focal"}, {"weights": [1.0, 2.0]}, {"weights": [1.0, 0.0, 1.0]},
    {"weights": [1.0, float("nan"), 1.0]},
])
def test_loss_settings_are_validated(kwargs):
    from segmentation_loss import SegmentationLoss

    with pytest.raises(ValueError):
        SegmentationLoss(3, **kwargs)


# ── train_builtin wiring ───────────────────────────────────────

def _dataset(root, *, with_dent=True):
    from PIL import Image

    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "masks" / split).mkdir(parents=True, exist_ok=True)
        for index in range(2):
            mask = np.zeros((32, 32), np.uint8)
            mask[4:10, 4:20] = 1
            if with_dent:
                mask[20:26, 8:14] = 2
            Image.fromarray((rng.random((32, 32, 3)) * 255).astype(np.uint8)).save(
                root / "images" / split / f"{index}.png")
            Image.fromarray(mask, "L").save(root / "masks" / split / f"{index}.png")
    return root


def _train(data, output, **kwargs):
    from train_builtin import train_builtin

    return train_builtin(
        "unet_resnet18", data, num_classes=3, input_size=32, epochs=1, batch_size=2,
        learning_rate=1e-3, weight_decay=0.0, scheduler_name="none", horizontal_flip=0.0,
        rotation=0.0, color_jitter=0.0, freeze_backbone=True, output_dir=output,
        class_names=["background", "scratch", "dent"], log=lambda *_: None, **kwargs)


def test_training_records_the_loss_and_pixel_class_weights(tmp_path):
    best = _train(_dataset(tmp_path / "data"), tmp_path / "run",
                  segmentation_loss="ce_dice", class_weights="sqrt")
    config = json.loads((best.parent / "training.json").read_text(encoding="utf-8"))["training_config"]
    assert config["segmentation_loss"] == "ce_dice" and config["class_weights"] == "sqrt"
    counts, weights = config["class_pixel_counts"], config["class_weight_values"]
    assert counts == [2 * (1024 - 96 - 36), 2 * 96, 2 * 36]
    assert weights[2] > weights[1] > weights[0]  # rarer class, larger weight
    # Normalised so the average weight over training pixels is 1 (loss scale unchanged).
    assert np.isclose(np.dot(counts, weights) / sum(counts), 1.0)


def test_class_weights_preserve_absent_class_channels(tmp_path):
    best = _train(_dataset(tmp_path / "data", with_dent=False), tmp_path / "run", class_weights="balanced")
    config = json.loads((best.parent / "training.json").read_text())["training_config"]
    assert config['class_pixel_counts'][2] == 0
    assert config['class_weight_values'][2] == 1.0
    assert len(config['class_weight_values']) == 3


def test_default_training_loss_is_ce_dice(tmp_path):
    from core.project import TrainingConfig

    assert TrainingConfig().segmentation_loss == "ce_dice"
    best = _train(_dataset(tmp_path / "data"), tmp_path / "run")
    config = json.loads((best.parent / "training.json").read_text(encoding="utf-8"))["training_config"]
    assert config["segmentation_loss"] == "ce_dice" and config["class_weights"] == "none"


# ── annotation: background-prepared mask documents ─────────────

def test_assigned_object_ids_keep_the_render_cache():
    from core.mask_annotations import MaskDocument

    document = MaskDocument(32, 24, 2, shapes=[{"kind": "polygon", "class_id": 1,
                                                "points": [.2, .2, .8, .2, .8, .8, .2, .8]}])
    document.ensure_object_ids("ns")
    document.render(copy_pixels=False)
    assert document._cache_valid
    document.ensure_object_ids("ns")  # second call (the workspace opening) changes nothing
    assert document._cache_valid


def test_mask_prefetch_is_used_only_while_classes_and_files_are_unchanged(tmp_path):
    pytest.importorskip("PySide6")
    from PIL import Image
    from core.project import ProjectManager
    from widgets.dataset_widget import _MaskDocumentPrefetcher

    project = ProjectManager.create_new("seg", "segment", str(tmp_path), ["background", "part"])
    paths = []
    for index in range(2):
        path = os.path.join(project.data.train_dir, f"{index}.png")
        Image.new("L", (40, 30), 20 * index).save(path)
        paths.append(path)
    prefetcher = _MaskDocumentPrefetcher(project, "train")
    try:
        prefetcher.request(paths[1])
        assert prefetcher.take(paths[0]) is None           # another image: not used
        prefetcher.request(paths[1])
        document = prefetcher.take(paths[1])
        assert document is not None and document._cache_valid
        assert prefetcher.take(paths[1]) is None           # handed out once

        prefetcher.request(paths[1])
        project.data.class_names.append("dent")            # class added while editing
        assert prefetcher.take(paths[1]) is None

        prefetcher.request(paths[1])
        Image.new("L", (41, 30), 0).save(paths[1])         # image replaced
        assert prefetcher.take(paths[1]) is None
    finally:
        prefetcher.close()


def test_custom_csp_segmentation_uses_the_same_loss_and_pixel_weights(tmp_path):
    """The legacy Custom CSP engine (gui/core/trainer.py) shares the loss with train_builtin."""
    from types import SimpleNamespace
    from core.project import ProjectManager
    from core.trainer import TrainWorker
    from dataset import SegmentationDataset
    from segmentation_loss import SegmentationLoss

    data = _dataset(tmp_path / "data")
    project = ProjectManager.create_new("seg", "segment", str(tmp_path / "p"), ["background", "scratch", "dent"])
    project.training.class_weights = "balanced"
    worker = TrainWorker(project)
    dataset = SegmentationDataset(str(data / "images" / "train"), str(data / "masks" / "train"),
                                  input_size=(32, 32), in_channels=1, is_train=False, num_classes=3)
    criterion = worker._create_criterion("segment", project.training,
                                         train_loader=SimpleNamespace(dataset=dataset), num_classes=3, device="cpu")
    assert isinstance(criterion, SegmentationLoss) and criterion.loss == "ce_dice"
    details = worker._class_weight_details
    assert details["counts"] == [2 * (1024 - 96 - 36), 2 * 96, 2 * 36] and details["unit"] == "pixels"
    assert torch.allclose(criterion.weight, torch.tensor(details["weights"], dtype=torch.float32))

    project.training.class_weights, project.training.segmentation_loss = "none", "ce"
    plain = TrainWorker(project)._create_criterion("segment", project.training)
    assert plain.loss == "ce" and plain.weight is None


def test_training_log_suggests_class_weights_for_a_very_rare_defect(tmp_path):
    """First-epoch pixel counts (no extra pass over the data) flag a class under 0.1%."""
    from PIL import Image
    from train_builtin import train_builtin

    data = _dataset(tmp_path / "data", with_dent=False)
    for split in ("train", "val"):
        for index in range(2):
            path = data / "masks" / split / f"{index}.png"
            mask = np.asarray(Image.open(path)).copy()
            mask[28, 28] = 2  # 1 of 1024 pixels
            Image.fromarray(mask, "L").save(path)
    messages = []
    train_builtin("unet_resnet18", data, num_classes=3, input_size=32, epochs=1, batch_size=2,
                  scheduler_name="none", horizontal_flip=0.0, rotation=0.0, color_jitter=0.0,
                  freeze_backbone=True, output_dir=tmp_path / "run",
                  class_names=["background", "scratch", "dent"], log=messages.append)
    hints = [m for m in messages if isinstance(m, str) and "희귀 결함 클래스" in m]
    assert len(hints) == 1 and "dent 0.098%" in hints[0] and "scratch" not in hints[0]

    messages.clear()
    train_builtin("unet_resnet18", data, num_classes=3, input_size=32, epochs=1, batch_size=2,
                  scheduler_name="none", horizontal_flip=0.0, rotation=0.0, color_jitter=0.0,
                  freeze_backbone=True, output_dir=tmp_path / "run2", class_weights="sqrt",
                  class_names=["background", "scratch", "dent"], log=messages.append)
    assert not [m for m in messages if isinstance(m, str) and "희귀 결함 클래스" in m]
