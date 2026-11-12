"""New defect shapes, visible controls, and lossless standalone image saving."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from core.segmentation_synthesis import synthesize
from core.segmentation_synthesis_store import SynthesisStore
from core.synthesis_variation import variation_settings, varied_patch
from PIL import Image

from tests.test_segmentation_synthesis import make_fixture
from webapp.storage import read_json


def source(policy, channels):
    mask = np.zeros((64, 64), np.uint8)
    image = np.full((64, 64), 110, np.uint8)
    if policy == "crack":
        mask[12:48, 30] = 1
        image[mask > 0] = 60
    else:
        mask[20:36, 25:39] = 2
        image[mask > 0] = 160
    if channels == 3:
        image = np.repeat(image[..., None], 3, axis=2)
        image[..., 1] += 15
    return image, mask


@pytest.mark.parametrize("policy", ["crack", "material"])
@pytest.mark.parametrize("channels", [1, 3])
def test_new_shapes_are_reproducible_diverse_and_native(policy, channels):
    image, mask = source(policy, channels)
    old_image, old_mask = image.copy(), mask.copy()
    settings = {"size_min": 1, "size_max": 1, "shape_amount": 1}
    shapes = set()
    for seed in range(10):
        patch, truth, recipe = varied_patch(
            image, mask, policy, np.random.default_rng(seed), settings
        )
        again = varied_patch(image, mask, policy, np.random.default_rng(seed), settings)
        np.testing.assert_array_equal(patch, again[0])
        np.testing.assert_array_equal(truth, again[1])
        assert recipe == again[2]
        assert patch.shape[:2] == truth.shape and patch.dtype == np.uint8
        assert patch.ndim == image.ndim
        assert set(np.unique(truth)) == {0, int(mask.max())}
        assert not truth[0].any() and not truth[-1].any()
        assert not truth[:, 0].any() and not truth[:, -1].any()
        binary = (truth > 0).astype(np.uint8)
        assert cv2.connectedComponents(binary, connectivity=8)[0] == 2
        if policy == "crack":
            assert (
                recipe["generator"] == "branched_crack_v1" and recipe["width_px"] >= 1
            )
        else:
            assert recipe["generator"] == "irregular_material_v1"
        shapes.add(hashlib.sha256(truth.tobytes()).hexdigest())
    assert len(shapes) == 10
    np.testing.assert_array_equal(image, old_image)
    np.testing.assert_array_equal(mask, old_mask)


@pytest.mark.parametrize("policy", ["crack", "material"])
@pytest.mark.parametrize("channels", [1, 3])
def test_size_strength_and_region_controls_change_real_output(policy, channels):
    patch, mask = source(policy, channels)
    target = np.full((240, 300) if channels == 1 else (240, 300, 3), 110, np.uint8)
    if channels == 3:
        target[..., 1] += 15
    labels = np.zeros((240, 300), np.uint8)
    labels[:3] = 255
    labels[200:205, 120:130] = 3
    allowed = np.zeros(labels.shape, bool)
    allowed[30:185, 35:265] = True
    outputs = []
    for strength in (0.5, 1.4):
        output, truth, changed, recipe = synthesize(
            target,
            labels,
            patch,
            mask,
            policy,
            np.random.default_rng(5),
            allowed=allowed,
            variation={
                "size_min": 1,
                "size_max": 1,
                "strength_min": strength,
                "strength_max": strength,
            },
        )
        assert output.shape == target.shape
        assert not changed[~allowed].any()
        np.testing.assert_array_equal(truth[labels > 0], labels[labels > 0])
        np.testing.assert_array_equal(output[labels > 0], target[labels > 0])
        assert recipe["strength"] == strength
        foreground = (truth == mask.max()) & (labels == 0)
        outputs.append(
            (foreground, np.abs(output.astype(float) - target)[foreground].mean())
        )
    np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
    assert outputs[1][1] > outputs[0][1] * 2
    areas, spans = [], []
    for scale in (0.6, 1.6):
        _, truth, _ = varied_patch(
            patch,
            mask,
            policy,
            np.random.default_rng(5),
            {"size_min": scale, "size_max": scale},
        )
        areas.append(np.count_nonzero(truth))
        spans.append(max(cv2.boundingRect((truth > 0).astype(np.uint8))[2:]))
    assert areas[1] > areas[0] * 2 and spans[1] > spans[0] * 2


@pytest.mark.parametrize(
    "values",
    [
        {"size_min": 2, "size_max": 1},
        {"strength_min": float("nan")},
        {"shape_amount": 2},
        {"mode": "random"},
        {"size_min": True},
        {"strength_max": 5},
        {"surprise": 1},
        [],
        None,
    ],
)
def test_invalid_variation_is_rejected(values):
    with pytest.raises(ValueError):
        variation_settings(values)


@pytest.mark.parametrize("channels", [1, 3])
def test_structure_variation_keeps_neighbouring_background_registered(channels):
    image, mask = source("structure", channels)
    # Distinct pins/edges: they must not turn into unlabelled shifted duplicates.
    image[3:8, :] = 225
    image[:, 3:8] = 30
    image[53:58, :] = 45
    old_image = image.copy()
    patch, truth, recipe = varied_patch(
        image,
        mask,
        "structure",
        np.random.default_rng(6),
        {"size_min": 1.4, "size_max": 1.4},
    )
    left, top, w, h = recipe["source_box"]
    aligned = patch[top : top + h, left : left + w]
    changed_region = (mask > 0) | (truth[top : top + h, left : left + w] > 0)
    np.testing.assert_array_equal(aligned[~changed_region], image[~changed_region])
    np.testing.assert_array_equal(image, old_image)
    assert np.count_nonzero(truth) > np.count_nonzero(mask)
    assert not truth[0].any() and not truth[-1].any()
    assert not truth[:, 0].any() and not truth[:, -1].any()


def test_save_new_png_pairs_twice_keeps_edited_truth_and_originals(tmp_path):
    project = make_fixture(tmp_path / "project")
    originals = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(project.data.root).rglob("*.png")
    }
    store = SynthesisStore(project)
    profile = store.analyze()
    result = store.generate(profile["profile_id"], {"1": 2, "2": 2}, seed=23)
    assert result["generated"] == 4 and not result["failures"]
    batch = store.load("batches", result["batch_id"])
    assert batch["variation"]["mode"] == "new_shape"
    row = batch["samples"][0]
    directory = store.directory("batches", batch["id"])
    truth = np.array(Image.open(store.file(directory, row["mask"])))
    truth[185:188, 205:208] = 2
    store.review(batch["id"], row["id"], "pending", mask=truth)
    saved = []
    for _ in range(2):
        output = store.save_images(batch["id"], [row["id"]], str(tmp_path / "output"))
        root = Path(output["export_dir"])
        report = read_json(root / "manifest.json")
        item = report["samples"][0]
        assert output["saved_images"] == 1 and item["revision"] == 1
        assert report["classes"] == project.data.class_names and report["seed"] == 23
        assert report["split"] == "train" and item["status"] == "pending"
        np.testing.assert_array_equal(
            np.array(Image.open(root / item["image"]["path"])),
            np.array(Image.open(store.file(directory, row["image"]))),
        )
        np.testing.assert_array_equal(
            np.array(Image.open(root / item["mask"]["path"])), truth
        )
        saved.append(root / item["image"]["path"])
    assert saved[0] != saved[1] and saved[0].name != saved[1].name
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == expected
        for path, expected in originals.items()
    )


def test_save_cancellation_and_validation_leave_no_partial_export(tmp_path):
    store = SynthesisStore(make_fixture(tmp_path / "project"))
    result = store.generate(
        store.analyze()["profile_id"], {"1": 1}, seed=23, split="val"
    )
    batch = store.load("batches", result["batch_id"])
    ids = [batch["samples"][0]["id"]]
    parent = tmp_path / "output"

    class CancelAfterCopy:
        notified = False

        def emit(self, *_):
            self.notified = True

        def cancelled(self):
            return self.notified

    with pytest.raises(InterruptedError):
        store.save_images(batch["id"], ids, parent, context=CancelAfterCopy())
    assert not list(parent.iterdir())
    for invalid in ([], ids * 2, ["absent"]):
        with pytest.raises(ValueError):
            store.save_images(batch["id"], invalid, parent)
    assert not list(parent.iterdir())
    output = store.save_images(batch["id"], ids, parent)
    report = read_json(Path(output["manifest_path"]))
    assert report["split"] == "val"
    assert report["samples"][0]["mask"]["path"].startswith("masks/val/")
