"""Native masks, split isolation, persistent review and actual dataset use."""

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT / "python")]
from core.project import ProjectManager
from core.segmentation_synthesis import (
    default_policy,
    extract_regions,
    synthesize,
    validate_pair,
)
from core.segmentation_synthesis_store import SynthesisStore

CLASSES = ["background", "crack", "foreign material", "shift", "double", "flip"]


def make_fixture(directory, size=(240, 480)):
    project = ProjectManager.create_new("검사", "segment", str(directory), CLASSES)
    h, w = size
    for split_index, split in enumerate(("train", "val", "test")):
        for index in range(2):
            # Distinct images, same periodic layout. Normal target has no defects.
            image = np.full((h, w, 3), 80 + split_index * 5 + index, np.uint8)
            mask = np.zeros((h, w), np.uint8)
            image[:3] = 50 + index + split_index
            mask[:3] = 255
            if index == 0:
                for cid in range(1, 6):
                    x, y = 30 + 80 * (cid - 1), 80
                    if cid == 1:
                        image[y : y + 20, x] = 20
                        mask[y : y + 20, x] = cid
                    elif cid == 2:
                        cv2.circle(image, (x + 5, y + 5), 5, (210, 90, 30), -1)
                        cv2.circle(mask, (x + 5, y + 5), 5, cid, -1)
                    elif cid == 4:
                        image[y : y + 9, x : x + 8] = 180
                        image[y + 11 : y + 20, x : x + 8] = 160
                        mask[y : y + 9, x : x + 8] = cid
                        mask[y + 11 : y + 20, x : x + 8] = cid
                    else:
                        image[y : y + 16, x : x + 10] = [140, 160, 200]
                        mask[y : y + 16, x : x + 10] = cid
            Image.fromarray(image).save(
                Path(getattr(project.data, split + "_dir")) / f"{index}.png"
            )
            Image.fromarray(mask).save(
                Path(project.data.root) / "masks" / split / f"{index}.png"
            )
    ProjectManager.save(project)
    return project


@pytest.mark.parametrize("cid", range(1, 6))
def test_all_classes_preserve_labels_pixels_and_seed(tmp_path, cid):
    project = make_fixture(tmp_path)
    image = np.array(Image.open(Path(project.data.train_dir) / "0.png"))
    labels = np.array(Image.open(Path(project.data.root) / "masks/train/0.png"))
    policy = default_policy(CLASSES[cid])
    regions = extract_regions(labels, cid, policy)
    assert len(regions) == 1  # double's disconnected pieces form one example
    region = regions[0]
    x, y, w, h = region["box"]
    target = np.array(Image.open(Path(project.data.train_dir) / "1.png"))
    target_mask = np.zeros(labels.shape, np.uint8)
    target_mask[:3] = 255
    target_mask[170:175, 200:210] = 5
    first = synthesize(
        target,
        target_mask,
        image[y : y + h, x : x + w],
        region["mask"],
        policy,
        np.random.default_rng(4),
    )
    second = synthesize(
        target,
        target_mask,
        image[y : y + h, x : x + w],
        region["mask"],
        policy,
        np.random.default_rng(4),
    )
    for a, b in zip(first[:3], second[:3]):
        np.testing.assert_array_equal(a, b)
    output, truth, changed, recipe = first
    assert output.shape == target.shape and output.dtype == np.uint8
    assert changed.any() and np.any((truth == cid) & (target_mask == 0))
    np.testing.assert_array_equal(
        truth[target_mask != 0], target_mask[target_mask != 0]
    )
    np.testing.assert_array_equal(output[target_mask != 0], target[target_mask != 0])
    np.testing.assert_array_equal(changed != 0, np.any(output != target, axis=2))
    if cid in (1, 3, 4, 5):
        assert recipe["angle"] == 0 and recipe["scale"] == 1
        assert np.count_nonzero((truth == cid) & (target_mask == 0)) == region["pixels"]
    if cid == 1:
        assert recipe["thin_preserved"]


def test_store_generate_review_edit_publish_and_reopen(tmp_path):
    project = make_fixture(tmp_path / "source")
    originals = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(project.data.root).rglob("*.png")
    }
    store = SynthesisStore(project)
    profile = store.analyze()
    assert profile["donors"] == 10
    result = store.generate(
        profile["profile_id"], {str(i): 1 for i in range(1, 6)}, seed=123
    )
    assert result["generated"] == 5 and not result["failures"]
    batch = store.load("batches", result["batch_id"])
    with pytest.raises(ValueError, match="승인"):
        store.publish(batch["id"])
    for row in batch["samples"]:
        store.review(batch["id"], row["id"], "approved", expected_revision=0)
    row = batch["samples"][0]
    with pytest.raises(ValueError, match="다른 창"):
        store.review(batch["id"], row["id"], "rejected", expected_revision=0)
    folder = store.directory("batches", batch["id"])
    mask = np.array(Image.open(store.file(folder, row["mask"])))
    mask[180:185, 300:310] = 2
    updated = store.review(
        batch["id"], row["id"], "approved", mask=mask, expected_revision=1
    )
    assert updated["status"] == "pending"
    store.review(batch["id"], row["id"], "approved", expected_revision=2)
    saved = SynthesisStore(project).publish(batch["id"])
    restored = ProjectManager.load(saved["project_path"])
    assert restored.data.class_names == CLASSES
    assert len(list(Path(restored.data.train_dir).glob("*.png"))) == 7
    assert len(list(Path(restored.data.val_dir).glob("*.png"))) == 2
    assert len(list(Path(restored.data.test_dir).glob("*.png"))) == 2
    for path, expected in originals.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    np.testing.assert_array_equal(
        np.array(
            Image.open(Path(restored.data.root) / "masks/train/synthetic-000000.png")
        ),
        mask,
    )
    assert not restored.runs


def test_val_never_published_train_and_test_not_donors(tmp_path):
    store = SynthesisStore(make_fixture(tmp_path))
    profile = store.analyze()
    p = store.load("profiles", profile["profile_id"])
    assert all(d["split"] != "test" for d in p["donors"])
    result = store.generate(p["id"], {"1": 1}, split="val")
    assert result["generated"] == 1
    store.review(result["batch_id"], "000000", "approved")
    with pytest.raises(ValueError, match="Val"):
        store.publish(result["batch_id"])


def test_cross_split_duplicate_invalid_mask_and_no_examples(tmp_path):
    project = make_fixture(tmp_path)
    source = Path(project.data.train_dir) / "1.png"
    target = Path(project.data.val_dir) / "1.png"
    original = target.read_bytes()
    target.write_bytes(source.read_bytes())
    store = SynthesisStore(project)
    with pytest.raises(ValueError, match="중복"):
        store.analyze()
    assert store.list("profiles") == []
    target.write_bytes(original)
    mask_path = Path(project.data.root) / "masks/train/0.png"
    good_mask = mask_path.read_bytes()
    image = np.array(Image.open(mask_path))
    image[0, 0] = 77
    Image.fromarray(image).save(mask_path)
    with pytest.raises(ValueError, match="클래스"):
        store.analyze()
    mask_path.write_bytes(good_mask)
    profile = store.analyze()
    with pytest.raises(ValueError, match="정답"):
        store.generate(profile["profile_id"], {"66": 1})


def test_cancel_and_file_tampering_no_partial_publication(tmp_path):
    store = SynthesisStore(make_fixture(tmp_path))

    class Cancel:
        def cancelled(self):
            return True

    with pytest.raises(InterruptedError):
        store.analyze(context=Cancel())
    assert not list((store.root / "profiles").iterdir())
    profile = store.analyze()
    with pytest.raises(InterruptedError):
        store.generate(profile["profile_id"], {"1": 1}, context=Cancel())
    assert not list((store.root / "batches").iterdir())
    generated = store.generate(profile["profile_id"], {"1": 1})
    store.review(generated["batch_id"], "000000", "approved")
    with pytest.raises(InterruptedError):
        store.publish(generated["batch_id"], context=Cancel())
    assert not list((store.root / "publications").iterdir())
    batch = store.load("batches", generated["batch_id"])
    path = (
        store.directory("batches", batch["id"]) / batch["samples"][0]["image"]["path"]
    )
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="손상"):
        store.publish(batch["id"])
    assert not list((store.root / "publications").iterdir())


def test_allowed_region_gray_and_no_space():
    image = np.full((100, 120), 120, np.uint8)
    labels = np.zeros(image.shape, np.uint8)
    patch = np.full((30, 30), 120, np.uint8)
    patch[10:20, 14] = 20
    mask = np.zeros(patch.shape, np.uint8)
    mask[10:20, 14] = 1
    allowed = np.zeros(image.shape, bool)
    allowed[35:80, 50:100] = True
    result = synthesize(
        image, labels, patch, mask, "crack", np.random.default_rng(2), allowed=allowed
    )
    assert not np.any((result[2] > 0) & ~allowed)
    with pytest.raises(ValueError, match="位置|위치"):
        synthesize(
            image,
            np.ones(image.shape, np.uint8),
            patch,
            mask,
            "crack",
            np.random.default_rng(2),
        )
    with pytest.raises(ValueError, match="8bit"):
        validate_pair(image.astype(np.uint16), labels, 2)


def test_published_dataset_trains_and_exports(tmp_path):
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    ort.disable_telemetry_events()
    from export_onnx import export_checkpoint
    from onnx_session import cpu_session_options
    from train_builtin import train_builtin

    store = SynthesisStore(make_fixture(tmp_path / "source"))
    profile = store.analyze()
    batch = store.generate(profile["profile_id"], {"1": 1, "2": 1})
    for row in store.load("batches", batch["batch_id"])["samples"]:
        store.review(batch["batch_id"], row["id"], "approved")
    published = store.publish(batch["batch_id"])
    project = ProjectManager.load(published["project_path"])
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        best = train_builtin(
            "unet_lite",
            project.data.root,
            num_classes=6,
            in_channels=3,
            input_size=64,
            epochs=1,
            batch_size=2,
            num_workers=0,
            patch_training=True,
            patches_per_image=2,
            horizontal_flip=0,
            rotation=0,
            class_names=CLASSES,
            output_dir=tmp_path / "train",
            log=lambda _: None,
        )
        assert best.is_file()
        output = tmp_path / "model.onnx"
        export_checkpoint(best, output, log=lambda _: None)
        session = ort.InferenceSession(
            str(output),
            cpu_session_options(num_threads=2),
            providers=["CPUExecutionProvider"],
        )
        prediction = session.run(
            None,
            {
                session.get_inputs()[0].name: np.zeros(
                    tuple(session.get_inputs()[0].shape), np.float32
                )
            },
        )[0]
        assert prediction.shape[:2] == (1, 6) and np.isfinite(prediction).all()
    finally:
        torch.set_num_threads(old)


def test_unlabelled_normal_opt_in_and_frozen_profile(tmp_path):
    project = make_fixture(tmp_path)
    normal_mask = Path(project.data.root) / "masks/train/1.png"
    normal_mask.unlink()
    store = SynthesisStore(project)
    first = store.analyze()
    assert store.load("profiles", first["profile_id"])["include_unlabelled"] is False
    second = store.analyze(include_unlabelled=True)
    result = store.generate(second["profile_id"], {"1": 1})
    batch = store.load("batches", result["batch_id"])
    profile = store.load("profiles", second["profile_id"])
    chosen = next(
        v for v in profile["images"] if v["id"] == batch["samples"][0]["target_id"]
    )
    assert not chosen["annotated"]
    # A frozen profile remains usable after source files move, never mixes
    # replacement pixels with old masks or changes a reviewed result.
    for path in Path(project.data.train_dir).glob("*.png"):
        path.unlink()
    again = store.generate(second["profile_id"], {"1": 1})
    assert again["generated"] == 1
    store.review(again["batch_id"], "000000", "approved")
    saved = store.publish(again["batch_id"])
    synthetic_project = ProjectManager.load(saved["project_path"])
    with pytest.raises(ValueError, match="실제 원본"):
        SynthesisStore(synthetic_project).analyze()


def test_ignored_other_defect_context_and_empty_mask_rejected(tmp_path):
    mask = np.zeros((100, 100), np.uint8)
    mask[40:50, 40:50] = 1
    mask[45:50, 52:56] = 2
    assert extract_regions(mask, 1, "structure") == []
    mask[45:50, 52:56] = 255
    assert extract_regions(mask, 1, "structure") == []
    store = SynthesisStore(make_fixture(tmp_path))
    profile = store.analyze()
    generated = store.generate(profile["profile_id"], {"1": 1})
    mask = np.zeros((240, 480), np.uint8)
    store.review(generated["batch_id"], "000000", "pending", mask=mask)
    with pytest.raises(ValueError, match="비어"):
        store.review(generated["batch_id"], "000000", "approved")
    with pytest.raises(ValueError, match="ID"):
        store.load("batches", "../../x")


def test_saved_roi_and_class_mapping_change_guard(tmp_path):
    store = SynthesisStore(make_fixture(tmp_path))
    profile = store.analyze()
    allowed = np.zeros((240, 480), np.uint8)
    allowed[120:230, 270:470] = 255
    file = tmp_path / "allowed.png"
    Image.fromarray(allowed).save(file)
    result = store.generate(profile["profile_id"], {"1": 1}, allowed_path=str(file))
    assert result["generated"] == 1
    data = store.load("batches", result["batch_id"])
    folder = store.directory("batches", data["id"])
    assert store.file(folder, data["allowed"]).is_file()
    changed = np.array(Image.open(store.file(folder, data["samples"][0]["changed"])))
    assert not np.any((changed > 0) & (allowed == 0))
    store.project.data.class_names[1] = "renamed"
    with pytest.raises(ValueError, match="클래스"):
        store.load("profiles", profile["profile_id"])


def test_thin_branch_survives_wide_junction_and_new_classes_can_reanalyze(tmp_path):
    from core.segmentation_synthesis import transformed_patch

    mask = np.zeros((60, 60), np.uint8)
    mask[20:40, 20:40] = 1
    mask[30, 40:55] = 1
    patch = np.full((60, 60, 3), 100, np.uint8)
    patch[mask == 1] = 20
    _, result, recipe = transformed_patch(
        patch, mask, "crack", np.random.default_rng(4)
    )
    np.testing.assert_array_equal(mask, result)
    assert recipe["thin_preserved"]
    project = make_fixture(tmp_path)
    store = SynthesisStore(project)
    store.analyze()
    project.data.class_names[2] = "FM-new-name"
    new = store.analyze()
    assert [p["id"] for p in store.list("profiles")] == [new["profile_id"]]
