"""External rename → save → explicit relink → real segmentation loaders."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from core.dataset_editor import EditTransaction, edit_dataset, scan_dataset
from core.dataset_links import orphan_masks, relink_mask, rename_image
from core.mask_annotations import MaskDocument
from core.project import ProjectManager
from PIL import Image
from spatial_data import semantic_training_pairs


@pytest.fixture
def data(tmp_path):
    project = ProjectManager.create_new("Rename", "segment", str(tmp_path / "p"), ["crack", "FM"])
    image = Path(project.data.train_dir) / "nested" / "old.png"
    image.parent.mkdir()
    Image.new("RGB", (32, 24), (90, 100, 110)).save(image)
    pixels = np.zeros((24, 32), np.uint8)
    pixels[5:10, 7:14] = 1
    pixels[15:18, 20:25] = 2
    pixels[0, 0] = 255
    document = MaskDocument(32, 24, 3, base=pixels)
    edit_dataset(project, "mask_annotations", [image], annotations=document.payload())
    mask = Path(project.data.root) / "masks/train/nested/old.png"
    return project, image, mask, pixels


def _rename_external(project, image):
    renamed = image.with_stem("새 이미지")
    image.rename(renamed)
    # This is exactly the reported operation: no label write during Ctrl+S.
    ProjectManager.save(project)
    return renamed


def test_external_rename_save_is_visible_then_relink_preserves_exact_labels(data, tmp_path):
    from crop_dataset import prepare_crop_dataset
    from dataset import PatchSegmentationDataset, SegmentationDataset

    project, image, mask, pixels = data
    original_mask = mask.read_bytes()
    renamed = _rename_external(project, image)
    index = scan_dataset(project)
    assert index["orphan_masks"] == [{"path": str(mask), "name": str(Path("nested/old.png")), "split": "train"}]
    assert index["images"][0]["training_target"] == "unresolved_annotation"
    with pytest.raises(ValueError, match="정답 연결 복구"):
        semantic_training_pairs(project.data.train_dir, mask.parent.parent)
    result = relink_mask(project, renamed, mask)
    restored = mask.with_stem(renamed.stem)
    assert restored.read_bytes() == original_mask  # PNG object metadata included
    assert not mask.exists() and renamed.is_file()
    assert (Path(result["backup_path"]) / "manifest.json").is_file()
    assert not scan_dataset(project)["orphan_masks"]
    dataset = SegmentationDataset(project.data.train_dir, str(restored.parent.parent),
                                  input_size=(24, 32), is_train=False, num_classes=3)
    _, target = dataset[0]
    np.testing.assert_array_equal(target.numpy(), pixels)
    patch_dataset = PatchSegmentationDataset(project.data.train_dir, str(restored.parent.parent),
                                            input_size=(24, 32), is_train=False, num_classes=3,
                                            patch_size=32)
    assert len(patch_dataset) > 0
    assert (patch_dataset[0][1] == 1).any()
    cropped = prepare_crop_dataset(project.data.root, str(tmp_path / "generated"), "segment",
                                   {"width": 28, "height": 20}, num_classes=3)
    pairs, missing = semantic_training_pairs(Path(cropped) / "images/train", Path(cropped) / "masks/train")
    assert len(pairs) == 1 and missing == []


@pytest.mark.parametrize("new_name", ["after", "OLD", "새 이름"])
def test_app_rename_moves_image_and_all_sidecars(data, new_name):
    project, image, mask, _ = data
    label = Path(project.data.root) / "labels/train/nested/old.txt"
    label.parent.mkdir(parents=True)
    label.write_text("1 .2 .2 .4 .2 .4 .4\n")
    image_bytes, mask_bytes = image.read_bytes(), mask.read_bytes()
    rename_image(project, image, new_name)
    assert image.with_stem(new_name).read_bytes() == image_bytes
    assert mask.with_stem(new_name).read_bytes() == mask_bytes
    assert label.with_stem(new_name).read_text() == "1 .2 .2 .4 .2 .4 .4\n"
    assert not orphan_masks(project)
    assert len(semantic_training_pairs(project.data.train_dir, mask.parent.parent)[0]) == 1


@pytest.mark.parametrize("invalid", ["../outside", "x/y", "NUL", "trailing.", ""])
def test_invalid_names_change_no_files(data, invalid):
    project, image, mask, _ = data
    before = mask.read_bytes()
    with pytest.raises(ValueError):
        rename_image(project, image, invalid)
    assert image.is_file() and mask.read_bytes() == before


def test_rename_collision_different_image_extension_or_orphan_mask(data):
    project, image, mask, _ = data
    collision = image.with_name("after.jpg")
    Image.new("RGB", (32, 24)).save(collision)
    with pytest.raises(FileExistsError):
        rename_image(project, image, "after")
    collision.unlink()
    mask.with_stem("after").write_bytes(mask.read_bytes())
    with pytest.raises(FileExistsError):
        rename_image(project, image, "after")
    assert image.exists() and mask.exists()


def test_relink_moves_nested_polygon_and_never_overwrites_existing_mask(data):
    project, image, mask, _ = data
    label = Path(project.data.root) / "labels/train/nested/old.txt"
    label.parent.mkdir(parents=True)
    label.write_text("1 .2 .2 .4 .2 .4 .4\n")
    renamed = _rename_external(project, image)
    other = mask.with_stem(renamed.stem)
    other.write_bytes(mask.read_bytes())
    with pytest.raises(FileExistsError, match="이미 정답"):
        relink_mask(project, renamed, mask)
    other.unlink()
    relink_mask(project, renamed, mask)
    assert not label.exists() and label.with_stem(renamed.stem).is_file()


@pytest.mark.parametrize("bad", ["size", "class", "rgb", "linked", "split"])
def test_relink_rejects_wrong_mask_without_losing_source(data, bad):
    project, image, mask, _ = data
    renamed = image if bad == "linked" else _rename_external(project, image)
    if bad == "size":
        Image.new("L", (33, 24)).save(mask)
    elif bad == "class":
        Image.new("L", (32, 24), 3).save(mask)
    elif bad == "rgb":
        Image.new("RGB", (32, 24)).save(mask)
    elif bad == "split":
        destination = Path(project.data.val_dir) / renamed.name
        renamed.rename(destination)
        renamed = destination
    before = mask.read_bytes()
    with pytest.raises(ValueError):
        relink_mask(project, renamed, mask)
    assert mask.read_bytes() == before


@pytest.mark.parametrize("operation", ["rename", "relink"])
def test_write_failure_restores_all_paths_and_project(data, operation):
    project, image, mask, _ = data
    if operation == "relink":
        image = _rename_external(project, image)
    before = {str(p): p.read_bytes() for p in Path(project.project_dir).rglob("*") if p.is_file()}
    with (patch.object(EditTransaction, "finish", side_effect=OSError("disk full")),
          pytest.raises(OSError, match="disk full")):
        if operation == "rename":
            rename_image(project, image, "new")
        else:
            relink_mask(project, image, mask)
    after = {str(p): p.read_bytes() for p in Path(project.project_dir).rglob("*") if p.is_file()}
    assert after == before


def test_detection_rename_keeps_box_label(tmp_path):
    p = ProjectManager.create_new("Detection", "detect", str(tmp_path / "d"), ["part"])
    image = Path(p.data.train_dir) / "old.bmp"
    Image.new("RGB", (20, 10)).save(image)
    edit_dataset(p, "annotations", [image], annotations=[{"class_id": 0, "coordinates": [.5, .5, .2, .2]}])
    label = Path(p.data.root) / "labels/train/old.txt"
    before = label.read_bytes()
    rename_image(p, image, "new")
    assert label.with_stem("new").read_bytes() == before
    assert scan_dataset(p)["images"][0]["annotated"]
