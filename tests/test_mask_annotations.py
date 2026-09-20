"""Editable PNG masks remain lossless training labels and transactional assets."""
import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from core.dataset_editor import edit_dataset
from core.mask_annotations import MaskDocument, METADATA_KEY, load_mask_document
from core.project import ProjectManager


def polygon(class_id=1):
    return {"kind": "polygon", "class_id": class_id,
            "points": [.2, .2, .8, .2, .8, .8, .2, .8]}


@pytest.fixture
def scene(tmp_path):
    project = ProjectManager.create_new("masks", "segment", str(tmp_path / "project"), ["background", "part", "defect"])
    image = Path(project.data.train_dir) / "nested" / "part.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), "gray").save(image)
    return project, image


def test_png_roundtrip_keeps_polygons_holes_ignore_and_external_edits():
    base = np.zeros((80, 100), dtype=np.uint8)
    base[0, 0] = 255
    document = MaskDocument(100, 80, 3, base, [polygon()])
    document.shapes.append({"kind": "stroke", "class_id": 0, "width": 12, "points": [.5, .5]})
    pixels = document.render()
    assert pixels[40, 50] == 0 and pixels[30, 30] == 1 and pixels[0, 0] == 255
    with Image.open(io.BytesIO(document.encode())) as image:
        restored = MaskDocument.from_image(image, 100, 80, 3)
        np.testing.assert_array_equal(np.array(image), pixels)
        metadata = image.info[METADATA_KEY]
    assert restored.shapes == document.shapes and restored.persisted
    restored.shapes[0]["class_id"] = 2
    assert restored.render()[30, 30] == 2
    # An external class remap with stale PNG text must keep the new pixels.
    pixels[30, 30] = 2
    info = PngImagePlugin.PngInfo()
    info.add_text(METADATA_KEY, metadata)
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG", pnginfo=info)
    with Image.open(io.BytesIO(stream.getvalue())) as image:
        external = MaskDocument.from_image(image, 100, 80, 3)
    assert external.shapes == []
    np.testing.assert_array_equal(external.render(), pixels)


def test_save_train_reopen_move_and_restore_on_failed_project_save(scene):
    from dataset import SegmentationDataset
    project, image = scene
    document = load_mask_document(project, image, "train")
    assert not document.persisted
    document.shapes.append(polygon())
    label = Path(project.data.root) / "labels/train/nested/part.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("1 .2 .2 .8 .2 .8 .8 .2 .8\n")
    mask = Path(project.data.root) / "masks/train/nested/part.png"
    edit_dataset(project, "mask_annotations", [str(image)], annotations=document.payload())
    assert not label.exists()
    before = mask.read_bytes()
    restored = load_mask_document(project, image, "train")
    assert restored.shapes == document.shapes
    dataset = SegmentationDataset(project.data.train_dir, str(mask.parents[1]),
                                  input_size=(80, 100), is_train=False, num_classes=3)
    _, target = dataset[0]
    np.testing.assert_array_equal(target.numpy(), document.render())
    restored.shapes[0]["class_id"] = 2
    with patch.object(ProjectManager, "save", side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            edit_dataset(project, "mask_annotations", [str(image)], annotations=restored.payload())
    assert mask.read_bytes() == before
    edit_dataset(project, "move", [str(image)], target_split="val")
    moved = Path(project.data.val_dir) / "nested/part.png"
    assert load_mask_document(project, moved, "val").shapes == document.shapes
    assert not mask.exists()


def test_failed_conversion_restores_legacy_labels_and_no_partial_mask(scene):
    project, image = scene
    label = Path(project.data.root) / "labels/train/nested/part.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("1 .2 .2 .8 .2 .8 .8 .2 .8\n")
    original = label.read_bytes()
    document = load_mask_document(project, image, "train")
    assert len(document.shapes) == 1
    with patch.object(ProjectManager, "save", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            edit_dataset(project, "mask_annotations", [str(image)], annotations=document.payload())
    assert label.read_bytes() == original
    assert not (Path(project.data.root) / "masks/train/nested/part.png").exists()


@pytest.mark.parametrize("shape", [
    dict(polygon(), class_id=3), dict(polygon(), points=[0, 0, 0, 0, 0, 0]),
    dict(polygon(), points=[0, 0, 1, 0, 1, float("nan")]),
    {"kind": "stroke", "class_id": 1, "points": [.5, .5], "width": 0},
])
def test_invalid_annotation_rejected(shape):
    with pytest.raises(ValueError):
        MaskDocument(100, 80, 3, shapes=[shape])


def test_background_only_image_gets_a_training_mask(scene):
    project, image = scene
    document = load_mask_document(project, image, "train")
    edit_dataset(project, "mask_annotations", [str(image)], annotations=document.payload())
    restored = load_mask_document(project, image, "train")
    assert restored.persisted and not restored.render().any()
