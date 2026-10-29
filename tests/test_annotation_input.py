"""Annotation workspace input: Space panning from any control, brush continuation, unlabeled images."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QLineEdit, QPushButton

from core.mask_annotations import MaskDocument
from widgets.segmentation_annotation import SegmentationAnnotationDialog


@pytest.fixture
def workspace(tmp_path):
    app = QApplication.instance() or QApplication([])
    image = tmp_path / "a.png"
    Image.new("RGB", (200, 160), 90).save(image)
    dialog = SegmentationAnnotationDialog(str(image), ["background", "scratch", "dent"],
                                          MaskDocument(200, 160, 3), lambda values: {})
    dialog.show()
    app.processEvents()
    dialog._fit()
    app.processEvents()
    yield app, dialog
    dialog.dirty = False
    dialog.reject()
    dialog.deleteLater()
    app.processEvents()


def _drag(app, dialog, points):
    viewport = dialog.view.viewport()
    mapped = [dialog.view.mapFromScene(x, y) for x, y in points]
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, mapped[0])
    for start, end in zip(mapped, mapped[1:]):
        for step in range(1, 6):
            QTest.mouseMove(viewport, start + (end - start) * step / 5)
            app.processEvents()
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, mapped[-1])
    app.processEvents()


# ── Space ──────────────────────────────────────────────────────

def test_space_pans_the_canvas_instead_of_pressing_the_focused_control(workspace):
    """Space used to click whichever button/checkbox/combo last took the keyboard focus."""
    app, dialog = workspace
    controls = dialog.findChildren(QPushButton) + dialog.findChildren(QCheckBox) + dialog.findChildren(QComboBox)
    assert controls and all(c.focusPolicy() == Qt.FocusPolicy.NoFocus for c in controls)
    check = dialog.empty_confirm
    check.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # worst case: a control holds the focus anyway
    check.setFocus()
    app.processEvents()
    before = check.isChecked()
    QTest.keyPress(check, Qt.Key.Key_Space)
    app.processEvents()
    assert check.isChecked() == before and dialog.view.space and app.focusWidget() is dialog.view
    QTest.keyRelease(dialog.view, Qt.Key.Key_Space)
    assert not dialog.view.space


def test_text_fields_still_type_spaces(workspace):
    app, dialog = workspace
    search = next(e for e in dialog.findChildren(QLineEdit) if e.placeholderText() == "클래스 검색")
    search.setFocus()
    app.processEvents()
    QTest.keyClicks(search, "a b")
    assert search.text() == "a b" and not dialog.view.space


# ── brush continuation ─────────────────────────────────────────

def test_brush_after_a_rectangle_starts_a_new_object(workspace):
    """A brush stroke used to join the just-drawn box, so the touching parts could never be split."""
    app, dialog = workspace
    dialog.class_combo.setCurrentIndex(dialog.class_combo.findData(1))
    dialog._set_mode("rectangle")
    _drag(app, dialog, [(20, 20), (80, 70)])
    dialog._set_mode("brush")
    _drag(app, dialog, [(70, 60), (150, 120)])
    kinds = sorted(sorted({s["kind"] for i, s in enumerate(dialog.document.shapes)
                           if dialog.document.key(i) == dialog._key(index)}) for index in dialog._indices())
    assert kinds == [["polygon"], ["stroke"]]


def test_consecutive_brush_strokes_still_paint_one_object_that_can_be_split(workspace):
    from unittest.mock import patch
    from PySide6.QtWidgets import QDialog

    app, dialog = workspace
    dialog.class_combo.setCurrentIndex(dialog.class_combo.findData(1))
    dialog._set_mode("brush")
    _drag(app, dialog, [(20, 20), (60, 20)])
    _drag(app, dialog, [(20, 120), (60, 120)])
    assert len(dialog._indices()) == 1
    dialog._select(dialog._indices()[0])
    with patch.object(QDialog, "exec", lambda self: 1), patch.object(QDialog, "result", lambda self: 1):
        dialog._split_components()
    assert len(dialog._indices()) == 2


# ── unlabeled images ───────────────────────────────────────────

def test_training_includes_images_without_masks_as_background_and_says_so(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    from spatial_data import labeled_semantic_pairs, semantic_pairs
    from dataset import SegmentationDataset, unlabeled_message

    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for index in range(3):
        Image.new("L", (16, 16), 40).save(images / f"{index}.png")
    Image.fromarray(np.zeros((16, 16), np.uint8), "L").save(masks / "0.png")  # background only, confirmed
    Image.fromarray(np.ones((16, 16), np.uint8), "L").save(masks / "1.png")
    pairs, unlabeled = labeled_semantic_pairs(images, masks)
    assert [Path(p).stem for p, _ in pairs] == ["0", "1"] and [Path(p).stem for p in unlabeled] == ["2"]
    with pytest.raises(ValueError, match="마스크 누락"):  # the strict helper is unchanged
        semantic_pairs(images, masks)
    dataset = SegmentationDataset(str(images), str(masks), input_size=(16, 16), in_channels=1,
                                  is_train=False, num_classes=2)
    assert len(dataset) == 3 and len(dataset.unlabeled) == 1
    assert (dataset[2][1] == 0).all()

    class Loader:
        pass
    loader = Loader()
    loader.dataset = dataset
    assert "train 1장" in unlabeled_message(loader, None)
    (masks / "0.png").unlink()
    (masks / "1.png").unlink()
    background = SegmentationDataset(str(images), str(masks), input_size=(16, 16), in_channels=1,
                                     is_train=False, num_classes=2)
    assert len(background) == 3 and all((background[i][1] == 0).all() for i in range(3))


# ── Shift: click selects, drag erases; a picked class applies to the selection ──

def _shift_click(app, dialog, x, y):
    QTest.mouseClick(dialog.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier,
                     dialog.view.mapFromScene(x, y))
    app.processEvents()


@pytest.fixture
def two_objects(tmp_path):
    app = QApplication.instance() or QApplication([])
    image = tmp_path / "a.png"
    Image.new("RGB", (200, 160), 90).save(image)
    saved = {}
    dialog = SegmentationAnnotationDialog(str(image), ["background", "scratch", "dent"], MaskDocument(200, 160, 3, shapes=[
        {"kind": "polygon", "class_id": 1, "points": [.1, .1, .18, .1, .18, .2, .1, .2], "object_id": "box"},
        {"kind": "stroke", "class_id": 1, "points": [.6, .6, .62, .6], "width": 8, "object_id": "blob"}]),
        lambda values: saved.setdefault("payload", values) or {})
    dialog.show()
    app.processEvents()
    dialog._fit()
    app.processEvents()
    yield app, dialog, saved
    dialog.dirty = False
    dialog.reject()
    dialog.deleteLater()
    app.processEvents()


@pytest.mark.parametrize("mode", ["brush", "rectangle", "polygon", "fill", "select"])
def test_shift_click_selects_in_every_tool_and_never_erases(two_objects, mode):
    """Shift+click in the brush tool used to erase a brush dot, so small boxes/blobs vanished."""
    import time

    app, dialog, saved = two_objects
    dialog._set_mode(mode)
    before = dialog.document.render().copy()
    _shift_click(app, dialog, 28, 24)
    _shift_click(app, dialog, 122, 96)
    assert np.array_equal(before, dialog.document.render())
    assert len(dialog.selected_indices) == 2 and not getattr(dialog.view, "points", [])
    dialog.class_list.setCurrentRow(2)  # pick "dent": applies to the user's selection
    app.processEvents()
    dialog._save()
    deadline = time.time() + 10
    while dialog.worker is not None and time.time() < deadline:
        app.processEvents()
    app.processEvents()
    pixels = MaskDocument.from_payload(saved["payload"], 200, 160, 3).render()
    assert pixels[24, 28] == 2 and pixels[96, 122] == 2


def test_shift_drag_still_erases_with_the_brush(two_objects):
    app, dialog, _saved = two_objects
    dialog._set_mode("brush")
    viewport = dialog.view.viewport()
    start, end = dialog.view.mapFromScene(20, 24), dialog.view.mapFromScene(36, 24)
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, start)
    for step in range(1, 9):
        QTest.mouseMove(viewport, start + (end - start) * step / 8)
        app.processEvents()
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, end)
    app.processEvents()
    assert dialog.document.shapes[-1].get("operation") == "erase"


def test_picking_the_next_class_does_not_relabel_the_object_just_drawn(workspace):
    app, dialog = workspace
    dialog.class_combo.setCurrentIndex(dialog.class_combo.findData(1))
    dialog._set_mode("rectangle")
    _drag(app, dialog, [(150, 20), (180, 50)])
    drawn = dialog.selected
    dialog.class_list.setCurrentRow(2)
    app.processEvents()
    assert dialog._class_id(drawn) == 1 and dialog._active_class_id() == 2


def test_detection_shift_click_selects_and_never_deletes_boxes(tmp_path):
    from widgets.annotation_workspace import DetectionAnnotationDialog

    app = QApplication.instance() or QApplication([])
    image = tmp_path / "a.png"
    Image.new("RGB", (200, 160), 90).save(image)
    rows = [{"class_id": 0, "coordinates": [.2, .2, .1, .1]}, {"class_id": 0, "coordinates": [.7, .7, .1, .1]}]
    for mode in ("draw", "erase"):
        dialog = DetectionAnnotationDialog(str(image), ["part", "defect"], [dict(r) for r in rows], lambda v: {})
        dialog.show()
        app.processEvents()
        dialog._fit()
        dialog._set_mode(mode)
        _shift_click(app, dialog, 40, 32)
        _shift_click(app, dialog, 140, 112)
        assert len(dialog.rows) == 2 and sorted(dialog.selected_indices) == [0, 1], mode
        dialog.class_list.setCurrentRow(1)
        app.processEvents()
        assert [row["class_id"] for row in dialog.rows] == [1, 1]
        dialog.dirty = False
        dialog.reject()
        dialog.deleteLater()
        app.processEvents()


def test_background_is_never_applied_as_a_class(two_objects):
    """Class 0 is background: 'changing' objects to it erased them (it is the first list entry)."""
    app, dialog, _saved = two_objects
    dialog._set_mode("select")
    _shift_click(app, dialog, 28, 24)
    _shift_click(app, dialog, 122, 96)
    before = dialog.document.render().copy()
    dialog.class_list.setCurrentRow(0)          # pick background in the class list
    dialog._apply_class()                       # ... and press "선택에 적용"
    dialog._change_class(dialog.selected, 0)    # ... or the object table's class combo
    app.processEvents()
    assert np.array_equal(before, dialog.document.render())
    assert "배경" in dialog.status.text() and "배경" in dialog.class_list.item(0).text()
