"""Real editing, training-pixel preservation, crash-safe drafts and grid contracts."""

import io
import threading
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from core.annotation_tools import (
    AnnotationDraftStore,
    atomic_write,
    grid_boxes,
)
from core.mask_annotations import MaskDocument
from PIL import Image


def polygon(cid=1, **kwargs):
    return dict(
        kind="polygon",
        class_id=cid,
        points=[0.1, 0.1, 0.6, 0.1, 0.6, 0.7, 0.1, 0.7],
        **kwargs,
    )


def test_scoped_erase_and_lock_flags_never_rewrite_existing_pixels():
    doc = MaskDocument(
        100,
        80,
        3,
        shapes=[
            polygon(object_id="first"),
            dict(
                polygon(2, object_id="second"),
                points=[0.4, 0.2, 0.9, 0.2, 0.9, 0.8, 0.4, 0.8],
            ),
        ],
    )
    original = doc.render()
    doc.objects["first"] = {"locked": True, "hidden": True}
    np.testing.assert_array_equal(doc.render(), original)
    erase = {
        "kind": "stroke",
        "class_id": 0,
        "operation": "erase",
        "points": [0.2, 0.4, 0.8, 0.4],
        "width": 12,
        "scope_object": "second",
        "protected_objects": ["first"],
    }
    doc.shapes.append(doc.validate_shape(erase))
    pixels = doc.render()
    assert pixels[32, 20] == 1 and pixels[32, 60] == 0
    doc.objects["first"]["hidden"] = False
    doc.objects["first"]["locked"] = False
    np.testing.assert_array_equal(doc.render(), pixels)
    with Image.open(io.BytesIO(doc.encode())) as image:
        restored = MaskDocument.from_image(image, 100, 80, 3)
    np.testing.assert_array_equal(restored.render(), pixels)
    assert restored.objects == doc.objects and restored.shapes == doc.shapes


def test_raster_holes_and_ignore_survive_save_reopen():
    doc = MaskDocument(100, 80, 3)
    mask = np.zeros((80, 100), bool)
    mask[10:60, 20:70] = True
    mask[25:40, 35:50] = False
    doc.shapes.append(doc.raster_shape(mask, 2, object_id="mask"))
    doc.shapes.append(
        {"kind": "stroke", "class_id": 255, "points": [0.05, 0.05], "width": 5}
    )
    assert (
        doc.render()[30, 40] == 0
        and doc.render()[20, 40] == 2
        and doc.render()[4, 5] == 255
    )
    with Image.open(io.BytesIO(doc.encode())) as image:
        result = MaskDocument.from_image(image, 100, 80, 3)
        np.testing.assert_array_equal(result.render(), doc.render())
    assert result.owner_at(40, 20) == 0


def test_preview_region_equals_full_render_without_replaying_history(monkeypatch):
    doc = MaskDocument(100, 80, 3, shapes=[polygon()])
    doc.render()
    shape = {
        "kind": "stroke",
        "class_id": 0,
        "operation": "erase",
        "scope_object": "shape:0",
        "points": [0.3, 0.3, 0.4, 0.3],
        "width": 7,
    }
    bounds, pixels = doc.preview_region(shape)
    x, y, w, h = bounds
    np.testing.assert_array_equal(pixels, doc.render(shape)[y : y + h, x : x + w])
    assert w < 100 and h < 80


def test_grid_affine_projective_count_bounds_and_anchor_centers():
    seed = [0.1, 0.1, 0.035, 0.025]
    rows = grid_boxes(seed, 8, 16, column_end=[0.8, 0.11], row_end=[0.12, 0.88])
    assert len(rows) == 128 and all(r["valid"] for r in rows)
    assert len({(r["row"], r["column"]) for r in rows}) == 128
    np.testing.assert_allclose(rows[7]["coordinates"][:2], [0.8, 0.11], atol=1e-8)
    projected = grid_boxes(
        seed,
        8,
        16,
        column_end=[0.8, 0.11],
        row_end=[0.12, 0.88],
        far_corner=[0.76, 0.85],
    )
    np.testing.assert_allclose(
        np.asarray(projected[-1]["points"]).reshape(-1, 2).mean(axis=0),
        [0.76, 0.85],
        atol=0.001,
    )
    assert all(r["valid"] for r in projected)
    assert not all(r["valid"] for r in grid_boxes([0.9, 0.9, 0.1, 0.1], 8, 16))
    with pytest.raises(ValueError):
        grid_boxes(seed, 0, 16)
    with pytest.raises(ValueError):
        grid_boxes(seed, 8, 16, column_end=[0.8, 0.1], row_end=[0.8, 0.1])


def test_draft_revision_and_atomic_failure_preserve_source(tmp_path):
    image = tmp_path / "sample.png"
    Image.new("RGB", (20, 20)).save(image)
    store = AnnotationDraftStore(tmp_path / "drafts")
    store.save(
        image,
        "detect",
        "source",
        {"items": []},
        names=["part"],
        pending={"points": [0.1, 0.2]},
    )
    assert store.load(image, "detect", "source", ["part"])["pending"]["points"] == [
        0.1,
        0.2,
    ]
    with pytest.raises(ValueError):
        store.load(image, "detect", "changed", ["part"])
    with pytest.raises(ValueError):
        store.load(image, "detect", "source", ["renamed"])
    file = tmp_path / "label.txt"
    file.write_bytes(b"original")
    with (
        patch("core.annotation_tools.os.replace", side_effect=OSError("disk full")),
        pytest.raises(OSError),
    ):
        atomic_write(file, b"new")
    assert file.read_bytes() == b"original" and not list(tmp_path.glob(".*.tmp"))


pytest.importorskip("PySide6")
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog
from widgets.annotation_workspace import DetectionAnnotationDialog, GridDialog
from widgets.segmentation_annotation import SegmentationAnnotationDialog


@pytest.fixture
def workspace(tmp_path):
    app = QApplication.instance() or QApplication([])
    image = tmp_path / "sample.png"
    Image.new("RGB", (600, 400), "gray").save(image)
    dialogs = []

    def make(task="segment", document=None, navigation=(0, 3), save=None):
        saved = []
        common = {
            "navigation": navigation,
            "draft_store": AnnotationDraftStore(tmp_path / "drafts"),
        }
        if task == "segment":
            dialog = SegmentationAnnotationDialog(
                image,
                ["background", "part", "defect"],
                document or MaskDocument(600, 400, 3),
                save or saved.append,
                **common,
            )
        else:
            dialog = DetectionAnnotationDialog(
                image, ["part", "defect"], [], save or saved.append, **common
            )
        dialogs.append(dialog)
        dialog.show()
        app.processEvents()
        return dialog, saved

    yield app, make
    for dialog in dialogs:
        dialog._draft_timer.stop()
        dialog._closing = True
        dialog._cancel_ai()
        for worker in (dialog.worker, dialog.draft_worker, dialog.ai_worker):
            if worker:
                worker.wait(10000)
        app.processEvents()
        dialog.dirty = False
        dialog.pending = False
        dialog.view.cancel_gesture()
        QDialog.reject(dialog)
        dialog.deleteLater()
    app.processEvents()


def drag(dialog, a, b, modifiers=Qt.KeyboardModifier.NoModifier):
    v = dialog.view
    QTest.mousePress(
        v.viewport(), Qt.MouseButton.LeftButton, modifiers, v.mapFromScene(QPointF(*a))
    )
    QTest.mouseMove(v.viewport(), v.mapFromScene(QPointF(*b)))
    QTest.mouseRelease(
        v.viewport(), Qt.MouseButton.LeftButton, modifiers, v.mapFromScene(QPointF(*b))
    )
    QApplication.processEvents()


def wait(app, dialog, attribute="worker"):
    worker = getattr(dialog, attribute)
    assert worker is not None
    assert worker.wait(10000)
    app.processEvents()


def test_brush_select_group_erase_and_undo(workspace):
    app, make = workspace
    dialog, _ = make()
    dialog._set_mode("brush")
    dialog.brush_size.setValue(20)
    drag(dialog, (100, 100), (160, 100))
    drag(dialog, (130, 100), (170, 110))
    assert len(dialog.document.shapes) == 2 and dialog.table.rowCount() == 1
    dialog._set_mode("select")
    QTest.mouseClick(
        dialog.view.viewport(),
        Qt.MouseButton.LeftButton,
        pos=dialog.view.mapFromScene(QPointF(140, 100)),
    )
    assert dialog.selected == 0
    dialog._set_mode("erase")
    drag(dialog, (140, 100), (140, 100))
    assert dialog.document.render()[100, 140] == 0
    dialog._undo()
    assert dialog.document.render()[100, 140] == 1
    dialog._redo()
    assert dialog.document.render()[100, 140] == 0
    dialog._save()
    wait(app, dialog)
    assert dialog.isVisible() and not dialog.dirty


def test_imported_png_pixel_selection_class_edit_and_save(workspace):
    app, make = workspace
    base = np.zeros((400, 600), np.uint8)
    base[100:200, 120:230] = 1
    dialog, saved = make(document=MaskDocument(600, 400, 3, base))
    dialog._set_mode("select")
    QTest.mouseClick(
        dialog.view.viewport(),
        Qt.MouseButton.LeftButton,
        pos=dialog.view.mapFromScene(QPointF(150, 150)),
    )
    assert dialog.selected == -3
    dialog._change_class(dialog.selected, 2)
    assert dialog.document.render()[150, 150] == 2
    dialog._save()
    wait(app, dialog)
    restored = MaskDocument.from_payload(saved[0], 600, 400, 3)
    np.testing.assert_array_equal(restored.render(), dialog.document.render())


def test_multi_selection_class_delete_lock_hide_and_undo(workspace):
    _app, make = workspace
    dialog, _ = make("detect")
    dialog._create_box([0.2, 0.2, 0.1, 0.1])
    dialog._create_box([0.4, 0.4, 0.1, 0.1])
    dialog._select_many({0, 1})
    dialog.class_combo.setCurrentIndex(1)
    dialog._apply_class()
    assert [r["class_id"] for r in dialog.rows] == [1, 1]
    dialog._set_flag(0, "locked", True)
    dialog._set_flag(1, "hidden", True)
    dialog._select_many({0, 1})
    dialog._remove()
    assert len(dialog.rows) == 1 and dialog.view.locked == {0}
    dialog._undo()
    assert len(dialog.rows) == 2
    dialog._select_many({0, 1})
    dialog._command("nudge:16777236:10")
    assert dialog.rows[0]["coordinates"][0] == 0.2  # locked box cannot move


def test_grid_128_candidates_and_one_step_undo(workspace):
    _app, make = workspace
    dialog, _ = make("detect")
    dialog._create_box([0.06, 0.04, 0.04, 0.025])
    before = list(dialog.rows)
    candidates = grid_boxes(
        dialog.rows[0]["coordinates"],
        8,
        16,
        column_end=[0.88, 0.04],
        row_end=[0.06, 0.94],
    )
    dialog._apply_grid(candidates)
    assert len(dialog.rows) == 129
    assert dialog.object_meta[dialog.ids[-1]] == {"row": 16, "column": 8}
    dialog._undo()
    assert dialog.rows == before
    dialog._select_box(0)
    grid = GridDialog(dialog, before[0]["coordinates"])
    grid.anchors[0].setValue(88)
    grid.anchors[3].setValue(94)
    grid.preview()
    grid.apply()
    assert len(dialog.rows) == 128
    dialog._undo()
    assert dialog.rows == before
    grid.deleteLater()


def test_pending_undo_vertex_insert_delete_and_self_intersection(workspace):
    _app, make = workspace
    dialog, _ = make()
    for point in [(100, 100), (250, 100), (250, 250)]:
        QTest.mouseClick(
            dialog.view.viewport(),
            Qt.MouseButton.LeftButton,
            pos=dialog.view.mapFromScene(QPointF(*point)),
        )
    QTest.keyClick(dialog.view, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert len(dialog.view.points) == 4 and dialog.pending
    dialog.view.cancel_gesture()
    dialog._create(dict(polygon(), points=[0.2, 0.2, 0.7, 0.2, 0.7, 0.7, 0.2, 0.7]))
    dialog._set_mode("select")
    dialog.view.insert_vertex(QPointF(0.45 * 599, 0.2 * 399))
    assert len(dialog.document.shapes[0]["points"]) == 10
    dialog.view.vertex = 1
    dialog.view.delete_vertex()
    assert len(dialog.document.shapes[0]["points"]) == 8
    with pytest.raises(ValueError):
        dialog.document.validate_shape(
            dict(polygon(), points=[0.1, 0.1, 0.8, 0.8, 0.1, 0.8, 0.8, 0.1])
        )


def test_pending_navigation_preserves_draft_without_negative_training_label(workspace):
    app, make = workspace
    dialog, saved = make()
    QTest.mouseClick(
        dialog.view.viewport(),
        Qt.MouseButton.LeftButton,
        pos=dialog.view.mapFromScene(QPointF(100, 100)),
    )
    dialog._navigate(1)
    wait(app, dialog, "draft_worker")
    assert dialog.navigation_delta == 1 and not saved
    reopened, _ = make()
    assert reopened._restorable
    reopened._restore_draft()
    assert reopened.pending and len(reopened.view.points) == 2


def test_failed_save_stays_open_and_edit_during_save_remains_dirty(workspace):
    app, make = workspace
    entered = threading.Event()
    release = threading.Event()
    saved = []

    def slow(payload):
        entered.set()
        release.wait(5)
        saved.append(payload)

    dialog, _ = make("detect", save=slow)
    dialog._create_box([0.2, 0.2, 0.1, 0.1])
    dialog._save()
    assert entered.wait(2)
    dialog._create_box([0.5, 0.5, 0.1, 0.1])
    release.set()
    wait(app, dialog)
    assert (
        dialog.dirty
        and len(saved[0]["items"] if isinstance(saved[0], dict) else saved[0]) == 1
    )
    dialog._save_callback = lambda _: (_ for _ in ()).throw(OSError("disk full"))
    dialog._save()
    wait(app, dialog)
    assert dialog.dirty and dialog.isVisible() and "disk full" in dialog.status.text()


def test_ai_stale_result_rejected_and_holes_preserved(workspace):
    app, make = workspace
    dialog, _ = make()
    entered = threading.Event()
    release = threading.Event()
    mask = np.zeros((400, 600), bool)
    mask[80:150, 100:200] = True
    mask[100:120, 130:150] = False

    class Engine:
        def predict(self, *args, **kwargs):
            entered.set()
            release.wait(5)
            return mask

    dialog._ai_engine = Engine()
    dialog._ai_engine_key = (dialog.ai_model.currentData(), 0)
    dialog._ai_prompt({"point": [110, 90], "label": 1})
    assert entered.wait(2)
    dialog._create(polygon())
    release.set()
    wait(app, dialog, "ai_worker")
    assert dialog._ai_proposal is None
    dialog._ai_prompt({"point": [110, 90], "label": 1})
    wait(app, dialog, "ai_worker")
    assert dialog.ai_apply.isEnabled()
    dialog._apply_proposal()
    assert dialog.document.render()[90, 110] == 1
    # This point is outside the earlier polygon, so the proposal's hole remains background.
    assert dialog.document.shapes[-1]["kind"] == "raster"
    with Image.open(io.BytesIO(dialog.document.encode())) as image:
        restored = MaskDocument.from_image(image, 600, 400, 3)
    np.testing.assert_array_equal(restored.render(), dialog.document.render())


def test_actual_zoom_and_small_window_controls(workspace):
    app, make = workspace
    dialog, _ = make("detect")
    dialog._actual_size()
    ratio = dialog.view.viewport().devicePixelRatioF()
    assert dialog.view.transform().m11() * ratio == pytest.approx(1)
    dialog._zoom(1)
    assert dialog.view.transform().m11() * ratio == pytest.approx(1.2)
    dialog.resize(900, 700)
    app.processEvents()
    assert not dialog.inspector.isVisible()
    assert dialog.save_button.isVisible() and dialog.review_button.isVisible()


def test_v1_draft_ids_stable_and_scoped_hole_survives_deleting_an_earlier_object(
    workspace,
):
    _app, make = workspace
    source = MaskDocument(
        600,
        400,
        3,
        shapes=[
            polygon(),
            dict(polygon(2), points=[0.65, 0.1, 0.9, 0.1, 0.9, 0.4, 0.65, 0.4]),
        ],
    )
    dialog, _ = make(document=source)
    keys = [s["object_id"] for s in dialog.document.shapes]
    doc2 = MaskDocument(
        600,
        400,
        3,
        shapes=[
            polygon(),
            dict(polygon(2), points=[0.65, 0.1, 0.9, 0.1, 0.9, 0.4, 0.65, 0.4]),
        ],
    )
    doc2.ensure_object_ids(dialog.image_path)
    assert [s["object_id"] for s in doc2.shapes] == keys
    dialog._select_box(1)
    dialog._set_mode("erase")
    dialog.class_combo.setCurrentIndex(2)
    drag(dialog, (460, 100), (460, 100))
    assert dialog.document.render()[100, 460] == 0
    dialog._select_box(0)
    dialog._remove()
    assert dialog.document.render()[100, 460] == 0


def test_ignore_is_explicit_and_fill_requires_confirmation(workspace):
    _app, make = workspace
    dialog, _ = make()
    dialog.class_combo.setCurrentIndex(dialog.class_combo.findData(255))
    dialog._set_mode("brush")
    drag(dialog, (80, 80), (120, 80))
    assert dialog.document.render()[80, 100] == 255
    dialog._set_mode("select")
    QTest.mouseClick(
        dialog.view.viewport(),
        Qt.MouseButton.LeftButton,
        pos=dialog.view.mapFromScene(QPointF(100, 80)),
    )
    assert dialog.selected != -1
    dialog._set_mode("erase")
    drag(dialog, (100, 80), (100, 80))
    assert dialog.document.render()[80, 100] == 0
    dialog._create(polygon())
    before = dialog.document.render()
    dialog.class_combo.setCurrentIndex(2)
    dialog._fill(160, 100)
    np.testing.assert_array_equal(dialog.document.render(), before)
    assert dialog.proposal_bar.isVisible()
    dialog._apply_proposal()
    assert dialog.document.render()[100, 160] == 2


def test_close_or_navigation_waits_for_ai_before_destroying_workspace(workspace):
    app, make = workspace
    dialog, _ = make("segment")
    entered = threading.Event()
    release = threading.Event()

    class Engine:
        def predict(self, *args, **kwargs):
            entered.set()
            release.wait(5)
            return np.zeros((400, 600), bool)

    dialog._ai_engine = Engine()
    dialog._ai_engine_key = (dialog.ai_model.currentData(), 0)
    dialog._ai_prompt({"point": [100, 100], "label": 1})
    assert entered.wait(2)
    dialog.navigation_delta = 1
    dialog.accept()
    assert dialog.isVisible() and dialog.ai_worker is not None
    release.set()
    wait(app, dialog, "ai_worker")
    assert (
        dialog.result() == QDialog.DialogCode.Accepted and dialog._ai_proposal is None
    )


def test_detection_pending_gesture_cannot_save_and_escape_keeps_draw_tool(workspace):
    _app, make = workspace
    dialog, saved = make("detect")
    view = dialog.view
    pos = view.mapFromScene(QPointF(100, 100))
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=pos)
    assert dialog.pending and not dialog.save_button.isEnabled()
    dialog._save()
    assert not saved and dialog.worker is None
    QTest.keyClick(view, Qt.Key.Key_Escape)
    assert not dialog.pending and view.mode == "draw"
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=pos)
    assert dialog.rows == []


def test_new_mask_payload_is_readable_by_actual_training_loader(tmp_path):
    from core.dataset_editor import edit_dataset
    from core.project import ProjectManager
    from dataset import SegmentationDataset

    project = ProjectManager.create_new(
        "newmask", "segment", str(tmp_path / "project"), ["bg", "part", "defect"]
    )
    path = Path(project.data.train_dir) / "part.png"
    Image.new("RGB", (100, 80), "gray").save(path)
    doc = MaskDocument(100, 80, 3, shapes=[polygon(object_id="part")])
    doc.objects["part"] = {"hidden": True, "locked": True}
    doc.shapes.append(
        doc.validate_shape(
            {
                "kind": "stroke",
                "class_id": 0,
                "operation": "erase",
                "scope_object": "part",
                "points": [0.3, 0.3],
                "width": 8,
            }
        )
    )
    edit_dataset(project, "mask_annotations", [str(path)], annotations=doc.payload())
    mask_root = Path(project.data.root) / "masks/train"
    dataset = SegmentationDataset(
        project.data.train_dir,
        str(mask_root),
        input_size=(80, 100),
        is_train=False,
        num_classes=3,
    )
    _, target = dataset[0]
    np.testing.assert_array_equal(target.numpy(), doc.render())


def test_process_crash_journal_recovers_before_dataset_scan(tmp_path):
    import os
    import subprocess
    import sys

    from core.annotation_journal import journal_paths
    from core.dataset_editor import scan_dataset
    from core.project import ProjectManager

    project = ProjectManager.create_new(
        "crash", "detect", str(tmp_path / "project"), ["part", "defect"]
    )
    ProjectManager.save(project)
    image = Path(project.data.train_dir) / "sample.png"
    Image.new("RGB", (20, 20)).save(image)
    label = Path(project.data.root) / "labels/train/sample.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    before = b"0 .5 .5 .4 .4\n"
    label.write_bytes(before)
    script = """import os,sys
from core.project import ProjectManager
from core.dataset_editor import EditTransaction
from pathlib import Path
p=ProjectManager.load(sys.argv[1])
t=EditTransaction(p,'annotations')
t.write(Path(sys.argv[2]),b'1 .5 .5 .2 .2\\n')
os._exit(23)
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "gui"))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            ProjectManager.get_active_filepath(project),
            str(label),
        ],
        env=env,
        check=False,
    )
    assert result.returncode == 23 and label.read_bytes() != before
    _, pointer, _ = journal_paths(project)
    assert pointer.is_file()
    rows = scan_dataset(project)["images"]
    assert label.read_bytes() == before and not pointer.exists()
    assert rows[0]["classes"] == ["part"]


def test_square_brush_and_vertex_snap(workspace):
    _app, make = workspace
    dialog, _ = make()
    dialog.brush_tip.setCurrentIndex(dialog.brush_tip.findData("square"))
    dialog.brush_size.setValue(9)
    dialog._set_mode("brush")
    drag(dialog, (100, 100), (100, 100))
    assert dialog.document.render()[96, 96] == 1
    circle = MaskDocument(
        600,
        400,
        3,
        shapes=[
            {
                "kind": "stroke",
                "class_id": 1,
                "width": 9,
                "points": [100 / 599, 100 / 399],
            }
        ],
    )
    assert circle.render()[96, 96] == 0
    dialog._create(
        {
            "kind": "polygon",
            "class_id": 1,
            "points": [
                200 / 599,
                200 / 399,
                400 / 599,
                200 / 399,
                400 / 599,
                300 / 399,
            ],
        }
    )
    dialog.snap.setChecked(True)
    dialog._set_mode("polygon")
    QTest.mouseClick(
        dialog.view.viewport(),
        Qt.MouseButton.LeftButton,
        pos=dialog.view.mapFromScene(QPointF(202, 202)),
    )
    np.testing.assert_allclose(
        dialog.view.points[:2], [200 / 599, 200 / 399], atol=0.003
    )


def test_grid_anchor_can_be_picked_directly_on_image(workspace):
    app, make = workspace
    dialog, _ = make("detect")
    dialog._create_box([0.1, 0.1, 0.04, 0.03])
    dialog._grid()
    grid = dialog._grid_dialog
    grid.pick_anchor(0)
    QTest.mouseClick(
        dialog.view.viewport(),
        Qt.MouseButton.LeftButton,
        pos=dialog.view.mapFromScene(QPointF(500, 50)),
    )
    assert grid.anchors[0].value() == pytest.approx(100 * 500 / 600, abs=0.3)
    assert grid.anchors[1].value() == pytest.approx(100 * 50 / 400, abs=0.3)
    assert dialog.view.mode != "grid_anchor"
    grid.reject()
    app.processEvents()


def test_shortcuts_can_be_rebound_without_intercepting_class_search(workspace):
    _app, make = workspace
    dialog, _ = make("detect")
    dialog._create_box([0.2, 0.2, 0.1, 0.1])
    values = dict(dialog.shortcuts.bindings, delete="X")
    dialog.shortcuts.update(values)
    QTest.keyClick(dialog.view, Qt.Key.Key_Delete)
    assert len(dialog.rows) == 1
    QTest.keyClick(dialog.view, Qt.Key.Key_X)
    assert dialog.rows == []
    dialog.class_search.setFocus()
    QTest.keyClicks(dialog.class_search, "vx")
    assert dialog.class_search.text() == "vx"
    with pytest.raises(ValueError):
        dialog.shortcuts.update(dict(dialog.shortcuts.bindings, delete="V"))
    with pytest.raises(ValueError):
        dialog.shortcuts.update(dict(dialog.shortcuts.bindings, delete="Space"))


def test_display_adjustment_does_not_modify_image_or_training_pixels(workspace):
    _app, make = workspace
    dialog, _ = make()
    dialog._create(polygon())
    before = dialog.document.render()
    source = Path(dialog.image_path).read_bytes()
    color = dialog.view.source_item.pixmap().toImage().pixelColor(10, 10).red()
    dialog.brightness.setValue(50)
    dialog._levels_timer.stop()
    dialog._apply_levels()
    assert dialog.view.source_item.pixmap().toImage().pixelColor(10, 10).red() > color
    assert Path(dialog.image_path).read_bytes() == source
    np.testing.assert_array_equal(dialog.document.render(), before)
    dialog._reset_levels()
    assert dialog.view.source_item.pixmap().toImage().pixelColor(10, 10).red() == color


def test_class_addition_does_not_make_unsaved_draft_baseline_invalid(workspace):
    app, make = workspace
    dialog, _ = make()
    dialog._create(polygon())
    dialog._add_class_callback = lambda name: dialog.names + [name]
    with patch(
        "widgets.annotation_workspace.QInputDialog.getText", return_value=("new", True)
    ):
        dialog._add_class()
    dialog._draft_timer.stop()
    dialog._write_draft()
    wait(app, dialog, "draft_worker")
    loaded = dialog.drafts.load(
        dialog.image_path, "segment", dialog._baseline, dialog.names
    )
    assert len(loaded["payload"]["shapes"]) == 1
    baseline = MaskDocument(600, 400, 4)
    baseline.ensure_object_ids(dialog.image_path)
    assert dialog._label_fingerprint({"document": baseline}) == dialog._baseline


def test_external_annotation_change_is_not_overwritten(tmp_path):
    from core.dataset_editor import annotation_revision, edit_dataset
    from core.project import ProjectManager

    p = ProjectManager.create_new(
        "revision", "detect", str(tmp_path / "project"), ["part"]
    )
    image = Path(p.data.train_dir) / "part.png"
    Image.new("RGB", (30, 20)).save(image)
    old = annotation_revision(p, image, "train")
    other = [{"class_id": 0, "coordinates": [0.5, 0.5, 0.2, 0.2]}]
    result = edit_dataset(
        p, "annotations", [str(image)], annotations=other, expected_revision=old
    )
    assert result["annotation_revision"] != old
    label = Path(p.data.root) / "labels/train/part.txt"
    before = label.read_bytes()
    with pytest.raises(ValueError, match="다른 곳에서 변경"):
        edit_dataset(
            p, "annotations", [str(image)], annotations=[], expected_revision=old
        )
    assert label.read_bytes() == before


def test_grid_rejects_stale_source_and_preserves_other_selected_objects(workspace):
    _app, make = workspace
    dialog, _ = make("detect")
    dialog._create_box([0.06, 0.04, 0.04, 0.025])
    dialog._create_box([0.5, 0.5, 0.03, 0.03])
    dialog._select_many({0, 1})
    dialog.selected = 0
    grid = GridDialog(dialog, dialog.rows[0]["coordinates"])
    grid.anchors[0].setValue(88)
    grid.anchors[3].setValue(94)
    grid.preview()
    grid.apply()
    assert len(dialog.rows) == 129  # 128 candidates plus the other selected object
    assert any(row["coordinates"] == [0.5, 0.5, 0.03, 0.03] for row in dialog.rows)
    dialog._select_box(0)
    grid = GridDialog(dialog, dialog.rows[0]["coordinates"])
    dialog._create_box([0.7, 0.7, 0.02, 0.02])
    before = len(dialog.rows)
    grid.apply()
    assert len(dialog.rows) == before and "수정됐습니다" in grid.info.text()
    grid.reject()


def test_ai_new_prompt_cannot_apply_previous_proposal_and_escape_cancels(workspace):
    app, make = workspace
    dialog, _ = make()
    entered = threading.Event()
    release = threading.Event()
    mask = np.zeros((400, 600), bool)
    mask[40:80, 40:80] = True

    class Engine:
        calls = 0

        def predict(self, *args, **kwargs):
            self.calls += 1
            if self.calls > 1:
                entered.set()
                release.wait(5)
            return mask

    dialog._ai_engine = Engine()
    dialog._ai_engine_key = (dialog.ai_model.currentData(), 0)
    dialog._set_mode("ai_point")
    dialog._ai_prompt({"point": [50, 50], "label": 1})
    wait(app, dialog, "ai_worker")
    assert dialog._ai_proposal is not None
    dialog._ai_prompt({"point": [55, 55], "label": 1})
    assert entered.wait(2)
    assert dialog._ai_proposal is None and not dialog.proposal_bar.isVisible()
    QTest.keyClick(dialog.view, Qt.Key.Key_Escape)
    release.set()
    wait(app, dialog, "ai_worker")
    assert dialog._ai_proposal is None and not dialog.document.shapes


def test_grid_can_copy_a_locked_seed_without_overwriting_or_duplicating_it(workspace):
    _app, make = workspace
    dialog, _ = make("detect")
    dialog._create_box([0.06, 0.04, 0.04, 0.025])
    seed_id = dialog.ids[0]
    dialog._set_flag(0, "locked", True)
    dialog._select_box(0)
    grid = GridDialog(dialog, dialog.rows[0]["coordinates"])
    grid.anchors[0].setValue(88)
    grid.anchors[3].setValue(94)
    grid.preview()
    assert not grid.replace_seed.isChecked() and len(grid.selected()) == 127
    grid.apply()
    assert len(dialog.rows) == 128 and dialog.ids[0] == seed_id
    assert dialog.object_meta[seed_id]["locked"]


def test_original_image_change_is_included_in_save_revision(tmp_path):
    from core.dataset_editor import annotation_revision, edit_dataset
    from core.project import ProjectManager

    p = ProjectManager.create_new(
        "image-revision", "detect", str(tmp_path / "project"), ["part"]
    )
    path = Path(p.data.train_dir) / "part.png"
    Image.new("RGB", (30, 20), "gray").save(path)
    old = annotation_revision(p, path, "train")
    Image.new("RGB", (30, 20), "white").save(path)
    with pytest.raises(ValueError, match="다른 곳에서 변경"):
        edit_dataset(
            p, "annotations", [str(path)], annotations=[], expected_revision=old
        )


def test_connected_components_preview_cancel_apply_and_undo_preserve_pixels(
    workspace, monkeypatch
):
    app, make = workspace
    pixels = np.zeros((400, 600), np.uint8)
    pixels[40:120, 40:120] = 1
    pixels[60:80, 60:80] = 0  # A hole must stay empty after splitting.
    pixels[180:220, 240:300] = 1
    pixels[300:360, 450:510] = 2
    pixels[10:30, 450:510] = 255
    doc = MaskDocument(600, 400, 3, base=pixels)
    dialog, _saved = make(document=doc)
    dialog._set_flag(-4, "locked", True)
    dialog._select_many({-3})
    baseline = dialog.document.render()
    original_history = len(dialog.undo_stack)
    monkeypatch.setattr(QDialog, "exec", lambda _: QDialog.DialogCode.Rejected)
    dialog._split_components()
    np.testing.assert_array_equal(dialog.document.render(), baseline)
    assert len(dialog.undo_stack) == original_history

    def confirm(preview):
        assert len(dialog.view.grid_preview) == 2
        np.testing.assert_array_equal(dialog.document.render(), baseline)
        preview.show()
        app.processEvents()  # Exercise the real preview painter as well.
        preview.hide()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", confirm)
    dialog._split_components()
    np.testing.assert_array_equal(dialog.document.render(), baseline)
    assert len(dialog.selected_indices) == 2
    assert len(dialog.undo_stack) == original_history + 1
    assert dialog.object_meta["base:2"]["locked"]
    with Image.open(io.BytesIO(dialog.document.encode())) as encoded:
        restored = MaskDocument.from_image(encoded, 600, 400, 3)
    np.testing.assert_array_equal(restored.render(), baseline)
    dialog._undo()
    np.testing.assert_array_equal(dialog.document.render(), baseline)
    assert not dialog.document.shapes


def test_object_table_stays_readable_with_application_style(workspace):
    app, make = workspace
    dialog, _ = make("detect")
    root = Path(__file__).resolve().parents[1]
    style = (
        (root / "gui/resources/styles/dark_theme.qss")
        .read_text()
        .replace("{ICON_DIR}", str(root / "gui/resources/icons"))
    )
    dialog.setStyleSheet(style)
    dialog._create_box([0.5, 0.5, 0.4, 0.4])
    app.processEvents()
    row_label = dialog.table.item(0, 0)
    assert (
        dialog.table.columnWidth(0)
        >= dialog.table.fontMetrics().horizontalAdvance("R16 C08") + 8
    )
    combo = dialog.table.cellWidget(0, 1)
    assert combo.isVisible()
    assert (
        combo.width() >= combo.fontMetrics().horizontalAdvance(combo.currentText()) + 30
    )
    assert dialog.table.visualItemRect(row_label).height() >= 24


def test_thumbnail_navigation_reuses_threads_and_releases_image_state(
    workspace, tmp_path
):
    import weakref

    from widgets.annotation_io import ThumbnailLoader

    app, _make = workspace
    image = tmp_path / "thumbnail.png"
    Image.new("RGB", (128, 128), "gray").save(image)
    thread_ids = set()
    jobs, providers = [], []

    class State:
        def __call__(self, _path):
            thread_ids.add(threading.get_ident())
            return "미작업"

    for _ in range(40):
        provider = State()
        providers.append(weakref.ref(provider))
        loader = ThumbnailLoader([(0, str(image))], provider)
        jobs.append(loader)
        loader.start()
    del provider
    for job in jobs:
        assert job.wait(10000)
        assert job.state_provider is None and job.paths == []
    app.processEvents()
    assert 1 <= len(thread_ids) <= 2
    assert all(reference() is None for reference in providers)
