"""Drive the real Qt teaching tools, including persistence and failed navigation."""
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog
from core.mask_annotations import MaskDocument
from widgets.segmentation_annotation import SegmentationAnnotationDialog


@pytest.fixture
def editor(tmp_path):
    app = QApplication.instance() or QApplication([])
    image = tmp_path / "sample.png"
    Image.new("RGB", (600, 400), "gray").save(image)
    saved = []
    dialog = SegmentationAnnotationDialog(image, ["background", "part", "defect"],
        MaskDocument(600, 400, 3), saved.append, navigation=(0, 2))
    dialog.show()
    app.processEvents()
    yield app, dialog, saved
    dialog.view.cancel_gesture()
    dialog.dirty = False
    dialog.close()
    dialog.deleteLater()
    app.processEvents()


def click(dialog, x, y):
    position = dialog.view.mapFromScene(QPointF(x, y))
    QTest.mouseClick(dialog.view.viewport(), Qt.MouseButton.LeftButton, pos=position)


def drag(dialog, start, end):
    view = dialog.view
    QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=view.mapFromScene(QPointF(*start)))
    QApplication.processEvents()  # A selected vertex must also paint before it moves.
    QTest.mouseMove(view.viewport(), view.mapFromScene(QPointF(*end)))
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=view.mapFromScene(QPointF(*end)))
    QApplication.processEvents()


def test_polygon_vertex_brush_eraser_undo_save_and_reopen(editor):
    app, dialog, saved = editor
    for point in [(100, 100), (400, 100), (400, 300), (100, 300)]:
        click(dialog, *point)
    assert dialog.pending and not dialog.save_button.isEnabled()
    # Clicking a toolbar action does not discard the pending polygon.
    dialog.finish_button.click()
    assert len(dialog.document.shapes) == 1 and not dialog.pending
    assert dialog.document.render()[200, 250] == 1
    dialog.mode_buttons["select"].click()
    drag(dialog, (400, 300), (450, 330))
    assert dialog.document.shapes[0]["points"][4] > .7
    dialog._zoom(1)
    app.processEvents()
    dialog.class_combo.setCurrentIndex(2)
    dialog.mode_buttons["brush"].click()
    dialog.brush_size.setValue(30)
    drag(dialog, (220, 200), (300, 200))
    assert dialog.document.render()[200, 250] == 2
    dialog.mode_buttons["erase"].click()
    drag(dialog, (250, 200), (250, 200))
    assert dialog.document.render()[200, 250] == 0
    assert dialog.table.rowCount() == 2  # polygon + painted brush; eraser is not an object row
    assert dialog.document.shapes[-1]["operation"] == "erase"
    QTest.keyClick(dialog.view, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert dialog.document.render()[200, 250] == 2
    dialog.redo_button.click()
    assert dialog.document.render()[200, 250] == 0
    dialog._change_class(0, 2)
    assert dialog.document.render()[200, 250] == 0
    expected = dialog.document.render()
    QTest.keyClick(dialog.view, Qt.Key.Key_K)
    dialog.worker.wait(10000)
    app.processEvents()
    assert dialog.navigation_delta == 1 and dialog.result() == QDialog.DialogCode.Accepted
    restored = MaskDocument.from_payload(saved[0], 600, 400, 3)
    np.testing.assert_array_equal(restored.render(), expected)


def test_failed_save_and_incomplete_polygon_cannot_navigate(editor):
    app, dialog, saved = editor
    click(dialog, 200, 100)
    dialog.next_button.click()
    assert dialog.navigation_delta == 0 and dialog.worker is None
    QTest.keyClick(dialog.view, Qt.Key.Key_Escape)
    dialog._save_callback = lambda _: (_ for _ in ()).throw(OSError("disk full"))
    dialog.next_button.click()
    dialog.worker.wait(10000)
    app.processEvents()
    assert dialog.navigation_delta == 0 and dialog.isVisible()
    assert "disk full" in dialog.status.text() and dialog.dirty
    dialog._save_callback = saved.append
    dialog.next_button.click()
    dialog.worker.wait(10000)
    app.processEvents()
    assert len(saved) == 1 and saved[0]["shapes"] == []


def test_dataset_teaching_button_opens_segment_editor(tmp_path):
    from core.project import ProjectManager
    from widgets.dataset_widget import DatasetWidget
    from pathlib import Path
    app = QApplication.instance() or QApplication([])
    project = ProjectManager.create_new("seg", "segment", str(tmp_path), ["background", "part"])
    image = Path(project.data.train_dir) / "sample.png"
    Image.new("RGB", (80, 60)).save(image)
    with patch.object(DatasetWidget, "_scan_all"):
        page = DatasetWidget()
        page.set_project(project)
        assert not page.annotation_row.isHidden()
        page._selected_paths = {str(image)}
        with patch("widgets.segmentation_annotation.SegmentationAnnotationDialog") as dialog:
            dialog.return_value.navigation_delta = 0
            page.annotate_btn.click()
            assert dialog.call_args.args[0] == str(image)
        page.close()
        page.deleteLater()
    app.processEvents()
