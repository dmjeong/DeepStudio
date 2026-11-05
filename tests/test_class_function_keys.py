"""라벨링 작업실: F1~F12로 클래스를 고른다. 선택한 객체가 있으면 그 객체의 클래스가 바뀐다.

    분할   F1 = 1번 클래스 (0번 배경은 건너뜀)      탐지   F1 = 0번 클래스
    포커스가 클래스 콤보·표에 있어도 동작한다 (창 단위 단축키).
"""

import pytest
from PIL import Image

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from core.annotation_tools import AnnotationDraftStore
from core.mask_annotations import MaskDocument

NAMES = ["background", "shift", "crack", "fm"]


def _rect(class_id, x0, y0, x1, y1, name):
    return {"kind": "polygon", "class_id": class_id, "object_id": name,
            "points": [x0 / 60, y0 / 40, x1 / 60, y0 / 40, x1 / 60, y1 / 40, x0 / 60, y1 / 40]}


@pytest.fixture
def workspace(tmp_path):
    from widgets.segmentation_annotation import SegmentationAnnotationDialog
    app = QApplication.instance() or QApplication([])
    Image.new("RGB", (60, 40), "gray").save(tmp_path / "a.png")
    document = MaskDocument(60, 40, 4, shapes=[_rect(1, 2, 2, 20, 18, "a"), _rect(1, 30, 2, 50, 18, "b")])
    dialog = SegmentationAnnotationDialog(tmp_path / "a.png", NAMES, document, lambda *_: None,
                                          draft_store=AnnotationDraftStore(tmp_path / "drafts"))
    dialog.show()
    dialog.activateWindow()
    app.processEvents()
    yield dialog, app
    dialog.close()
    dialog.deleteLater()


def _press(app, widget, key):
    QTest.keyClick(widget, key)
    app.processEvents()


def test_function_key_changes_the_selected_object(workspace):
    dialog, app = workspace
    dialog.view.choose({0}, 0)
    app.processEvents()
    dialog.view.setFocus()
    _press(app, dialog.view, Qt.Key.Key_F3)                  # F3 = 3번 클래스 fm
    assert dialog._class_id(0) == 3 and dialog._class_id(1) == 1   # 선택한 객체만
    assert "fm" in dialog.status.text()
    dialog._undo()
    assert dialog._class_id(0) == 1


def test_function_key_works_with_focus_on_the_class_combo(workspace):
    dialog, app = workspace
    dialog.view.choose({1}, 1)
    app.processEvents()
    dialog.class_combo.setFocus()
    _press(app, dialog.class_combo, Qt.Key.Key_F2)           # F2 = 2번 클래스 crack
    assert dialog._class_id(1) == 2


def test_function_key_without_selection_picks_the_next_drawing_class(workspace):
    dialog, app = workspace
    dialog.view.choose(set())
    app.processEvents()
    _press(app, dialog.view, Qt.Key.Key_F2)
    assert dialog._active_class_id() == 2 and dialog._class_id(0) == 1   # 기존 객체는 그대로
    _press(app, dialog.view, Qt.Key.Key_F9)                  # 없는 클래스: 아무 일도 없음
    assert dialog._active_class_id() == 2


def test_class_list_shows_the_function_keys(workspace):
    dialog, _ = workspace
    texts = [dialog.class_list.item(i).text() for i in range(dialog.class_list.count())]
    assert "[F" not in texts[0] and texts[1].endswith("[F1]") and texts[3].endswith("[F3]")


def test_detection_function_key_starts_at_the_first_class(tmp_path):
    from widgets.annotation_workspace import DetectionAnnotationDialog
    app = QApplication.instance() or QApplication([])
    Image.new("RGB", (60, 40), "gray").save(tmp_path / "a.png")
    rows = [{"class_id": 0, "coordinates": [0.3, 0.3, 0.2, 0.2]}]
    dialog = DetectionAnnotationDialog(tmp_path / "a.png", ["chip", "crack"], rows, lambda *_: None,
                                       draft_store=AnnotationDraftStore(tmp_path / "drafts"))
    dialog.show()
    app.processEvents()
    dialog.view.choose({0}, 0)
    app.processEvents()
    _press(app, dialog.view, Qt.Key.Key_F2)                  # 탐지: F2 = 1번 클래스 crack
    assert dialog.rows[0]["class_id"] == 1
    dialog.close()
    dialog.deleteLater()
