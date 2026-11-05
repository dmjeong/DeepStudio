"""배경 없이 만든 분할 프로젝트(0번 = 결함 이름)를 데스크톱 앱이 드러내고 막는지.

    분할 마스크: 칠하지 않은 픽셀 = 0번. 0번이 '시프트 더블'이면
      · 배경 전체가 '시프트 더블'로 학습되고 (추론에서 배경이 그 이름으로 보인다)
      · '시프트 더블'로 칠해도 0이라 저장·학습되지 않는다.
    ① 데이터셋 화면에 번호와 경고  ② 작업실에서 0번 선택 불가  ③ 학습 시작 때 확인
"""

from unittest.mock import patch

import pytest
from PIL import Image

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QMessageBox

from core.annotation_tools import AnnotationDraftStore
from core.mask_annotations import MaskDocument
from core.project import ProjectManager

NO_BACKGROUND = ["시프트 더블", "crack"]


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _project(tmp_path, names):
    project = ProjectManager.create_new("bg-slot", "segment", str(tmp_path / "project"), ["crack"])
    project.data.class_names, project.data.num_classes = list(names), len(names)  # 예전에 배경 없이 만든 프로젝트
    return project


def test_dataset_page_shows_class_numbers_and_warns_when_zero_is_not_background(app, tmp_path):
    from widgets.dataset_widget import DatasetWidget
    with patch.object(DatasetWidget, "_scan_all"):
        widget = DatasetWidget()
        widget.set_project(_project(tmp_path, NO_BACKGROUND))
        text = widget.class_index_label.text()
        assert "0번 '시프트 더블'" in text and "번호: 0 시프트 더블 · 1 crack" in text
        assert not widget.class_index_label.isHidden() and not widget.fix_background_btn.isHidden()
        widget.set_project(_project(tmp_path / "ok", ["background", "시프트 더블", "crack"]))
        assert widget.class_index_label.text() == "번호: 0 background · 1 시프트 더블 · 2 crack"
        assert widget.fix_background_btn.isHidden()
        widget.deleteLater()


def _workspace(tmp_path, names):
    from widgets.segmentation_annotation import SegmentationAnnotationDialog
    image = tmp_path / "chip.png"
    Image.new("RGB", (60, 40), "gray").save(image)
    return SegmentationAnnotationDialog(image, names, MaskDocument(60, 40, len(names)), lambda *_: None,
                                        draft_store=AnnotationDraftStore(tmp_path / "drafts"))


def test_workspace_cannot_draw_with_a_defect_name_in_the_background_slot(app, tmp_path):
    dialog = _workspace(tmp_path, NO_BACKGROUND)
    dialog.show()
    app.processEvents()
    item = dialog.class_combo.model().item(0)
    assert not item.isEnabled() and "배경 자리" in item.text()
    assert not dialog.background_warning.isHidden()
    assert dialog._active_class_id() == 1
    dialog.class_combo.setCurrentIndex(0)          # 코드로 골라도 0번으로는 그리지 않는다
    app.processEvents()
    assert dialog._active_class_id() == 1 and dialog.view.class_id == 1
    dialog._class_list_selected(0)                 # 클래스 목록에서 0번을 눌러도
    assert dialog._active_class_id() == 1 and "배경 자리" in dialog.status.text()
    assert "배경 자리" in dialog.class_list.item(0).text()
    dialog.close()
    dialog.deleteLater()


def test_workspace_with_background_first_is_unchanged(app, tmp_path):
    dialog = _workspace(tmp_path, ["background", "시프트 더블"])
    dialog.show()
    app.processEvents()
    assert dialog.class_combo.model().item(0).isEnabled() and dialog.class_combo.itemText(0) == "background"
    assert dialog.background_warning.isHidden()
    dialog.close()
    dialog.deleteLater()


@pytest.mark.parametrize("answer,expected", [("취소", False), ("그대로 학습", True)])
def test_training_start_asks_when_zero_is_not_background(app, tmp_path, answer, expected):
    from widgets.training_widget import TrainingWidget
    widget = TrainingWidget()
    widget.project = _project(tmp_path, NO_BACKGROUND)
    shown = []

    def pick(box):
        shown.append(box.text())
        return next(b for b in box.buttons() if b.text() == answer)

    with patch.object(QMessageBox, "exec", lambda box: None), patch.object(QMessageBox, "clickedButton", pick):
        assert widget._background_slot_ok() is expected
    assert shown and "배경 전체가 '시프트 더블'" in shown[0]
    widget.project = _project(tmp_path / "ok", ["background", "crack"])
    shown.clear()
    with patch.object(QMessageBox, "exec", lambda box: None), patch.object(QMessageBox, "clickedButton", pick):
        assert widget._background_slot_ok() is True
    assert not shown                                 # 배경이 0번이면 묻지 않는다
    widget.deleteLater()


def test_training_start_fix_button_runs_the_dataset_page_background_insert(app, tmp_path):
    from widgets.training_widget import TrainingWidget
    widget = TrainingWidget()
    widget.project = _project(tmp_path, NO_BACKGROUND)
    calls = []

    class Window:
        class dataset_page:
            @staticmethod
            def _insert_background():
                calls.append(True)

    def pick(box):
        return next(b for b in box.buttons() if b.text() == "배경 추가")

    with patch.object(widget, "window", lambda: Window), patch.object(QMessageBox, "exec", lambda box: None), \
            patch.object(QMessageBox, "clickedButton", pick):
        assert widget._background_slot_ok() is False  # 고친 뒤 사용자가 학습 시작을 다시 누른다
    assert calls == [True]
    widget.deleteLater()


def test_segmentation_augmentation_can_be_edited_on_the_desktop_training_page(app, tmp_path):
    """분할 학습은 좌우반전·회전·색상을 적용하는데 칸이 잠겨 기본값(0.5 · 15°)을 바꿀 수 없었다."""
    from widgets.training_widget import TrainingWidget
    project = ProjectManager.create_new("aug", "segment", str(tmp_path / "p"), ["background", "shift"])
    widget = TrainingWidget()
    widget.set_project(project)
    widget.model_id_combo.setCurrentIndex(widget.model_id_combo.findData("unet_lite"))
    for control in (widget.hflip_spin, widget.rotation_spin, widget.color_jitter_spin):
        assert control.isEnabled()
    widget.hflip_spin.setValue(0)
    widget.rotation_spin.setValue(0)
    widget._sync_config()
    assert (project.training.augmentation.horizontal_flip, project.training.augmentation.rotation) == (0, 0)
    widget.deleteLater()
