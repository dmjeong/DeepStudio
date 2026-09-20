"""모델 관리 Settings가 실제 실행 가능 여부를 혼동 없이 표시하는지 검사한다."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox

from core.project import ProjectManager

from app.main_window import MainWindow
from widgets.model_manager_widget import ModelManagerWidget


@pytest.fixture
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def manager(qt_app):
    page = ModelManagerWidget()
    yield page
    page.close()
    page.deleteLater()
    qt_app.processEvents()


def _row_for(page, model_name):
    for index in range(page.catalog.topLevelItemCount()):
        row = page.catalog.topLevelItem(index)
        if row.text(2) == model_name:
            return row
    raise AssertionError(f"model row not found: {model_name}")


def test_model_manager_uses_plain_rows_and_english_task_names(manager):
    assert manager.catalog.columnCount() == 4
    assert not manager.catalog.alternatingRowColors()
    native = _row_for(manager, "DeepLab V3+ ResNet34")
    assert native.text(0) == "기본 제공"
    assert native.text(1) == "Segmentation"
    assert native.text(3) == "내장"

    pending = _row_for(manager, "SAM2 Hiera Tiny")
    assert pending.text(0) == "기본 제공"
    assert pending.text(1) == "Segmentation"
    assert pending.text(3) == "내장"
    assert manager.sam2_variant.count() == 4
    assert manager.sam2_download_button.text() == "사전학습 가중치 다운로드"
    assert manager.sam2_apply_button.text() == "현재 프로젝트에 적용"


def test_downloaded_sam2_weight_can_be_applied_to_the_current_segment_project(manager, tmp_path, monkeypatch):
    from core import model_registry
    from sam2_assets import get_sam2_asset

    cache = tmp_path / "models"
    monkeypatch.setattr(model_registry, "default_installed_model_root", lambda: cache)
    checkpoint = cache / "_builtin_assets" / "sam2" / get_sam2_asset("sam2_hiera_tiny").filename
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"official-checkpoint")
    project = ProjectManager.create_new("Segment", "segment", str(tmp_path / "project"), ["part"])
    manager.set_project(project)
    manager._update_sam2_apply_state()
    assert manager.sam2_apply_button.isEnabled()
    manager._apply_sam2_to_project()
    assert project.model.model_id == "sam2_hiera_tiny"
    assert project.model.pretrained_weights == str(checkpoint.resolve())


def test_sam2_variant_change_refreshes_without_treating_index_as_path(manager, qt_app):
    errors = []
    import sys
    original = sys.excepthook
    sys.excepthook = lambda kind, value, traceback: errors.append(str(value))
    try:
        manager.sam2_variant.setCurrentIndex(1)
        qt_app.processEvents()
    finally:
        sys.excepthook = original
    assert errors == []


def test_settings_can_open_before_a_project_exists(qt_app, monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)
    window = MainWindow()
    try:
        window._navigate_to(6)
        assert window.project is None
        assert window.content_stack.currentWidget() is window.model_manager_page
        assert window.nav_buttons[6].isChecked()
    finally:
        window.close()
        window.deleteLater()
        qt_app.processEvents()
