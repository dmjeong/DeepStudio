"""Drive the recovery UI after an external rename, including the actual save worker."""

import time
from pathlib import Path
from unittest.mock import patch

import pytest
from core.dataset_editor import edit_dataset, scan_dataset
from core.mask_annotations import MaskDocument
from core.project import ProjectManager
from PIL import Image

pytest.importorskip("PySide6")
from PySide6.QtCore import QCoreApplication, QEvent, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox
from widgets.dataset_link_dialog import DatasetLinkDialog
from widgets.dataset_widget import DatasetWidget


def _dataset(tmp_path):
    project = ProjectManager.create_new("Rename UI", "segment", str(tmp_path / "project"), ["crack"])
    image = Path(project.data.train_dir) / "before.png"
    Image.new("RGB", (64, 48), (120, 120, 130)).save(image)
    document = MaskDocument(64, 48, 2, shapes=[{
        "kind": "polygon", "class_id": 1, "points": [.2, .2, .5, .2, .5, .6],
    }])
    edit_dataset(project, "mask_annotations", [image], annotations=document.payload())
    mask = Path(project.data.root) / "masks/train/before.png"
    renamed = image.with_stem("after")
    image.rename(renamed)
    ProjectManager.save(project)
    return project, renamed, mask


def test_recovery_button_preview_and_real_worker_restore_mask(tmp_path):
    app = QApplication.instance() or QApplication([])
    project, image, mask = _dataset(tmp_path)
    original = mask.read_bytes()
    with patch.object(DatasetWidget, "_scan_all"):
        page = DatasetWidget()
        page.set_project(project)
        page.resize(1200, 800)
        page.show()
        page._scan_worker = None
        page._scan_ready(scan_dataset(project), {})
        app.processEvents()
        assert page.link_issue_row.isVisible()
        assert "1개" in page.link_issue_label.text()
        observed = []

        def choose_link():
            dialog = None
            try:
                dialog = next(w for w in app.topLevelWidgets() if isinstance(w, DatasetLinkDialog) and w.isVisible())
                assert not dialog.link_button.isEnabled()
                dialog.mask_combo.setCurrentIndex(1)
                assert not dialog.link_button.isEnabled()  # no automatic target guess
                dialog.image_combo.setCurrentIndex(1)
                assert dialog.selection == (str(image), str(mask), "train")
                assert dialog.link_button.isEnabled()
                assert not dialog.preview.pixmap().isNull()
                observed.append(True)
                QTest.mouseClick(dialog.link_button, Qt.MouseButton.LeftButton)
            except Exception as exc:  # noqa: BLE001 - propagate Qt callback failures below, never hang exec()
                observed.append(exc)
                if dialog is not None:
                    dialog.reject()

        try:
            QTimer.singleShot(30, choose_link)
            QTest.mouseClick(page.repair_links_btn, Qt.MouseButton.LeftButton)
            deadline = time.monotonic() + 20
            while page._class_worker is not None and time.monotonic() < deadline:
                app.processEvents()
                QTest.qWait(5)
            assert observed == [True]
            assert page._class_worker is None
            assert not mask.exists()
            assert mask.with_stem("after").read_bytes() == original
            page._scan_ready(scan_dataset(project), {})
            assert not page.link_issue_row.isVisible()
            assert page._dataset_index["images"][0]["training_target"] == "mask"
        finally:
            if page._class_worker is not None:
                assert page._class_worker.wait(20000)
                app.processEvents()
            page.close()
            page.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture(autouse=True)
def _surface_worker_errors(monkeypatch):
    # Never hide the real failure in a modal dialog or a Qt teardown abort.
    def report(parent, title, message, *args, **kwargs):
        print(f"DATASET_UI_ERROR: {title}: {message}", flush=True)
        return QMessageBox.StandardButton.Ok
    monkeypatch.setattr(QMessageBox, "warning", report)


def test_preview_rejects_wrong_size_and_cancel_changes_nothing(tmp_path):
    app = QApplication.instance() or QApplication([])
    project, image, mask = _dataset(tmp_path)
    Image.new("RGB", (80, 60)).save(image)
    before = mask.read_bytes()
    dialog = DatasetLinkDialog(project)
    try:
        dialog.show()
        dialog.mask_combo.setCurrentIndex(1)
        dialog.image_combo.setCurrentIndex(1)
        assert not dialog.link_button.isEnabled()
        assert dialog.selection is None
        assert "크기 불일치" in dialog.details.text()
        dialog.reject()
        assert dialog.result() == QDialog.DialogCode.Rejected
        assert mask.read_bytes() == before
    finally:
        dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_rename_ui_moves_selected_image_and_its_mask(tmp_path):
    app = QApplication.instance() or QApplication([])
    project, image, mask = _dataset(tmp_path)
    # Restore the original pair before exercising in-app rename.
    image = image.rename(image.with_stem("before"))
    with patch.object(DatasetWidget, "_scan_all"):
        page = DatasetWidget()
        page.set_project(project)
        try:
            with patch("widgets.dataset_widget.QInputDialog.getText", return_value=("renamed", True)):
                page._rename_image(str(image), "train")
            deadline = time.monotonic() + 5
            while page._class_worker is not None and time.monotonic() < deadline:
                app.processEvents()
                QTest.qWait(5)
            assert page._class_worker is None
            assert image.with_stem("renamed").is_file()
            assert mask.with_stem("renamed").is_file()
            assert not scan_dataset(project)["orphan_masks"]
        finally:
            page.close()
            page.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
