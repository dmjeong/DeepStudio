"""Drive the real desktop/subprocess path, not a mocked generator."""

import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "gui")]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from tests.test_segmentation_synthesis import make_fixture


def test_desktop_process_review_publish_restore(tmp_path, monkeypatch):
    from core import job_manager
    from core.project import ProjectManager
    from PySide6.QtWidgets import QApplication
    from widgets.defect_gen_widget import DefectGenWidget

    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("DEEP_STUDIO_DESKTOP_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(job_manager, "_manager", None)
    project = make_fixture(tmp_path / "project")
    widget = DefectGenWidget()
    widget.set_project(project)
    widget.resize(1400, 950)
    widget.show()
    seg = widget.segmentation
    assert widget.tabs.currentWidget() is seg
    assert seg.classes.rowCount() == 5

    def wait():
        deadline = time.monotonic() + 40
        while seg._worker is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert seg._worker is None
        assert seg._result["status"] == "completed", seg.message.text()

    try:
        seg.analyze()
        assert widget._run_project is project
        widget.set_project(None)
        assert widget.project is project
        wait()
        assert seg.profiles.count() == 1
        assert seg.classes.item(0, 1).text() == "1"
        seg.generate(True)
        wait()
        assert seg.samples.count() == 5
        assert seg.generated._source_pixmap.width() == 480
        seg.overlay.setChecked(False)
        seg.focus_defect()
        # Use the real modal annotation editor and its asynchronous save callback.
        import numpy as np
        from PIL import Image
        from PySide6.QtCore import QTimer

        edits = {"count": 0, "error": ""}
        deadline = time.monotonic() + 10

        def continue_edit():
            dialog = app.activeModalWidget()
            if time.monotonic() > deadline:
                edits["error"] = dialog.status.text()
                dialog.dirty = False
                dialog.reject()
                return
            if dialog.worker or dialog.draft_worker or dialog.dirty:
                QTimer.singleShot(20, continue_edit)
                return
            if edits["count"] == 2:
                dialog.reject()
                return
            add_and_save(dialog)

        def add_and_save(dialog):
            pixels = np.zeros((240, 480), bool)
            y = 190 + edits["count"] * 10
            pixels[y : y + 5, 300:310] = True
            dialog._create(dialog.document.raster_shape(pixels, 2))
            edits["count"] += 1
            dialog.save_button.click()
            QTimer.singleShot(20, continue_edit)

        QTimer.singleShot(100, lambda: add_and_save(app.activeModalWidget()))
        seg.edit_mask()
        assert not edits["error"], edits["error"]
        current = seg.selected()
        assert current["status"] == "pending" and current["revision"] == 2
        folder = seg.store().directory("batches", seg.batch["id"])
        revised = np.array(Image.open(seg.store().file(folder, current["mask"])))
        assert revised[192, 305] == 2 and revised[202, 305] == 2
        assert widget.item_bar.isHidden() and widget.message.isHidden()
        for index in range(5):
            seg.samples.setCurrentRow(index)
            seg.review("approved")
        seg.publish()
        wait()
        assert seg.output_path and Path(seg.output_path).is_file()
        new = ProjectManager.load(seg.output_path)
        assert len(list(Path(new.data.train_dir).glob("*.png"))) == 7
        widget.grab().save(str(tmp_path / "seg-synthesis-ui.png"))
        restored = DefectGenWidget()
        restored.set_project(project)
        assert restored.segmentation.samples.count() == 5
        assert all(
            v["status"] == "approved" for v in restored.segmentation.batch["samples"]
        )
        # Policy changes must trigger re-analysis rather than silently use old patches.
        seg.classes.cellWidget(0, 3).setCurrentIndex(2)
        with pytest.raises(ValueError, match="다시"):
            seg.generate()
        seg.classes.cellWidget(0, 3).setCurrentIndex(0)
        # A classification project retains the existing four tabs.
        other = ProjectManager.create_new(
            "cls", "classify", str(tmp_path / "cls"), ["a", "b"]
        )
        restored.set_project(other)
        assert restored.tabs.count() == 4
        restored.close()
    finally:
        if seg._worker:
            seg._worker.stop()
            seg._worker.wait(30000)
            app.processEvents()
        job_manager.desktop_manager().close()
        widget.close()
        app.processEvents()
