"""Desktop worker, live classes, paired original preview, portable publication."""
import time
from pathlib import Path

import pytest
from core import job_manager
from core.project import ProjectManager
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QApplication
from tests.test_project_generation import make_project
from widgets.defect_gen_widget import DefectGenWidget


@pytest.mark.parametrize("task", ["classify", "segment", "detect", "obb", "anomaly"])
def test_worker_generate_review_save_all_tasks(tmp_path, monkeypatch, task):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("DEEP_STUDIO_DESKTOP_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(job_manager, "_manager", None)
    project = make_project(tmp_path, task)
    page = DefectGenWidget()
    page.set_project(project)
    page.resize(1280, 1100)
    page.show()
    gen = page.project_generation
    assert page.tabs.currentWidget() is gen
    assert not page.item_bar.isVisible()
    assert gen.classes.rowCount() == (2 if task in {"segment", "anomaly"} else 3)
    row = next(i for i in range(gen.classes.rowCount()) if gen.classes.item(i, 0).text() == "코팅부족")
    gen.classes.item(row, 0).setCheckState(Qt.CheckState.Checked)
    gen.classes.cellWidget(row, 1).setCurrentIndex(gen.classes.cellWidget(row, 1).findData("color"))
    gen.classes.cellWidget(row, 2).setValue(1)

    def wait():
        deadline = time.monotonic() + 60
        while gen._worker is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert gen._worker is None, "Generation worker did not finish"
        assert gen._result["status"] == "completed", gen.message.text()

    try:
        gen.generate_button.click()
        assert page._run_project is project
        wait()
        assert gen.samples.count() == 1
        assert gen.compare.original.width() == 128
        assert gen.compare.original.height() == 96
        gen.overlay.setChecked(True)
        app.processEvents()
        assert not gen.compare.overlay.pixmap().isNull()
        assert "원본:" in gen.message.text()
        gen.samples.item(0).setCheckState(Qt.CheckState.Checked)
        gen.publish()
        wait()
        restored = ProjectManager.load(gen.output_path)
        assert Path(restored.data.train_dir).exists()
        assert restored.task == task
        assert restored.data.class_names == project.data.class_names
        assert page._run_project is None
        # New window reads persistent project candidates; no generation needed.
        second = DefectGenWidget()
        try:
            second.set_project(project)
            assert second.project_generation.samples.count() == 1
        finally:
            second.close()
            second.deleteLater()
    finally:
        if gen._worker is not None:
            gen.stop()
            gen._worker.wait(60000)
            app.processEvents()
        job_manager.desktop_manager().close()
        page.close()
        page.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
