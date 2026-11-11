"""Actual mouse clicks, failure diagnostics and retry of annotation analysis."""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "gui")]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from tests.test_segmentation_synthesis import make_fixture


@pytest.fixture
def page(tmp_path, monkeypatch):
    from core import job_manager
    from PySide6.QtWidgets import QApplication
    from widgets.segmentation_synthesis_widget import SegmentationSynthesisWidget

    app = QApplication.instance() or QApplication([])
    monkeypatch.setenv("DEEP_STUDIO_DESKTOP_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(job_manager, "_manager", None)
    project = make_fixture(tmp_path / "project")
    widget = SegmentationSynthesisWidget()
    widget.set_project(project)
    widget.resize(1280, 720)
    widget.show()
    app.processEvents()
    yield app, widget, project
    if widget._worker:
        widget._worker.stop()
        widget._worker.wait(30000)
        app.processEvents()
    widget._notice.close()
    job_manager.desktop_manager().close()
    widget.close()
    app.processEvents()


def click_and_wait(app, widget, button):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    deadline = time.monotonic() + 30
    while widget._worker and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert widget._worker is None
    app.processEvents()


def test_missing_masks_visible_failure_then_retry_generates(page):
    from core.segmentation_synthesis_store import SynthesisStore
    from PySide6.QtCore import QPoint, Qt

    app, widget, project = page
    # An earlier successful analysis must not silently take over after failure.
    SynthesisStore(project).analyze()
    widget.refresh()
    masks = [
        p
        for split in ("train", "val")
        for p in (Path(project.data.root) / "masks" / split).glob("*.png")
    ]
    saved = {p: p.read_bytes() for p in masks}
    for p in masks:
        p.unlink()
    click_and_wait(app, widget, widget.analyze_button)
    assert widget._result["status"] == "failed"
    assert widget.profiles.currentData() is None
    assert all(widget.classes.item(i, 1).text() == "0" for i in range(5))
    assert widget._notice.isVisible()
    assert "정답 파일을 찾지 못했습니다" in widget.message.text()
    assert (
        widget.message.mapTo(widget, QPoint()).y()
        < widget.controls.mapTo(widget, QPoint()).y()
    )
    assert "정답 파일 0장" in widget.analysis_summary.text()
    widget.copy_button.click()
    copied = app.clipboard().text()
    assert str(Path(project.data.root) / "masks/train") in copied
    assert "이미지 2장" in copied
    # Mouse click also reports missing profile, rather than silently doing nothing.
    widget._notice.close()
    click_and_wait(app, widget, widget.preview_button)
    assert widget._notice.isVisible() and "먼저 프로젝트 정답" in widget.message.text()
    widget._notice.close()
    for path, payload in saved.items():
        path.write_bytes(payload)
    click_and_wait(app, widget, widget.analyze_button)
    assert widget._result["status"] == "completed"
    assert all(
        widget.classes.item(i, 0).checkState() == Qt.CheckState.Checked
        for i in range(5)
    )
    click_and_wait(app, widget, widget.preview_button)
    assert widget._result["status"] == "completed", widget.message.text()
    assert widget.samples.count() == 5


def test_only_val_references_select_on_switch_and_keep_manual_choices(page):
    from PySide6.QtCore import Qt

    app, widget, project = page
    for path in (Path(project.data.root) / "masks/train").glob("*.png"):
        Image.fromarray(np.zeros((240, 480), np.uint8)).save(path)
    click_and_wait(app, widget, widget.analyze_button)
    assert widget._result["status"] == "completed"
    assert all(
        widget.classes.item(i, 0).checkState() == Qt.CheckState.Unchecked
        for i in range(5)
    )
    assert "TRAIN 사용 가능 0개" in widget.message.text()
    widget.split.setCurrentIndex(widget.split.findData("val"))
    assert all(
        widget.classes.item(i, 0).checkState() == Qt.CheckState.Checked
        for i in range(5)
    )
    widget.classes.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    widget.refresh()
    assert widget.classes.item(0, 0).checkState() == Qt.CheckState.Unchecked
    click_and_wait(app, widget, widget.preview_button)
    assert widget.samples.count() == 4
    assert not widget.publish_button.isEnabled()


@pytest.mark.parametrize("problem", ["background", "bad_mask"])
def test_analysis_failure_explains_reason_and_records_counts(page, problem):
    app, widget, project = page
    root = Path(project.data.root)
    if problem == "background":
        for split in ("train", "val"):
            for path in (root / "masks" / split).glob("*.png"):
                Image.fromarray(np.zeros((240, 480), np.uint8)).save(path)
        expected = "배경/제외"
    else:
        Image.fromarray(np.full((240, 480), 70, np.uint8)).save(
            root / "masks/train/0.png"
        )
        expected = "클래스"
    click_and_wait(app, widget, widget.analyze_button)
    assert widget._result["status"] == "failed"
    assert expected in widget.message.text()
    assert widget._notice.isVisible()
    assert widget._analysis["status"] == "failed"
    assert "마스크 폴더:" in widget.diagnostic_text()
    from core.job_manager import desktop_manager

    jobs = desktop_manager().list()
    assert any(
        (desktop_manager().directory(j["id"]) / "segmentation-analysis.json").is_file()
        for j in jobs
    )


def test_refresh_failure_is_not_reported_as_success(page, monkeypatch):
    app, widget, _ = page

    def fail():
        raise OSError("분석 결과 읽기 실패")

    monkeypatch.setattr(widget, "refresh", fail)
    click_and_wait(app, widget, widget.analyze_button)
    assert widget._result["status"] == "completed"
    assert "분석 결과 읽기 실패" in widget.message.text()
    assert "분석 완료" not in widget.message.text()
    assert widget._notice.isVisible()


def test_duplicate_images_analyze_and_generate_without_warning_dialog(page):
    app, widget, project = page
    source = Path(project.data.train_dir) / "0.png"
    mask = Path(project.data.root) / "masks/train/0.png"
    for split in ("val", "test"):
        (Path(getattr(project.data, split + "_dir")) / "0.png").write_bytes(
            source.read_bytes()
        )
        (Path(project.data.root) / "masks" / split / "0.png").write_bytes(
            mask.read_bytes()
        )
    click_and_wait(app, widget, widget.analyze_button)
    assert widget._result["status"] == "completed", widget.message.text()
    assert widget._analysis["duplicate_images"] == 2
    assert not widget._notice.isVisible()
    assert all(int(widget.classes.item(i, 1).text()) > 0 for i in range(5))
    click_and_wait(app, widget, widget.preview_button)
    assert widget._result["status"] == "completed", widget.message.text()
    assert widget.samples.count() == 5 and not widget._notice.isVisible()
