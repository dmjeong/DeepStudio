"""Image navigation in the annotation workspace: release, thumbnail reuse, prefetch, tiles."""

import gc
from pathlib import Path
import os
import weakref
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QDialog

from core.mask_annotations import MaskDocument
from widgets import annotation_io
from widgets.annotation_io import ImagePrefetcher, ThumbnailLoader
from widgets.segmentation_annotation import SegmentationAnnotationDialog


def _app():
    return QApplication.instance() or QApplication([])


def _segment_project(tmp_path, count=4):
    from core.project import ProjectManager

    project = ProjectManager.create_new("seg", "segment", str(tmp_path), ["background", "part"])
    paths = []
    for index in range(count):
        path = os.path.join(project.data.train_dir, f"{index}.png")
        Image.new("L", (64, 48), 40 * index).save(path)
        paths.append(path)
    return project, paths


def _run_navigation(project, paths, on_workspace):
    """데이터셋 화면에서 작업실을 열고, 작업실이 바뀔 때마다 on_workspace(창, 작업실, 순번)를 부른다."""
    from widgets.annotation_navigation import AnnotationWindow
    from widgets.dataset_widget import DatasetWidget

    app = _app()
    seen = []

    def poll():
        windows = [w for w in app.topLevelWidgets() if isinstance(w, AnnotationWindow) and w.isVisible()]
        if not windows:
            return
        window = windows[0]
        workspace = window.workspace
        # 작업실마다 한 번 (지워진 작업실의 id()는 다음 작업실이 다시 쓸 수 있어 표시를 객체에 단다)
        if workspace is None or workspace.property("navigation_test_seen"):
            return
        workspace.setProperty("navigation_test_seen", True)
        seen.append(len(seen))
        on_workspace(window, workspace, len(seen) - 1)

    with patch.object(DatasetWidget, "_scan_all"):
        page = DatasetWidget()
        page.set_project(project)
        page._dataset_index = {"images": [{"path": p, "split": "train", "classes": []} for p in paths]}
        timer = QTimer()
        timer.timeout.connect(poll)
        timer.start(5)
        QTimer.singleShot(0, lambda: (page._open_image_preview(paths[0]), app.quit()))
        app.exec()
        timer.stop()
    return page, seen


def _next_or_close(workspace):
    # 실제 [다음]: '배경만 있음' 확인이면 빈 마스크를 작업 스레드에서 저장한 뒤 이동한다.
    workspace.empty_confirm.setChecked(True)
    if workspace.navigation[0] + 1 < workspace.navigation[1]:
        workspace._navigate(1)
    else:
        workspace.dirty = False
        QDialog.reject(workspace)


def test_navigating_images_releases_each_previous_workspace(tmp_path):
    """이미지를 넘길 때마다 이전 작업실(원본 해상도 버퍼)이 바로 지워진다. 창 안에 작업실은 늘 하나."""
    project, paths = _segment_project(tmp_path)
    alive, released = [], []
    previous = {}

    def on_workspace(window, workspace, _number):
        previous_canvas = previous.get("canvas")
        previous["canvas"] = weakref.ref(workspace.view)  # 약한 참조만: 테스트가 작업실을 붙잡지 않게

        def measure():
            if previous_canvas is not None:  # 바로 전 작업실은 이미 지워졌다
                released.append(previous_canvas() is None)
            alive.append(len(window.findChildren(SegmentationAnnotationDialog)))
            QTimer.singleShot(0, lambda: _next_or_close(workspace))
        QTimer.singleShot(50, measure)

    page, seen = _run_navigation(project, paths, on_workspace)
    page.close()
    page.deleteLater()
    _app().processEvents()
    assert len(seen) == 4
    assert alive == [1, 1, 1, 1]
    assert released == [True, True, True]


@pytest.mark.parametrize("maximized", [False, True])
def test_previous_next_keep_one_window_that_never_closes(tmp_path, maximized):
    """[이전]/[다음]은 창을 닫았다 열지 않는다. 창 하나에서 작업실만 바뀌고, 크기·최대화·패널 배치가 그대로다."""
    from PySide6.QtCore import QEvent, QObject, QRect
    from widgets.annotation_navigation import AnnotationWindow

    project, paths = _segment_project(tmp_path)
    app = _app()
    events, windows, geometries, states, layouts = [], [], [], [], []

    class Recorder(QObject):
        def eventFilter(self, obj, event):
            if isinstance(obj, AnnotationWindow) and event.type() in (QEvent.Type.Show, QEvent.Type.Hide):
                events.append("show" if event.type() == QEvent.Type.Show else "hide")
            return False

    def on_workspace(window, workspace, number):
        windows.append((id(window), int(window.winId())))
        if number == 0:  # 사용자가 첫 이미지에서 창을 옮기고·키우고, 패널을 숨기고, 대비를 올린다
            window.setGeometry(QRect(70, 80, 640, 420))  # 800×600 오프스크린 화면 안
            if maximized:
                window.showMaximized()
            workspace.inspector.hide()
            workspace.inspector.setCurrentIndex(0)
            workspace.contrast.setValue(40)
        geometries.append(window.geometry().getRect())
        states.append(window.isMaximized())
        layouts.append((workspace.inspector.isVisibleTo(workspace), workspace.inspector.currentIndex(),
                        workspace.contrast.value(), workspace.windowTitle() == window.windowTitle(),
                        workspace.view.hasFocus() or not window.isActiveWindow()))
        QTimer.singleShot(30, lambda: _next_or_close(workspace))

    recorder = Recorder()
    app.installEventFilter(recorder)
    try:
        page, seen = _run_navigation(project, paths, on_workspace)
    finally:
        app.removeEventFilter(recorder)
    assert len(seen) == 4
    # 창은 처음 한 번 뜨고 마지막 닫기에서만 숨는다: 이미지를 넘기는 동안 사라지는 순간이 없다.
    assert events == ["show", "hide"], events
    assert len(set(windows)) == 1  # 같은 창 객체, 같은 네이티브 창
    # 크기·위치·최대화 상태가 이미지를 넘겨도 그대로
    assert len(set(geometries[1:])) == 1 and set(states) == {maximized}
    if not maximized:
        assert geometries[1] == (70, 80, 640, 420)
    # 숨긴 패널·고른 탭·대비가 다음 작업실로 이어지고, 창 제목은 지금 이미지
    assert all(layout[:4] == (False, 0, 40, True) for layout in layouts[1:]), layouts
    # ... 그리고 다음에 작업실을 열 때도 같은 크기·위치
    assert page._annotation_geometry is not None
    page.close()
    page.deleteLater()
    app.processEvents()


def test_thumbnail_jump_and_unreadable_next_image(tmp_path):
    """썸네일 더블클릭 점프도 같은 창에서 바뀐다. 다음 이미지를 열 수 없으면 창을 닫고 이유를 알린다."""
    project, paths = _segment_project(tmp_path, count=4)
    Path(paths[3]).write_bytes(b"not an image")  # 4번째 파일이 깨졌다
    positions, warnings = [], []

    def on_workspace(window, workspace, number):
        positions.append(workspace.navigation[0])
        if number == 0:
            QTimer.singleShot(30, lambda: workspace._navigate_to(2))  # 0 → 2 점프
        else:
            def step():
                workspace.empty_confirm.setChecked(True)
                workspace._navigate(1)  # 2 → 3: 열 수 없는 이미지
            QTimer.singleShot(30, step)

    with patch("widgets.dataset_widget.QMessageBox.warning", lambda *args: warnings.append(args[2])):
        page, seen = _run_navigation(project, paths, on_workspace)
    assert positions == [0, 2]
    assert len(warnings) == 1 and "다음 이미지를 열 수 없어" in warnings[0] and "3.png" in warnings[0]
    page.close()
    page.deleteLater()
    _app().processEvents()


@pytest.mark.parametrize("task", ["detect", "obb"])
def test_box_editors_also_switch_images_inside_one_window(tmp_path, task):
    """탐지 작업실과 OBB 편집도 같은 창 안에서 이미지가 바뀐다."""
    from PySide6.QtCore import QEvent, QObject
    from core.project import ProjectManager
    from widgets.annotation_navigation import AnnotationWindow

    project = ProjectManager.create_new("box", task, str(tmp_path), ["part"])
    paths = []
    for index in range(3):
        path = os.path.join(project.data.train_dir, f"{index}.png")
        Image.new("RGB", (64, 48), (40 * index, 0, 0)).save(path)
        paths.append(path)
    app = _app()
    events, names = [], []

    class Recorder(QObject):
        def eventFilter(self, obj, event):
            if isinstance(obj, AnnotationWindow) and event.type() in (QEvent.Type.Show, QEvent.Type.Hide):
                events.append(event.type())
            return False

    def on_workspace(window, workspace, _number):
        names.append(type(workspace).__name__)

        def step():
            if task == "detect":
                _next_or_close(workspace)
            elif workspace.next_button.isEnabled():
                workspace._navigate(1)
            else:
                workspace.dirty = False
                QDialog.reject(workspace)
        QTimer.singleShot(30, step)

    recorder = Recorder()
    app.installEventFilter(recorder)
    try:
        page, seen = _run_navigation(project, paths, on_workspace)
    finally:
        app.removeEventFilter(recorder)
    assert len(seen) == 3 and len(events) == 2  # 창은 한 번 뜨고 한 번 닫힌다
    assert set(names) == {"DetectionAnnotationDialog" if task == "detect" else "OrientedBoxAnnotationDialog"}
    page.close()
    page.deleteLater()
    app.processEvents()


def test_thumbnails_are_reused_across_windows_until_the_file_changes(tmp_path):
    _app()
    path = tmp_path / "a.png"
    Image.new("RGB", (320, 240), "gray").save(path)
    annotation_io._thumbnails.clear()
    reads = []
    real_read = annotation_io.QImageReader.read

    def counting_read(self):
        reads.append(1)
        return real_read(self)

    def run_once():
        loader = ThumbnailLoader([(0, str(path))])
        results = []
        loader.ready.connect(results.append)
        loader.run()  # synchronously, as the pool would
        return results[0][1]

    with patch.object(annotation_io.QImageReader, "read", counting_read):
        first = run_once()
        second = run_once()
        assert len(reads) == 1 and first.size() == second.size() and not first.isNull()
        Image.new("RGB", (320, 240), "white").save(path)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        run_once()
        assert len(reads) == 2


def test_closing_a_workspace_stops_its_remaining_thumbnails(tmp_path):
    app = _app()
    image = tmp_path / "a.png"
    Image.new("RGB", (64, 48)).save(image)
    dialog = SegmentationAnnotationDialog(image, ["background", "part"], MaskDocument(64, 48, 2),
                                          lambda _: None)
    worker = dialog.thumbnail_worker
    dialog.dirty = False
    QDialog.reject(dialog)
    assert worker.isInterruptionRequested()
    dialog.deleteLater()
    app.processEvents()


def test_prefetched_image_is_used_only_for_the_unchanged_file(tmp_path):
    _app()
    path = tmp_path / "next.png"
    pixels = (np.arange(48 * 64) % 251).astype(np.uint8).reshape(48, 64)
    Image.fromarray(pixels).save(path)
    annotation_io._prefetched.clear()
    ImagePrefetcher.request([str(path)])
    image = ImagePrefetcher.take(str(path))
    assert image is not None and image == QImage(str(path))
    assert ImagePrefetcher.take(str(path)) is None  # handed over once, not kept twice

    ImagePrefetcher.request([str(path)])
    ImagePrefetcher._pool.waitForDone()
    Image.fromarray(255 - pixels).save(path)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert ImagePrefetcher.take(str(path)) is None  # stale pixels never shown


def test_workspace_opens_from_the_prefetched_image(tmp_path):
    app = _app()
    paths = []
    for index in range(2):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (64, 48), (60 * index, 0, 0)).save(path)
        paths.append(str(path))
    annotation_io._prefetched.clear()
    first = SegmentationAnnotationDialog(paths[0], ["background", "part"], MaskDocument(64, 48, 2),
                                         lambda _: None, navigation=(0, 2), image_paths=paths)
    ImagePrefetcher._pool.waitForDone()  # the neighbour decodes while the user edits
    key = annotation_io._file_key(paths[1])
    assert annotation_io._prefetched.get(key) is not None
    second = SegmentationAnnotationDialog(paths[1], ["background", "part"], MaskDocument(64, 48, 2),
                                          lambda _: None, navigation=(1, 2), image_paths=paths)
    assert annotation_io._prefetched.get(key) is None  # consumed instead of decoding again
    assert second.pixmap.toImage().pixelColor(0, 0).red() == 60
    for dialog in (first, second):
        dialog.dirty = False
        QDialog.reject(dialog)
        dialog.deleteLater()
    app.processEvents()


def test_background_tiles_are_not_painted(tmp_path):
    app = _app()
    image = tmp_path / "large.png"
    Image.new("L", (1024, 1024)).save(image)
    document = MaskDocument(1024, 1024, 3)
    document.shapes.append({"kind": "polygon", "class_id": 2, "object_id": "a",
                            "points": [0.02, 0.02, 0.2, 0.02, 0.2, 0.2]})
    dialog = SegmentationAnnotationDialog(image, ["background", "part", "defect"], document,
                                          lambda _: None)
    painted = {key for key, tile in dialog.view.tiles.items() if not tile.pixmap().isNull()}
    assert painted == {(0, 0)}  # a 1024² mask has 16 tiles; 15 are pure background
    # A region that becomes empty again clears its tile instead of keeping stale colour.
    dialog.view._set_overlay_pixels(np.zeros((1024, 1024), np.uint8), (0, 0, 256, 256))
    assert dialog.view.tiles[(0, 0)].pixmap().isNull()
    dialog.dirty = False
    QDialog.reject(dialog)
    dialog.deleteLater()
    app.processEvents()
    gc.collect()


@pytest.mark.parametrize(
    ("values", "classes", "valid"),
    [([0, 1, 2], 3, True), ([0, 2, 255], 3, True), ([0, 3], 3, False), ([0, 3, 255], 3, False)],
)
def test_mask_class_range_check(values, classes, valid):
    from core.mask_annotations import _validate_pixels

    pixels = np.array([values], dtype=np.uint8)
    if valid:
        assert np.array_equal(_validate_pixels(pixels, len(values), 1, classes), pixels)
    else:
        with pytest.raises(ValueError):
            _validate_pixels(pixels, len(values), 1, classes)
