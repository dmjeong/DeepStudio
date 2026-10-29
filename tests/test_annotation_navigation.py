"""Image navigation in the annotation workspace: release, thumbnail reuse, prefetch, tiles."""

import gc
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


def test_navigating_images_releases_each_previous_workspace(tmp_path):
    """exec() → deleteLater() → next exec() kept every window (and its buffers) alive."""
    from core.project import ProjectManager
    from widgets.dataset_widget import DatasetWidget

    app = _app()
    project = ProjectManager.create_new("seg", "segment", str(tmp_path), ["background", "part"])
    paths = []
    for index in range(4):
        path = os.path.join(project.data.train_dir, f"{index}.png")
        Image.new("L", (64, 48), 40 * index).save(path)
        paths.append(path)
    alive, released = [], []
    previous = {}

    def poll():
        shown = [w for w in app.topLevelWidgets()
                 if isinstance(w, SegmentationAnnotationDialog) and w.isVisible()]
        if not shown or ("dialog" in previous and shown[0] is previous["dialog"]()):
            return
        dialog = shown[0]
        # Weak references only: the test itself must not keep a window alive.
        previous_canvas = previous.get("canvas")
        previous.update(dialog=weakref.ref(dialog), canvas=weakref.ref(dialog.view))

        def measure():
            # The next window is shown just before the previous one hides (no blank moment), so
            # both exist for an instant. Count once that hand-off has finished.
            if previous_canvas is not None:  # the window before this one must be gone
                released.append(previous_canvas() is None)
            alive.append(len(page.findChildren(SegmentationAnnotationDialog)))
            QTimer.singleShot(0, next_image)

        def next_image(d=dialog):
            # The real "next" button: confirming "background only" saves the empty
            # mask on a worker thread first, so release must wait for that thread.
            d.empty_confirm.setChecked(True)
            if d.navigation[0] + 1 < d.navigation[1]:
                d._navigate(1)
            else:
                d.dirty = False
                QDialog.reject(d)
        QTimer.singleShot(50, measure)

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
        page.close()
        page.deleteLater()
    app.processEvents()
    assert alive == [1, 1, 1, 1]
    assert released == [True, True, True]


@pytest.mark.parametrize("maximized", [False, True])
def test_next_image_window_appears_in_place_before_the_previous_one_hides(tmp_path, maximized):
    """[다음] used to hide the window, then open a default-size one elsewhere ("껐다 켜짐")."""
    from PySide6.QtCore import QEvent, QObject, QRect
    from core.project import ProjectManager
    from widgets.dataset_widget import DatasetWidget

    app = _app()
    project = ProjectManager.create_new("seg", "segment", str(tmp_path), ["background", "part"])
    paths = []
    for index in range(4):
        path = os.path.join(project.data.train_dir, f"{index}.png")
        Image.new("L", (64, 48), 40 * index).save(path)
        paths.append(path)
    events, geometries, states = [], [], []

    class Recorder(QObject):
        def eventFilter(self, obj, event):
            if isinstance(obj, SegmentationAnnotationDialog) and event.type() in (QEvent.Type.Show, QEvent.Type.Hide):
                events.append(("show" if event.type() == QEvent.Type.Show else "hide", obj.windowTitle()))
            return False

    recorder = Recorder()
    app.installEventFilter(recorder)
    seen = set()

    def poll():
        # Mark windows on the object itself: id() of a deleted window can be reused by the next one.
        shown = [w for w in app.topLevelWidgets()
                 if isinstance(w, SegmentationAnnotationDialog) and w.isVisible()
                 and not w.property("navigation_test_seen")]
        if not shown:
            return
        dialog = shown[0]
        dialog.setProperty("navigation_test_seen", True)
        seen.add(len(seen))
        if len(seen) == 1:  # the user resizes / moves / maximizes the first window
            dialog.setGeometry(QRect(70, 80, 640, 420))  # fits the 800×600 offscreen screen
            if maximized:
                dialog.showMaximized()
        geometries.append(dialog.geometry().getRect())
        states.append(dialog.isMaximized())

        def next_image(d=dialog):
            d.empty_confirm.setChecked(True)
            if d.navigation[0] + 1 < d.navigation[1]:
                d._navigate(1)
            else:
                d.dirty = False
                QDialog.reject(d)
        QTimer.singleShot(30, next_image)

    try:
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
    finally:
        app.removeEventFilter(recorder)
    # Never a moment without a visible workspace: each new window shows before the old one hides.
    visible = 0
    for position, (kind, _window) in enumerate(events):
        visible += 1 if kind == "show" else -1
        assert visible >= 1 or position == len(events) - 1, events  # only the final close leaves none
    hides = [i for i, e in enumerate(events) if e[0] == "hide"]
    shows = [i for i, e in enumerate(events) if e[0] == "show"]
    assert len(seen) == 4 and len(shows) >= 4
    for n in range(3):  # window n hides only after window n+1 is shown
        assert shows[n + 1] < hides[n], events
    # Size, position and maximized state carry over to every next image.
    assert len(set(geometries[1:])) == 1 and set(states) == {maximized}
    if not maximized:
        assert geometries[1] == (70, 80, 640, 420)
    # ... and to the next time the workspace is opened.
    assert page._annotation_geometry is not None
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
