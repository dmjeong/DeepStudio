"""Original/result pairing uses file identity, independent of Grad-CAM and list order."""
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from PIL import Image
import pytest

pytest.importorskip('PySide6')
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from core.inference_types import InferenceResult
from widgets.inference_widget import InferenceWidget


@pytest.fixture
def view(tmp_path):
    app = QApplication.instance() or QApplication([])
    widget = InferenceWidget()
    paths = []
    for folder, color in (('first', (210, 10, 20)), ('second', (10, 180, 30))):
        path = tmp_path / folder / 'same.png'
        path.parent.mkdir()
        Image.new('RGB', (30, 20), color).save(path)
        paths.append(str(path))
    widget.model = object()
    widget._batch_images = paths
    widget._current_image = paths[0]
    widget._clear_results()
    yield widget, paths, app
    widget._heatmap_timer.stop()
    widget._preview_cache.clear()
    widget.deleteLater()
    app.processEvents()


def color(label):
    return label._source_pixmap.toImage().pixelColor(0, 0).getRgb()[:3]


@pytest.mark.parametrize("task", ("classify", "detect", "segment", "anomaly"))
def test_no_gradcam_single_run_updates_both_panels(view, task):
    widget, paths, _ = view

    def infer(path):
        with Image.open(path) as image:
            widget._original_preview_rgb = np.array(image)
        widget._display_numpy_image(np.full((20, 30, 3), 55, np.uint8))
        return InferenceResult(path, 'ok', task, 'crack')

    with patch.object(widget, '_run_custom_inference', side_effect=infer):
        widget._run_single_inference(paths[1])
    widget.compare_check.setChecked(True)
    assert color(widget.original_label) == (10, 180, 30)
    assert color(widget.image_label) == (55, 55, 55)


def test_compare_uses_selected_snapshot_after_sort_and_source_change(view):
    widget, paths, _ = view
    expected = {}
    for i, path in enumerate(paths):
        original = np.array(Image.open(path))
        expected[path] = tuple(original[0, 0])
        widget._preview_cache.put(path, original + 1, original=original)
        result = InferenceResult(path, 'ok', 'segment', f'crack {i}')
        widget._inference_results[path] = result
        widget.result_review.update_result(result)
    widget.result_review.table.sortByColumn(1, Qt.SortOrder.DescendingOrder)
    Image.new('RGB', (30, 20), 'white').save(paths[1])
    for path in paths[::-1] + paths:
        widget.result_review.select_path(path)
        widget.compare_check.setChecked(False)
        widget.compare_check.setChecked(True)
        assert widget._current_image == path
        assert color(widget.original_label) == expected[path]
        assert color(widget.image_label) == tuple(v + 1 for v in expected[path])


def test_new_project_or_missing_source_never_retains_old_original(view):
    widget, paths, _ = view
    widget._current_image = None
    widget._batch_images = []
    widget._clear_results()
    widget.compare_check.setChecked(True)
    assert widget.original_label._source_pixmap.isNull()
    widget._current_image = str(Path(paths[0]).parent / 'missing.png')
    widget._display_cached_result(widget._current_image)
    assert widget.original_label._source_pixmap.isNull()
    assert widget.image_label._source_pixmap.isNull()


def test_cache_rejects_another_images_heatmap(view):
    widget, paths, _ = view
    heatmap = {'image_path': paths[0], 'original': np.zeros((20, 30, 3), np.uint8),
               'activation': np.zeros((20, 30), np.float32), 'kind': 'Grad-CAM', 'info': ''}
    with pytest.raises(ValueError, match='이미지 경로'):
        widget._preview_cache.put(paths[1], heatmap['original'], heatmap)
    assert widget._preview_cache.get(paths[1]) is None
