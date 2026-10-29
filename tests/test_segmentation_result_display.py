"""Visual result contract: untouched background, labelled foreground, no automatic heatmap."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from PIL import Image
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'gui'), str(ROOT / 'python')]
from core.spatial_preview import segmentation_preview
from core.inference_engine import InferenceEngine
from core.inference_cache import InferencePreviewCache

NAMES = ['background', '스크래치', '찍힘']


@pytest.fixture
def app():
    from PySide6.QtWidgets import QApplication
    instance = QApplication.instance() or QApplication([])
    yield instance
    instance.processEvents()


def sample():
    rng = np.random.default_rng(12)
    original = rng.integers(70, 170, (120, 160, 3), dtype=np.uint8)
    mask = np.zeros((120, 160), dtype=np.int64)
    mask[30:45, 20:75] = 1
    mask[60:80, 90:115] = 1
    mask[85:110, 35:65] = 2
    mask[95:100, 45:50] = 0  # a background hole inside a predicted region
    return original, mask


def test_background_and_holes_unchanged_and_objects_have_class_names():
    original, mask = sample()
    saved = original.copy()
    details = {}
    preview, counts = segmentation_preview(mask, original, 3, class_names=NAMES, details_out=details)
    np.testing.assert_array_equal(preview[mask == 0], original[mask == 0])
    np.testing.assert_array_equal(original, saved)
    assert (preview[mask > 0] != original[mask > 0]).any()
    assert details['region_counts'] == {1: 2, 2: 1}
    assert details['region_count'] == 3
    assert {r['name'] for r in details['segmentation_regions']} == {'스크래치', '찍힘'}
    assert all(r['class_id'] != 0 for r in details['segmentation_regions'])
    assert sum(counts.values()) == mask.size
    assert sum(r['area_px'] for r in details['segmentation_regions']) == np.count_nonzero(mask)


@pytest.mark.parametrize('kind', ['background', 'one_pixel', 'edge', 'full'])
def test_small_and_edge_foreground_do_not_color_neighbors(kind):
    original = np.full((25, 31, 3), 120, dtype=np.uint8)
    mask = np.zeros((25, 31), dtype=np.int64)
    if kind == 'one_pixel': mask[12, 13] = 1
    if kind == 'edge': mask[0:5, 0:8] = 1
    if kind == 'full': mask[:] = 2
    details = {}
    preview, _ = segmentation_preview(mask, original, 3, class_names=NAMES, details_out=details)
    np.testing.assert_array_equal(preview[mask == 0], original[mask == 0])
    assert details['region_count'] == (0 if kind == 'background' else 1)
    if kind != 'background': assert (preview[mask > 0] != original[mask > 0]).any()


def test_label_limit_does_not_suppress_any_predicted_mask():
    mask = np.zeros((80, 80), dtype=np.int64)
    mask[::3, ::3] = 1
    original = np.full((80, 80, 3), 120, dtype=np.uint8)
    details = {}
    preview, counts = segmentation_preview(mask, original, 3, class_names=NAMES, details_out=details)
    assert len(details['segmentation_regions']) == 256
    assert details['region_count'] == counts[1] == 729
    assert details['region_labels_truncated']
    assert np.all(np.any(preview[mask == 1] != original[mask == 1], axis=1))
    np.testing.assert_array_equal(preview[mask == 0], original[mask == 0])


class KnownSegmentation(torch.nn.Module):
    task = 'segment'
    in_channels = 3
    num_classes = 3
    def __init__(self, mask):
        super().__init__()
        self.register_buffer('logits', torch.nn.functional.one_hot(torch.as_tensor(mask), 3).permute(2, 0, 1).float()[None])
    def forward(self, images):
        return self.logits


def engine_for(mask, crop=None):
    return InferenceEngine({'model': KnownSegmentation(mask), '_infer_device': 'cpu',
        '_input_size': mask.shape, '_normalization': ([0, 0, 0], [1, 1, 1]),
        'class_names': NAMES, '_center_crop': crop}, gradcam=False, runtime='pytorch')


def test_no_gt_still_displays_predictions_and_crop_coordinates(tmp_path):
    original, mask = sample()
    path = tmp_path / 'part.png'; Image.fromarray(original).save(path)
    crop = {'width': 80, 'height': 60}
    small_mask = mask[30:90, 40:120]
    engine = engine_for(small_mask, crop)
    result = engine.infer(str(path))
    assert result.status == 'ok', result.error
    assert result.details['segmentation_evaluation']['status'] == 'unavailable'
    assert result.details['segmentation_regions']
    assert '스크래치' in result.summary and 'background' not in result.summary
    preview = engine._current_preview_rgb
    np.testing.assert_array_equal(preview[:30], original[:30])
    np.testing.assert_array_equal(preview[:, :40], original[:, :40])
    for region in result.details['segmentation_regions']:
        x1, y1, x2, y2 = region['bbox']
        assert 40 <= x1 < x2 <= 120 and 30 <= y1 < y2 <= 90


@pytest.mark.parametrize('process', [False, True])
def test_segmentation_load_defaults_to_prediction_view(app, process):
    from widgets.inference_widget import InferenceWidget
    w = InferenceWidget()
    w.show()
    try:
        w.gradcam_checkbox.setChecked(True)
        if process:
            w._model_inspected({'status': 'completed', 'output': {'weights': 'model.pt',
                'task': 'segment', 'class_names': NAMES, 'input_size': [120, 160], 'device': 'cpu'}})
        else:
            cam = SimpleNamespace(release=lambda: None)
            w._activate_model(path='model.pt', device=torch.device('cpu'), model=KnownSegmentation(sample()[1]),
                              gradcam=cam, class_names=NAMES)
        assert not w.gradcam_checkbox.isChecked()
        assert w.region_names_check.isVisible()
        assert w.heatmap_lower_spin.isHidden()
        assert w.gradcam_checkbox.isEnabled()
        w.gradcam_checkbox.setChecked(True)  # explicit opt-in remains available
        assert not w.heatmap_lower_spin.isHidden()
    finally:
        w.close(); w.deleteLater(); app.processEvents()


def test_cached_results_and_original_comparison_without_source_file(app, tmp_path):
    from core.inference_worker import InferenceWorker
    from widgets.inference_widget import InferenceWidget
    original, mask = sample()
    path = tmp_path / 'part.png'; Image.fromarray(original).save(path)
    cache = InferencePreviewCache()
    engine = engine_for(mask)
    worker = InferenceWorker(engine, [str(path)], cache)
    results = []
    worker.result_ready.connect(results.append)
    worker.run()
    assert not worker.error
    result = results[0]
    np.testing.assert_array_equal(cache.get(str(path))['original'], original)
    path.unlink()  # preview and comparison must use the saved images
    w = InferenceWidget()
    w.resize(1280, 900)
    try:
        w._preview_cache = cache
        w._inference_results[str(path)] = result
        w._batch_images = [str(path)]
        w._batch_results = {0: result}
        w.gradcam_checkbox.setChecked(False)
        w.show(); w._display_cached_result(str(path)); app.processEvents()
        assert len(w.image_label._regions) == 3
        assert w.image_label._region_label_rects()
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFontMetrics
        font = QFontMetrics(w.image_label._badge_font(), w.image_label)
        for rect, text, _ in w.image_label._region_label_rects():
            assert font.elidedText(text, Qt.TextElideMode.ElideRight, rect.width() - 8) == text
        assert w.image_label._regions[0]['name'] == '스크래치'
        a = w.original_label._source_pixmap.toImage().pixelColor(0, 0).getRgb()[:3]
        b = w.image_label._source_pixmap.toImage().pixelColor(0, 0).getRgb()[:3]
        assert a == b == tuple(original[0, 0])
        assert w.image_label._pixel_source is not None
        w._switch_view_mode('grid'); app.processEvents()
        assert len(w._grid_cells[0].thumb_label._regions) == 3
        w.region_names_check.setChecked(False)
        assert not w.image_label._region_label_rects()
        assert not w._grid_cells[0].thumb_label._regions_visible
        w.region_names_check.setChecked(True)
        w._on_grid_cell_clicked(0); app.processEvents()
        assert len(w.image_label._regions) == 3
        # Drawn text is a UI overlay, not baked into model data or stored image pixels.
        np.testing.assert_array_equal(cache.get(str(path))['preview'], engine._current_preview_rgb)
    finally:
        w.close(); w.deleteLater(); worker.deleteLater(); app.processEvents()


def test_region_labels_follow_zoom_and_clear_with_new_image(app):
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QPixmap, QColor
    from widgets.result_image_label import ResultImageLabel
    label = ResultImageLabel()
    label.resize(500, 500)
    pixmap = QPixmap(400, 400); pixmap.fill(QColor('white'))
    label.setPixmap(pixmap)
    label.set_regions([{'name': '스크래치', 'bbox': [160, 160, 200, 200], 'color': '#00d5b0'}], [400, 400])
    label.show(); app.processEvents()
    try:
        first = label._region_label_rects()[0][0]
        label.set_view(2, QPointF(10, 20)); app.processEvents()
        second = label._region_label_rects()[0][0]
        assert second.x() != first.x() and second.height() == first.height()
        assert label._source_pixmap.toImage().pixelColor(160, 160) == QColor('white')
        label.setText('이미지 없음')
        assert label._regions == [] and label._region_label_rects() == []
    finally:
        label.close(); label.deleteLater(); app.processEvents()
