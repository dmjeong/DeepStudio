"""Registered classes remain visible, without counting background as a legacy defect."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from core.dataset_editor import scan_dataset
from core.mask_annotations import MaskDocument
from core.project import ProjectManager


@pytest.fixture
def page():
    pytest.importorskip('PySide6')
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication
    from widgets.dataset_widget import DatasetWidget

    app = QApplication.instance() or QApplication([])
    with patch.object(DatasetWidget, '_scan_all'):
        widget = DatasetWidget()
        widget.resize(1200, 800)
        widget.show()

        def load(project):
            widget.set_project(project)
            widget._scan_worker = None
            widget._scan_ready(scan_dataset(project), {})
            app.processEvents()

        yield app, widget, load
        widget.close()
        widget.deleteLater()
        # processEvents() alone does not drain deferred QObject destruction.
        # Release parented buttons while QApplication is alive, not at atexit.
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def _table(widget):
    return {widget.stats_table.item(i, 0).text(): widget.stats_table.item(i, 1).text()
            for i in range(widget.stats_table.rowCount())}


def _shown(widget):
    return {Path(thumb.image_path).stem for thumb in widget._thumbs['train']}


def _options(widget):
    combo = widget.tab_filters['train']
    return [combo.itemText(i) for i in range(combo.count())]


def test_new_project_lists_all_classes_even_without_images(tmp_path, page):
    _, widget, load = page
    project = ProjectManager.create_new('New', 'segment', str(tmp_path / 'p'), ['scratch', 'dent'])
    load(ProjectManager.load(ProjectManager.save(project)))
    assert list(_table(widget))[:3] == ['background', 'scratch', 'dent']
    assert _options(widget)[1:4] == ['background', 'scratch', 'dent']
    assert _table(widget)['scratch'] == '0개 · 0장'
    assert '등록 클래스 3개' in widget.summary_label.text()
    assert not widget.repair_class_btn.isVisible()


@pytest.mark.parametrize('background', ['background', '배경', 'good'])
def test_background_filter_and_count_use_actual_pixels_or_automatic_background(tmp_path, page, background):
    _, widget, load = page
    project = ProjectManager.create_new('Background', 'segment', str(tmp_path / 'p'), [background, 'scratch'])
    image_root = Path(project.data.train_dir)
    mask_root = Path(project.data.root) / 'masks/train'
    for name in ['mixed', 'background', 'foreground', 'ignore', 'automatic', 'invalid']:
        Image.new('L', (16, 16), 80).save(image_root / f'{name}.png')
    for name, value in [('mixed', 0), ('background', 0), ('foreground', 1), ('ignore', 255)]:
        pixels = np.full((16, 16), value, np.uint8)
        if name == 'mixed':
            pixels[4:8, 4:8] = 1
        Image.fromarray(pixels).save(mask_root / f'{name}.png')
    (mask_root / 'invalid.png').write_bytes(b'broken mask')
    load(project)
    assert _table(widget)[background] == '배경 · 3장'
    widget.tab_filters['train'].setCurrentText(background)
    assert _shown(widget) == {'mixed', 'background', 'automatic'}
    widget.tab_filters['train'].setCurrentText('scratch')
    assert _shown(widget) == {'mixed', 'foreground'}
    from widgets.dataset_widget import region_classes
    assert region_classes(project) == ['scratch']  # Editing/deletion still excludes background.


@pytest.mark.parametrize('legacy_names', [['scratch'], ['scratch', 'dent']])
def test_reopened_legacy_first_class_is_visible_and_not_misreported(tmp_path, page, legacy_names):
    _, widget, load = page
    project = ProjectManager.create_new('Legacy', 'segment', str(tmp_path / 'p'), ['background'])
    project.data.class_names = legacy_names
    project.data.num_classes = len(legacy_names)
    saved = Path(ProjectManager.save(project))
    project = ProjectManager.load(str(saved))
    image = Path(project.data.train_dir) / 'part.png'
    Image.new('L', (16, 16), 80).save(image)
    document = MaskDocument(16, 16, len(legacy_names), shapes=[
        {'kind': 'polygon', 'class_id': 0, 'points': [.2, .2, .8, .2, .8, .8, .2, .8]}])
    mask = Path(project.data.root) / 'masks/train/part.png'
    mask.write_bytes(document.encode())
    before = saved.read_bytes(), mask.read_bytes()
    load(project)
    assert list(_table(widget))[:len(legacy_names)] == legacy_names
    assert 'scratch' in _options(widget)
    assert _table(widget)['scratch'] == '복구 필요'
    assert widget.repair_class_btn.isVisible()
    assert 'scratch' in widget.class_issue_label.text() and '0번' in widget.class_issue_label.text()
    widget.tab_filters['train'].setCurrentText('scratch')
    assert not _shown(widget), 'Raster background must not be reported as a scratch.'
    labels = [widget.tab_grids['train'].itemAt(i).widget() for i in range(widget.tab_grids['train'].count())]
    assert any('첫 클래스 복구' in item.text() for item in labels if hasattr(item, 'text'))
    assert before == (saved.read_bytes(), mask.read_bytes()), 'Viewing does not rewrite stored IDs.'

    # Exercise the actual UI action, including async worker and subsequent class refresh.
    widget.repair_class_btn.click()
    assert widget._class_worker.wait(10000)
    from PySide6.QtCore import QCoreApplication
    QCoreApplication.processEvents()
    assert widget._class_worker is None
    load(project)
    assert not widget.repair_class_btn.isVisible()
    assert not widget.class_issue_label.isVisible()
    assert widget.tab_filters['train'].currentText() == 'scratch'
    assert _shown(widget) == {'part'}
    assert _table(widget)['scratch'] == '1개 · 1장'
    assert _options(widget)[1:1 + len(project.data.class_names)] == project.data.class_names
    assert ProjectManager.load(str(saved)).data.class_names == project.data.class_names


def test_detection_first_class_is_editable_and_counted(tmp_path, page):
    _, widget, load = page
    project = ProjectManager.create_new('Detect', 'detect', str(tmp_path / 'p'), ['scratch'])
    Image.new('L', (16, 16), 80).save(Path(project.data.train_dir) / 'a.png')
    (Path(project.data.root) / 'labels/train/a.txt').write_text('0 .5 .5 .25 .25\n')
    load(project)
    assert _table(widget)['scratch'] == '1개 · 1장'
    widget.tab_filters['train'].setCurrentText('scratch')
    assert _shown(widget) == {'a'}
    from widgets.dataset_widget import region_classes
    assert region_classes(project) == ['scratch']
    assert not widget.repair_class_btn.isVisible()
