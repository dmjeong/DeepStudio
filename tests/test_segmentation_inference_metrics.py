"""Native-resolution inference evaluation, count aggregation and desktop presentation."""
from dataclasses import asdict
from pathlib import Path
import json
import sys
import time

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gui'))
sys.path.insert(0, str(ROOT / 'python'))
from core.project import ProjectData, DataConfig
from core.segmentation_evaluation import (evaluate_segmentation, aggregate_evaluations,
                                          project_snapshot, evaluation_lines)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'data'
    for split in ('train', 'val', 'test'):
        (root / 'images' / split / 'lot').mkdir(parents=True)
        (root / 'masks' / split / 'lot').mkdir(parents=True)
    project = ProjectData(task='segment', data=DataConfig(root=str(root),
        train_dir=str(root / 'images/train'), val_dir=str(root / 'images/val'),
        test_dir=str(root / 'images/test'), class_names=['background', 'scratch', 'dent'], num_classes=3))
    def add(target, name='one', split='val'):
        path = root / 'images' / split / 'lot' / (name + '.png')
        Image.new('L', (target.shape[1], target.shape[0]), 100).save(path)
        mask = root / 'masks' / split / 'lot' / path.name
        Image.fromarray(target.astype(np.uint8)).save(mask)
        return path, mask
    return project, add


def evaluate(project, path, pred, shape=None, **kwargs):
    return evaluate_segmentation(pred, path, shape or pred.shape, project.data.class_names, project, **kwargs)


def test_known_metrics_and_json_roundtrip(dataset):
    project, add = dataset
    target = np.array([[0, 0], [1, 1]])
    path, _ = add(target)
    row = evaluate(project, path, np.array([[0, 1], [1, 1]]))
    assert row['status'] == 'ok'
    assert row['per_class'][0]['iou'] == .5
    assert row['per_class'][1]['iou'] == pytest.approx(2 / 3)
    assert row['per_class'][0]['dice'] == pytest.approx(2 / 3)
    assert row['per_class'][1]['dice'] == .8
    assert row['per_class'][2]['iou'] is None
    assert row['mIoU'] == pytest.approx(7 / 12)
    assert row['pixel_accuracy'] == .75
    assert row['sources'][0].endswith('masks/val/lot/one.png')
    assert json.loads(json.dumps(row, allow_nan=False)) == row


def test_pool_counts_not_image_means(dataset):
    project, add = dataset
    path, _ = add(np.zeros((2, 2)), 'small')
    small = evaluate(project, path, np.zeros((2, 2), dtype=int))
    path, _ = add(np.zeros((4, 4)), 'large')
    large = evaluate(project, path, np.ones((4, 4), dtype=int))
    pooled = aggregate_evaluations([small, large, {'status': 'unavailable'}])
    assert pooled['mIoU'] == .2  # 4/20, not image mean .5
    assert pooled['evaluated'] == 2 and pooled['total'] == 3
    assert pooled['per_class'][1]['iou'] is None  # no GT for scratch, false positives penalize background
    large['class_names'] = ['background', 'dent', 'scratch']
    assert aggregate_evaluations([small, large])['status'] == 'error'


@pytest.mark.parametrize('kind', ['missing', 'outside', 'ignored', 'corrupt', 'size', 'rgb', 'class_id', 'names', 'duplicate'])
def test_unavailable_or_invalid_gt_never_becomes_perfect_background(dataset, tmp_path, kind):
    project, add = dataset
    target = np.zeros((4, 4), dtype=np.uint8)
    path, mask = add(target)
    expected = 'error'
    if kind == 'missing':
        mask.unlink(); expected = 'unavailable'
    elif kind == 'outside':
        path = tmp_path / 'one.png'; Image.fromarray(target).save(path); expected = 'unavailable'
    elif kind == 'ignored':
        Image.fromarray(np.full((4, 4), 255, dtype=np.uint8)).save(mask); expected = 'unavailable'
    elif kind == 'corrupt':
        mask.write_bytes(b'broken')
    elif kind == 'size':
        Image.fromarray(target[:2]).save(mask)
    elif kind == 'rgb':
        Image.new('RGB', (4, 4)).save(mask)
    elif kind == 'class_id':
        target[0, 0] = 3; Image.fromarray(target).save(mask)
    elif kind == 'names':
        project = project_snapshot(project)
        project['data']['class_names'] = ['background', 'dent', 'scratch']
    elif kind == 'duplicate':
        Image.fromarray(target).save(mask.with_suffix('.bmp'))
    row = evaluate_segmentation(np.zeros((4, 4), dtype=int), path, (4, 4),
                                 ['background', 'scratch', 'dent'], project)
    assert row['status'] == expected and row['reason']
    assert row.get('mIoU') is None


def test_ignore_and_native_resolution_preserve_tiny_defect(dataset):
    project, add = dataset
    target = np.zeros((4, 4), dtype=np.uint8)
    target[0, 0] = 1
    target[3, 3] = 255
    path, _ = add(target)
    row = evaluate(project, path, np.zeros((1, 1), dtype=int), shape=(4, 4))
    assert row['total_pixels'] == 15
    assert row['gt_area'] == [14, 1, 0]
    assert row['per_class'][1]['iou'] == 0
    assert row['mIoU'] == pytest.approx((14 / 15) / 2)


def test_crop_ignores_outside_region_and_validates_full_target(dataset):
    project, add = dataset
    target = np.ones((6, 6), dtype=np.uint8)
    target[1:5, 1:5] = 2
    path, mask = add(target)
    row = evaluate(project, path, np.full((2, 2), 2), shape=(6, 6), crop_box=(1, 1, 5, 5))
    assert row['mIoU'] == 1 and row['gt_area'] == [0, 0, 16]
    target[0, 0] = 99
    Image.fromarray(target).save(mask)
    assert evaluate(project, path, np.full((2, 2), 2), shape=(6, 6), crop_box=(1, 1, 5, 5))['status'] == 'error'


def test_nested_same_filename_and_polygon_background_annotation(dataset):
    project, add = dataset
    path, mask = add(np.zeros((4, 4), dtype=np.uint8))
    other = path.parent.parent / 'other' / path.name
    other.parent.mkdir(); Image.new('L', (4, 4)).save(other)
    assert evaluate(project, other, np.zeros((4, 4), dtype=int))['status'] == 'unavailable'
    mask.unlink()
    label = Path(project.data.root) / 'labels/val/lot/one.txt'
    label.parent.mkdir(parents=True)
    label.write_text('1 0 0 1 0 1 1 0 1\n')
    assert evaluate(project, path, np.ones((4, 4), dtype=int))['mIoU'] == 1
    label.write_text('')  # explicit saved empty annotation is background
    assert evaluate(project, path, np.zeros((4, 4), dtype=int))['mIoU'] == 1
    label.write_text('bad polygon')
    assert evaluate(project, path, np.zeros((4, 4), dtype=int))['status'] == 'error'


def test_snapshot_does_not_follow_project_mutation(dataset):
    project, _ = dataset
    snapshot = project_snapshot(project)
    project.data.class_names[1] = 'changed'
    assert snapshot['data']['class_names'][1] == 'scratch'
    assert project_snapshot(None)['task'] == ''


def test_engine_evaluation_outside_inference_timer(dataset, monkeypatch):
    import torch
    from core.inference_engine import InferenceEngine
    import core.segmentation_evaluation as evaluation
    project, add = dataset
    path, _ = add(np.ones((16, 16), dtype=np.uint8))
    class Segment(torch.nn.Module):
        task = 'segment'
        in_channels = 1
        num_classes = 3
        def forward(self, images):
            logits = torch.zeros((1, 3, 4, 4))
            logits[:, 1] = 1
            return logits
    engine = InferenceEngine({'model': Segment(), '_infer_device': 'cpu', '_input_size': (16, 16),
        '_normalization': ([0], [1]), 'class_names': project.data.class_names,
        '_evaluation_project': project_snapshot(project)}, gradcam=False, runtime='pytorch')
    real = evaluation.evaluate_segmentation
    def slow_evaluation(*args, **kwargs):
        time.sleep(.08)
        return real(*args, **kwargs)
    monkeypatch.setattr(evaluation, 'evaluate_segmentation', slow_evaluation)
    result = engine.infer(str(path))
    assert result.status == 'ok', result.error
    assert result.details['segmentation_evaluation']['mIoU'] == 1
    assert result.elapsed_sec - result.inference_sec >= .08
    assert engine._current_preview_rgb.shape == (16, 16, 3)


def test_qt_batch_and_selected_metrics_survive_saved_result(dataset):
    from PySide6.QtWidgets import QApplication
    from core.inference_types import InferenceResult
    from core.inference_review import restored_result
    from widgets.inference_review import InferenceReview
    from widgets.inference_widget import InferenceWidget
    app = QApplication.instance() or QApplication([])
    project, add = dataset
    path, _ = add(np.ones((4, 4), dtype=np.uint8))
    report = evaluate(project, path, np.ones((4, 4), dtype=int))
    result = InferenceResult(str(path), 'ok', 'segment', '세그멘테이션', inference_sec=.012,
        details={'pixel_counts': {1: 16}, 'class_names': project.data.class_names,
                 'segmentation_evaluation': report})
    result = restored_result(json.loads(json.dumps(asdict(result))))
    review = InferenceReview()
    widget = InferenceWidget()
    try:
        review.set_context([str(path), 'unlabelled.png'], project)
        review.update_result(result)
        review.update_result(InferenceResult('unlabelled.png', 'ok', 'segment', '세그멘테이션',
            details={'segmentation_evaluation': {'status': 'unavailable', 'reason': '정답 마스크 없음'}}))
        assert review.model.item(0, 7).data(0) == 100
        assert review.model.item(0, 6).data(0) == 12
        assert review.model.item(0, 5).data(0) is None
        assert review.model.item(1, 7).data(0) is None
        assert '정답 마스크 없음' in review.model.item(1, 9).text()
        assert review.segmentation_table.item(1, 0).text() == 'scratch'
        assert review.segmentation_table.item(1, 1).text() == '100.00%'
        assert '정답 비교 1 / 처리 2 / 전체 2장' in review.segmentation_summary.text()
        review.class_filter.setCurrentIndex(review.class_filter.findData('scratch'))
        assert review.proxy.rowCount() == 1
        assert review.model.item(0, 2).text() == 'scratch'  # not the folder name "lot"
        review.class_filter.setCurrentIndex(0)
        review.search.setText('unlabelled')
        assert 'mIoU 100.00%' in review.segmentation_summary.text()  # whole batch, not filtered
        widget._update_task_details(result)
        assert 'scratch: IoU 100.00%' in widget.task_details_label.text()
        assert '정답 픽셀 없음' in widget.task_details_label.text()
        review.set_context([], ProjectData(task='classify'))
        assert review.segmentation_group.isHidden() and review.table.isColumnHidden(7)
        assert review.segmentation_table.rowCount() == 0
    finally:
        review.close(); review.deleteLater()
        widget.close(); widget.deleteLater()
        app.processEvents()


@pytest.mark.parametrize('model_id', ['unet_resnet18', 'deeplabv3plus_resnet34', 'dinov2_vits14'])
def test_real_builtin_forward_metrics_and_worker_persistence(dataset, tmp_path, monkeypatch, model_id):
    import torch
    from builtin_models import build_builtin_model
    from core.inference_engine import InferenceEngine
    import core.inference_loading as loading
    from core.desktop_jobs import inference_result
    from webapp.worker import infer
    project, add = dataset
    target = np.zeros((32, 32), dtype=np.uint8)
    target[8:24, 8:24] = 1
    path, _ = add(target)
    torch.manual_seed(92)
    model = build_builtin_model(model_id, 3, 3).eval()
    engine = InferenceEngine({'model': model, '_infer_device': 'cpu', '_input_size': (32, 32),
        '_normalization': ([0, 0, 0], [1, 1, 1]), 'class_names': project.data.class_names},
        gradcam=False, runtime='pytorch')
    # Use the real model; only checkpoint loading is substituted to avoid large test files.
    monkeypatch.setattr(loading, 'load_inference_engine', lambda *a, **k: engine)
    class Context:
        directory = tmp_path / 'job'
        def cancelled(self): return False
        def emit(self, *args): pass
    Context.directory.mkdir()
    weights = tmp_path / 'identity.pt'; weights.write_bytes(b'identity for manifest hashing')
    output = infer(Context(), {'weights': str(weights), 'images': [str(path)], 'gradcam': False,
                              'project': project_snapshot(project)})
    assert output['status'] == 'completed'
    stored = json.loads((Context.directory / 'results/0.json').read_text())
    result = inference_result(stored)
    report = result.details['segmentation_evaluation']
    assert report['status'] == 'ok' and report['total_pixels'] == 1024
    # Independent confusion counts from the unchanged model output.
    with torch.inference_mode():
        pred = model(torch.full((1, 3, 32, 32), 100 / 255)).argmax(1)[0].numpy()
    expected = [np.logical_and(pred == c, target == c).sum() /
                np.logical_or(pred == c, target == c).sum() for c in (0, 1)]
    assert report['mIoU'] == pytest.approx(float(np.mean(expected)))
    assert result.inference_sec is not None and stored['cache_ready']
