"""Segmentation class zero is background; missing masks are background training samples."""
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from core.project import ProjectManager
from core.class_management import ClassManager


def test_first_user_class_is_foreground_and_survives_reopen(tmp_path):
    project = ProjectManager.create_new('First', 'segment', str(tmp_path / 'project'), ['scratch'])
    assert project.data.class_names == ['background', 'scratch']
    assert project.data.num_classes == 2
    path = ProjectManager.save(project)
    assert ProjectManager.load(path).data.class_names == ['background', 'scratch']


def test_first_class_added_to_empty_legacy_project_reserves_background(tmp_path):
    project = ProjectManager.create_new('Empty', 'segment', str(tmp_path / 'project'), [])
    project.data.class_names = []
    project.data.num_classes = 0
    ClassManager.add(project, 'scratch')
    assert project.data.class_names == ['background', 'scratch']


def _data(tmp_path):
    root = tmp_path / 'data'
    for split in ('train', 'val'):
        images, masks = root / 'images' / split, root / 'masks' / split
        images.mkdir(parents=True)
        masks.mkdir(parents=True)
        for name in ('foreground', 'normal'):
            Image.new('RGB', (32, 24), (100, 120, 140)).save(images / (name + '.png'))
        mask = np.zeros((24, 32), np.uint8)
        mask[5:12, 6:16] = 1
        Image.fromarray(mask).save(masks / 'foreground.png')
    return root


def test_missing_masks_are_background_in_train_and_validation(tmp_path):
    from dataset import create_segmentation_loaders, unlabeled_message
    root = _data(tmp_path)
    loaders = create_segmentation_loaders(str(root), input_size=(24, 32), num_classes=2,
                                         flip_prob=0, batch_size=2)
    for loader in loaders:
        assert len(loader.dataset) == 2
        idx = next(i for i, pair in enumerate(loader.dataset.pairs) if Path(pair[0]).stem == 'normal')
        image, mask = loader.dataset[idx]
        assert tuple(image.shape) == (3, 24, 32)
        assert (mask == 0).all()
        assert loader.dataset.class_pixel_counts(2) == [2*24*32 - 70, 70]
    assert '배경' in unlabeled_message(*loaders) and '제외합니다' not in unlabeled_message(*loaders)
    assert not (root / 'masks/train/normal.png').exists()


def test_sam2_background_prompts_and_ignore_pixels_are_retained():
    import torch
    from sam2_trainer import _prompt_targets
    masks = torch.zeros((3, 8, 8), dtype=torch.long)
    masks[0, 2:4, 2:4] = 1
    masks[0, 0, :] = 255
    masks[1, 0, :] = 255
    masks[2, :, :] = 255
    targets, coords, indices = _prompt_targets(masks)
    assert indices.tolist() == [0, 1]
    assert (targets[:, 0, :] == 255).all()
    assert targets[0, 2, 2] == 1 and targets[1, 2, 2] == 0
    assert tuple(coords.shape) == (2, 1, 2)


@pytest.mark.parametrize('names,expected', [([], ['background']), (['background', 'scratch'], ['background', 'scratch']),
                                          (['배경', '결함'], ['배경', '결함']), (['good', 'scratch'], ['good', 'scratch'])])
def test_explicit_background_is_not_duplicated(tmp_path, names, expected):
    p = ProjectManager.create_new('Names', 'segment', str(tmp_path / 'p'), names)
    assert p.data.class_names == expected


def test_detection_first_class_still_uses_zero(tmp_path):
    p = ProjectManager.create_new('Detect', 'detect', str(tmp_path / 'p'), ['scratch'])
    assert p.data.class_names == ['scratch']


def test_class_limit_includes_reserved_background(tmp_path):
    with pytest.raises(ValueError, match='255'):
        ProjectManager.create_new('TooMany', 'segment', str(tmp_path / 'p'), [f'c{i}' for i in range(255)])
    assert not (tmp_path / 'p').exists()


def test_legacy_repair_preserves_other_ids_restores_drawn_zero_and_backs_up(tmp_path):
    from core.segmentation_classes import repair_first_class, needs_first_class_repair
    from core.mask_annotations import MaskDocument
    p = ProjectManager.create_new('Legacy', 'segment', str(tmp_path / 'p'), ['background', 'dent'])
    p.data.class_names = ['scratch', 'dent']  # old broken vocabulary
    project_path = ProjectManager.save(p)
    image = Path(p.data.train_dir) / 'a.png'
    Image.new('RGB', (16, 16), 'gray').save(image)
    base = np.zeros((16, 16), np.uint8)
    base[1:3, 1:3] = 1
    base[0, :] = 255
    document = MaskDocument(16, 16, 2, base=base, shapes=[
        {'kind': 'polygon', 'class_id': 0, 'points': [.5, .5, .9, .5, .9, .9, .5, .9]}])
    mask = Path(p.data.root) / 'masks/train/a.png'
    mask.write_bytes(document.encode())
    before = mask.read_bytes()
    assert needs_first_class_repair(p)
    result = repair_first_class(p)
    assert p.data.class_names == ['background', 'dent', 'scratch']
    assert result['restored_regions'] == 1 and Path(result['backup']).is_dir()
    assert any(f.read_bytes() == before for f in Path(result['backup']).iterdir() if f.is_file())
    assert ProjectManager.load(project_path).data.class_names == p.data.class_names
    values = np.array(Image.open(mask))
    assert (values[1:3, 1:3] == 1).all() and (values[0, :] == 255).all()
    assert values[10, 10] == 2 and values[4, 4] == 0
    assert not needs_first_class_repair(p)
    assert repair_first_class(p) == {'changed': False, 'restored_regions': 0}


def test_legacy_repair_rolls_back_on_save_failure(tmp_path, monkeypatch):
    from core.segmentation_classes import repair_first_class
    p = ProjectManager.create_new('Legacy', 'segment', str(tmp_path / 'p'), ['background', 'dent'])
    p.data.class_names = ['scratch', 'dent']
    saved = Path(ProjectManager.save(p))
    before = saved.read_bytes()
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr(ProjectManager, 'save', fail)
    with pytest.raises(OSError, match='disk full'):
        repair_first_class(p)
    assert p.data.class_names == ['scratch', 'dent'] and p.data.num_classes == 2
    assert saved.read_bytes() == before


@pytest.mark.parametrize('problem', ['rgb', 'size', 'class', 'corrupt'])
def test_bad_masks_are_never_treated_as_background(tmp_path, problem):
    from dataset import SegmentationDataset
    root = _data(tmp_path)
    bad = root / 'masks/train/normal.png'
    if problem == 'corrupt':
        bad.write_bytes(b'not an image')
    elif problem == 'rgb':
        Image.new('RGB', (32, 24)).save(bad)
    else:
        Image.new('L', (31, 24) if problem == 'size' else (32, 24), 9 if problem == 'class' else 0).save(bad)
    ds = SegmentationDataset(str(root / 'images/train'), str(root / 'masks/train'),
                             input_size=(8, 8), num_classes=2, is_train=False)
    with pytest.raises((ValueError, OSError)):
        ds[1]


def test_small_invalid_pixel_cannot_disappear_during_resize(tmp_path):
    from dataset import SegmentationDataset
    root = _data(tmp_path)
    mask = np.zeros((24, 32), np.uint8)
    mask[0, 0] = 9
    Image.fromarray(mask).save(root / 'masks/train/foreground.png')
    ds = SegmentationDataset(str(root / 'images/train'), str(root / 'masks/train'),
                             input_size=(2, 2), num_classes=2, is_train=False)
    with pytest.raises(ValueError, match='클래스 범위'):
        ds[0]


def test_polygon_only_is_not_silently_used_as_background(tmp_path):
    from spatial_data import semantic_training_pairs
    root = _data(tmp_path)
    label = root / 'labels/train/normal.txt'
    label.parent.mkdir(parents=True)
    label.write_text('1 0.1 0.1 0.9 0.1 0.9 0.9\n')
    with pytest.raises(ValueError, match='폴리곤'):
        semantic_training_pairs(root / 'images/train', root / 'masks/train')


def test_crop_keeps_automatic_background_and_foreground(tmp_path):
    from crop_dataset import prepare_crop_dataset
    from dataset import create_segmentation_loaders
    root = _data(tmp_path)
    result = Path(prepare_crop_dataset(root, tmp_path / 'run', 'segment',
                                      {'width': 16, 'height': 16}, num_classes=2))
    loaders = create_segmentation_loaders(str(result), input_size=(16, 16), num_classes=2)
    assert all(len(loader.dataset) == 2 for loader in loaders)
    assert not (root / 'masks/train/normal.png').exists()
    assert not (result / 'masks/train/normal.png').exists()
    assert (loaders[0].dataset[1][1] == 0).all()


def test_background_only_and_ignore_only_sam_preflight(tmp_path):
    from dataset import create_segmentation_loaders
    from sam2_trainer import _validate_prompt_masks
    root = _data(tmp_path)
    (root / 'masks/val/foreground.png').unlink()
    train, val = create_segmentation_loaders(str(root), num_classes=2)
    report = _validate_prompt_masks(val, 'val', ['background', 'scratch'])
    assert report['background_images'] == 2 and report['automatic_background_images'] == 2
    Image.new('L', (32, 24), 255).save(root / 'masks/val/foreground.png')
    Image.new('L', (32, 24), 255).save(root / 'masks/val/normal.png')
    _, val = create_segmentation_loaders(str(root), num_classes=2)
    with pytest.raises(ValueError, match='255'):
        _validate_prompt_masks(val, 'val', ['background', 'scratch'])


def test_sam_loss_ignores_excluded_pixels_and_penalizes_false_positive_background():
    import torch
    from sam2_trainer import _loss_and_scores
    target = torch.zeros((1, 8, 8), dtype=torch.long)
    target[:, 0, :] = 255
    logits = torch.full((1, 8, 8), -5., requires_grad=True)
    good, dice, _ = _loss_and_scores(logits, target)
    bad, bad_dice, _ = _loss_and_scores(-logits, target)
    assert good < bad and dice > bad_dice
    good.backward()
    assert (logits.grad[:, 0, :] == 0).all() and torch.isfinite(logits.grad).all()


def test_dataset_reports_automatic_background(tmp_path):
    from core.dataset_editor import scan_dataset
    p = ProjectManager.create_new('Scan', 'segment', str(tmp_path / 'p'), ['scratch'])
    Image.new('RGB', (16, 16)).save(Path(p.data.train_dir) / 'normal.png')
    row = scan_dataset(p)['images'][0]
    assert row['training_target'] == 'automatic_background' and not row['annotated'] and not row['error']


def test_first_class_can_be_drawn_saved_reopened_and_reclassified(tmp_path):
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from core.mask_annotations import MaskDocument
    from widgets.segmentation_annotation import SegmentationAnnotationDialog
    app = QApplication.instance() or QApplication([])
    p = ProjectManager.create_new('Draw', 'segment', str(tmp_path / 'p'), ['scratch', 'dent'])
    image = Path(p.data.train_dir) / 'a.png'
    Image.new('RGB', (80, 60), 'gray').save(image)
    dialog = SegmentationAnnotationDialog(str(image), p.data.class_names, MaskDocument(80, 60, 3), lambda _: {})
    try:
        dialog.show()
        app.processEvents()
        assert dialog._active_class_id() == 1
        assert not dialog.class_combo.model().item(0).isEnabled()
        from PySide6.QtTest import QTest
        from PySide6.QtCore import Qt
        dialog._fit()
        dialog._set_mode('rectangle')
        view = dialog.view
        QTest.mousePress(view.viewport(), Qt.LeftButton, Qt.NoModifier, view.mapFromScene(10, 10))
        QTest.mouseMove(view.viewport(), view.mapFromScene(60, 40))
        QTest.mouseRelease(view.viewport(), Qt.LeftButton, Qt.NoModifier, view.mapFromScene(60, 40))
        app.processEvents()
        assert dialog.document.render()[20, 20] == 1
        mask = Path(p.data.root) / 'masks/train/a.png'
        mask.write_bytes(dialog.document.encode())
        from core.mask_annotations import load_mask_document
        reopened = load_mask_document(p, image, 'train')
        assert reopened.render()[20, 20] == 1
        dialog._select(dialog._indices()[0])
        dialog._reclass(dialog.selected_indices, 2)
        assert dialog.document.render()[20, 20] == 2
        dialog._reclass(dialog.selected_indices, 1)
        assert dialog.document.render()[20, 20] == 1
    finally:
        dialog.dirty = False
        dialog.reject()
        dialog.deleteLater()
        app.processEvents()


def test_workspace_add_first_class_to_empty_project(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication, QInputDialog
    from core.mask_annotations import MaskDocument
    from widgets.segmentation_annotation import SegmentationAnnotationDialog
    app = QApplication.instance() or QApplication([])
    p = ProjectManager.create_new('Empty', 'segment', str(tmp_path / 'p'), [])
    p.data.class_names, p.data.num_classes = [], 0
    image = Path(p.data.train_dir) / 'a.png'
    Image.new('RGB', (20, 20)).save(image)
    def add(name):
        ClassManager.add(p, name)
        return p.data.class_names
    dialog = SegmentationAnnotationDialog(str(image), [], MaskDocument(20, 20, 1), lambda _: {}, add_class=add)
    monkeypatch.setattr(QInputDialog, 'getText', lambda *args: ('scratch', True))
    try:
        dialog._add_class()
        assert dialog.names == ['background', 'scratch']
        assert dialog.document.classes == 2 and dialog._active_class_id() == 1
    finally:
        dialog.dirty = False
        dialog.reject()
        dialog.deleteLater()
        app.processEvents()


def test_unlabeled_pairing_keeps_duplicate_and_orphan_guards(tmp_path):
    from spatial_data import semantic_training_pairs
    root = _data(tmp_path)
    Image.new('L', (32, 24), 0).save(root / 'masks/train/orphan.png')
    with pytest.raises(ValueError, match='이미지 누락'):
        semantic_training_pairs(root / 'images/train', root / 'masks/train')
    (root / 'masks/train/orphan.png').unlink()
    Image.new('L', (32, 24), 0).save(root / 'images/train/normal.jpg')
    with pytest.raises(ValueError, match='이름 중복'):
        semantic_training_pairs(root / 'images/train', root / 'masks/train')


def test_mask_directory_cannot_be_replaced_by_a_file(tmp_path):
    from spatial_data import semantic_training_pairs
    images, masks = tmp_path / 'images', tmp_path / 'masks'
    images.mkdir()
    Image.new('RGB', (16, 16)).save(images / 'a.png')
    masks.write_text('not a folder')
    with pytest.raises(ValueError, match='폴더 경로'):
        semantic_training_pairs(images, masks)


def test_legacy_repair_does_not_guess_foreground_from_plain_zero_pixels(tmp_path):
    from core.segmentation_classes import repair_first_class
    p = ProjectManager.create_new('Legacy', 'segment', str(tmp_path / 'p'), ['background', 'dent'])
    p.data.class_names = ['scratch', 'dent']
    ProjectManager.save(p)
    Image.new('L', (16, 16)).save(Path(p.data.train_dir) / 'a.png')
    mask = Path(p.data.root) / 'masks/train/a.png'
    Image.new('L', (16, 16), 0).save(mask)
    before = mask.read_bytes()
    result = repair_first_class(p)
    assert result['restored_regions'] == 0 and mask.read_bytes() == before
    assert p.data.class_names == ['background', 'dent', 'scratch']


def test_legacy_repair_keeps_class_scoped_erase_holes(tmp_path):
    from core.segmentation_classes import repair_first_class
    from core.mask_annotations import MaskDocument
    p = ProjectManager.create_new('Legacy', 'segment', str(tmp_path / 'p'), ['background'])
    p.data.class_names = ['scratch']
    ProjectManager.save(p)
    Image.new('L', (32, 32)).save(Path(p.data.train_dir) / 'a.png')
    document = MaskDocument(32, 32, 1, shapes=[
        {'kind':'polygon', 'class_id':0, 'points':[.1,.1,.9,.1,.9,.9,.1,.9]},
        {'kind':'stroke', 'operation':'erase', 'class_id':0, 'scope_class':0,
         'points':[.5,.25,.5,.75], 'width':4}])
    path = Path(p.data.root) / 'masks/train/a.png'
    path.write_bytes(document.encode())
    repair_first_class(p)
    values = np.array(Image.open(path))
    assert values[16, 10] == 1 and values[16, 16] == 0
