"""DINOv2 integration tests: offline CI uses random weights, real smoke is separate."""
import hashlib
import json

import numpy as np
import pytest
import torch

from builtin_models import build_builtin_model, load_builtin_checkpoint, make_builtin_checkpoint


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('channels,shape', [(1, (57, 71)), (3, (56, 84)), (3, (32, 33))])
def test_shapes_and_native_resolution_detail_gradient(channels, shape, monkeypatch):
    monkeypatch.setattr(torch.hub, 'load', lambda *a, **k: pytest.fail('must remain offline'))
    model = build_builtin_model('dinov2_vits14', 3, channels)
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    x = torch.randn(2, channels, *shape)
    observed = []
    handle = model.detail.register_forward_hook(lambda _m, _x, y: observed.append(y.shape[-2:]))
    logits = model(x)
    handle.remove()
    assert logits.shape == (2, 3, *shape)
    assert observed == [shape]
    loss = torch.nn.functional.cross_entropy(logits, torch.randint(0, 3, (2, *shape)))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.detail[0].weight.grad.abs().sum() > 0
    assert model.projections[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.parameters())


def test_gray_conversion_matches_rgb_replication_and_checkpoint_roundtrip():
    model = build_builtin_model('dinov2_vits14', 3, 1).eval()
    raw = torch.rand(1, 1, 42, 56)
    gray = (raw - 0.449) / 0.226
    expected = (raw.repeat(1, 3, 1, 1) - model.rgb_mean) / model.rgb_std
    torch.testing.assert_close(model.backbone_input(gray), expected, atol=1e-6, rtol=1e-6)
    ckpt = make_builtin_checkpoint('dinov2_vits14', model, num_classes=3, in_channels=1,
                                  input_size=(42, 56), class_names=['background', 'scratch', 'dent'])
    restored = load_builtin_checkpoint(ckpt)
    with torch.inference_mode():
        torch.testing.assert_close(model(gray), restored(gray), rtol=0, atol=0)
    assert restored.weight_provenance == model.weight_provenance


def test_wrong_backbone_is_rejected_instead_of_partial_loading(tmp_path):
    path = tmp_path / 'wrong.pth'
    torch.save({'not_a_dino_weight': torch.ones(1)}, path)
    with pytest.raises(ValueError, match='가중치 구조'):
        build_builtin_model('dinov2_vits14', 2, backbone_weights=path)


def test_full_finetuning_reaches_transformer_backbone():
    model = build_builtin_model('dinov2_vits14', 2, 3).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    parameter = model.backbone.blocks[0].attn.qkv.weight
    before = parameter.detach().clone()
    output = model(torch.randn(1, 3, 42, 56))
    loss = torch.nn.functional.cross_entropy(output, torch.randint(0, 2, (1, 42, 56)))
    loss.backward()
    assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, parameter)


def test_builtin_preparation_pin_and_offline_registration(tmp_path, monkeypatch):
    import builtin_assets as assets
    import dinov2_segmentation as dino
    import model_download
    import prepare_builtin_assets as prepare
    payload = b'official fixture'
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(dino, 'DINOV2_SHA256', digest)
    calls = []
    def download(url, destination, expected_sha256):
        calls.append((url, expected_sha256))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return destination
    monkeypatch.setattr(model_download, 'cached_https_weights', download)
    prepare.prepare(tmp_path, ['dinov2_vits14'])
    assert calls == [(dino.DINOV2_URL, digest)]
    (tmp_path / 'manifest.json').unlink()
    prepare.prepare(tmp_path, ['dinov2_vits14'], register_existing=True)
    assert len(calls) == 1
    assert 'dinov2_vits14' in assets.SEGMENTATION_ASSET_IDS
    assert assets.required_pretrained_asset('segment', 'builtin_finetune', 'dinov2_vits14') == 'dinov2_vits14'
    (tmp_path / assets.ASSET_FILES['dinov2_vits14']).write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='무결성'):
        assets.verify_local_asset('dinov2_vits14', tmp_path)


def test_gui_routes_to_builtin_training_and_names_pretraining(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from core.project import ProjectManager
    from core.training_modes import training_capabilities
    from widgets.training_widget import TrainingWidget
    monkeypatch.setenv('DVS_BUILTIN_ASSETS_DIR', str(tmp_path))
    app = QApplication.instance() or QApplication([])
    project = ProjectManager.create_new('DINO segmentation', 'segment', str(tmp_path/'project'), ['background','scratch'])
    project.model.model_id = 'dinov2_vits14'
    widget = TrainingWidget()
    try:
        widget.set_project(project)
        widget.model_id_combo.setCurrentIndex(widget.model_id_combo.findData('dinov2_vits14'))
        assert 'LVD-142M' in widget.mode_combo.itemText(0)
        assert '기본 가중치 준비 필요' in widget.mode_help.text()
        assert '.dvmodel' not in widget.mode_desc.text()
        assert training_capabilities('segment', 'builtin_finetune', model_id='dinov2_vits14')
    finally:
        widget.close(); widget.deleteLater(); app.processEvents()


def test_train_save_resume_and_transfer_with_new_classes(tmp_path):
    from PIL import Image
    from train_builtin import train_builtin
    from core.inference_loading import load_inference_engine
    data = tmp_path / 'data'
    for split in ('train', 'val'):
        for index in range(2):
            image = np.full((42, 42), 30, np.uint8)
            mask = np.zeros((42, 42), np.uint8)
            image[8:26, 11+index:19+index] = 200
            mask[8:26, 11+index:19+index] = 1
            for name, array in [('images', image), ('masks', mask)]:
                folder = data / name / split; folder.mkdir(parents=True, exist_ok=True)
                Image.fromarray(array).save(folder / f'{index}.png')
    args = dict(num_classes=2, in_channels=1, input_size=42, batch_size=2,
                freeze_backbone=True, num_workers=0, horizontal_flip=0, rotation=0,
                color_jitter=0, scheduler_name='none', log=lambda *_: None)
    first = train_builtin('dinov2_vits14', data, epochs=1, class_names=['background','scratch'],
                          output_dir=tmp_path/'first', **args)
    checkpoint = torch.load(first, weights_only=False)
    assert checkpoint['class_names'] == ['background', 'scratch']
    assert checkpoint['training_config']['freeze_backbone'] is True
    engine = load_inference_engine(str(first), gradcam=False, device='cpu', runtime='pytorch')
    result = engine.infer(str(data/'images/val/0.png'))
    assert result.status == 'ok', result.error
    assert result.task == 'segment'
    train_builtin('dinov2_vits14', data, epochs=2, class_names=['background','scratch'],
                  output_dir=tmp_path/'resumed', resume=tmp_path/'first/last.pt', **args)
    last = torch.load(tmp_path/'resumed/last.pt', weights_only=False)
    assert last['epoch'] > checkpoint['epoch']
    transferred = train_builtin('dinov2_vits14', data, epochs=1, class_names=['background','dent'],
                                output_dir=tmp_path/'transfer', initial_weights=first, **args)
    assert torch.load(transferred, weights_only=False)['class_names'] == ['background','dent']
