"""Distribution-to-loader regression: source, manual assets and partial repair."""
import hashlib
import json
from pathlib import Path
from urllib.error import URLError

import pytest

import builtin_assets as assets
import model_download
import prepare_builtin_assets as prepare
import sam2_assets


@pytest.mark.parametrize("model_id", list(sam2_assets.SAM2_ASSETS))
def test_manual_sam2_is_authenticated_offline_without_manifest(tmp_path, monkeypatch, model_id):
    payload = model_id.encode()
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(sam2_assets.SAM2_SHA256, model_id, digest)
    monkeypatch.setenv("DVS_BUILTIN_ASSETS_DIR", str(tmp_path))
    path = tmp_path / assets.ASSET_FILES[model_id]
    path.parent.mkdir()
    path.write_bytes(payload)
    monkeypatch.setattr(model_download, "download_context", lambda: pytest.fail("offline"))
    assert assets.builtin_asset_path(model_id) == path
    prepare.prepare(tmp_path, [model_id], register_existing=True)
    assert json.loads((tmp_path / 'manifest.json').read_text())['files'][assets.ASSET_FILES[model_id]]['sha256'] == digest
    path.write_bytes(b'broken')
    with pytest.raises(RuntimeError, match='무결성'):
        assets.builtin_asset_path(model_id)


def test_sam2_network_fallback_keeps_identical_expected_digest(tmp_path, monkeypatch):
    calls = []
    def download(url, destination, sha256):
        calls.append((url, destination, sha256))
        if len(calls) == 1:
            raise URLError('blocked host')
        return destination
    monkeypatch.setattr(model_download, 'cached_https_weights', download)
    sam2_assets.download_sam2_pretrained('sam2_hiera_small', tmp_path)
    assert len(calls) == 2
    assert calls[0][1:] == calls[1][1:]
    assert calls[1][0].startswith('https://huggingface.co/facebook/')


def test_sam2_hash_failure_never_tries_another_source(tmp_path, monkeypatch):
    calls = []
    def download(*args):
        calls.append(args)
        raise ValueError('SHA256 mismatch')
    monkeypatch.setattr(model_download, 'cached_https_weights', download)
    with pytest.raises(ValueError, match='SHA256'):
        sam2_assets.download_sam2_pretrained('sam2_hiera_tiny', tmp_path)
    assert len(calls) == 1


def test_sam2_mirror_errors_preserve_both_download_causes(tmp_path, monkeypatch):
    monkeypatch.setattr(model_download, 'cached_https_weights',
                        lambda *args: (_ for _ in ()).throw(URLError('CERTIFICATE_VERIFY_FAILED')))
    with pytest.raises(RuntimeError) as error:
        sam2_assets.download_sam2_pretrained('sam2_hiera_tiny', tmp_path)
    assert 'dl.fbaipublicfiles.com' in str(error.value)
    assert 'huggingface.co' in str(error.value)
    assert 'CERTIFICATE_VERIFY_FAILED' in str(error.value)


def test_segmentation_group_covers_every_builtin_pretrained_model():
    from core.model_registry import builtin_model_specs
    # Each model's public catalog contract must be represented in preparation.
    ids = {spec.model_id for spec in builtin_model_specs() if spec.task == 'segment' and spec.pretrained_asset}
    expected = {assets.BUILTIN_BACKBONE_ASSETS.get(model, model) for model in ids}
    assert expected == set(assets.SEGMENTATION_ASSET_IDS)


def test_check_default_root_matches_loader_independent_of_cwd(tmp_path, monkeypatch, capsys):
    root = tmp_path / 'app asset root'
    monkeypatch.setenv('DVS_BUILTIN_ASSETS_DIR', str(root))
    monkeypatch.chdir(tmp_path)
    assert prepare.main(['--group', 'segment', '--check']) == 1
    output = capsys.readouterr().out
    assert str(root) in output and '[check] 0/6 ready' in output
    for model_id in assets.SEGMENTATION_ASSET_IDS:
        assert f'[failed] {model_id}:' in output
    assert not root.exists(), 'Read-only check must not create files.'


def test_default_prepare_uses_exact_runtime_root(tmp_path, monkeypatch):
    root = tmp_path / 'weights'
    monkeypatch.setenv('DVS_BUILTIN_ASSETS_DIR', str(root))
    calls = []
    monkeypatch.setattr(prepare, 'prepare', lambda *args, **kwargs: calls.append((args, kwargs)))
    assert prepare.main(['--group', 'segment']) == 0
    assert calls[0][0] == (assets.builtin_asset_root(), list(assets.SEGMENTATION_ASSET_IDS))
    assert prepare.main([]) == 0
    assert calls[1][0] == (root, list(assets.ASSET_FILES))


def test_relative_asset_override_is_anchored_to_app_not_working_directory(tmp_path, monkeypatch):
    monkeypatch.setenv('DVS_BUILTIN_ASSETS_DIR', 'local_weights')
    root = Path(assets.__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    assert assets.builtin_asset_root() == root / 'local_weights'
    monkeypatch.chdir(root / 'gui')
    assert assets.builtin_asset_root() == root / 'local_weights'
    import sys
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, 'executable', str(tmp_path / 'bin' / 'app.exe'))
    assert assets.builtin_asset_root() == tmp_path / 'bin' / 'local_weights'


def test_sam2_missing_help_never_offers_nonexistent_scratch_mode():
    message = assets.missing_asset_help('sam2_hiera_tiny')
    assert 'prepare_model_weights.bat' in message
    assert '무작위 초기화 모드는 지원하지 않습니다' in message
    assert "모드를 '무작위 초기화로 시작'으로 바꾸면" not in message


def test_new_batch_and_builder_share_runtime_root():
    root = Path(__file__).resolve().parents[1]
    script = (root / 'gui/prepare_model_weights.bat').read_text()
    assert 'cd /d "%~dp0"' in script
    assert '--group segment' in script and '--check' in script
    assert '--output' not in script and 'powershell' not in script.lower()
    assert 'if errorlevel 1 goto :failed' in script
    builder = (root / 'gui/build.bat').read_text()
    assert builder.index('"weights"') < builder.index('prepare_torch_runtime.py')
    assert '"setup"' in builder
    assert 'prepare_builtin_assets.py --output' not in builder
    packaging = (root / 'gui/build_exe.py').read_text()
    assert 'f"{builtin_asset_root()}{os.pathsep}builtin_assets"' in packaging


def test_training_mode_reports_missing_asset_before_start(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication
    from core.project import ProjectManager
    from widgets.training_widget import TrainingWidget
    QApplication.instance() or QApplication([])
    monkeypatch.setenv('DVS_BUILTIN_ASSETS_DIR', str(tmp_path / 'empty'))
    project = ProjectManager.create_new('test', 'segment', str(tmp_path / 'project'), ['background', 'chip'])
    widget = TrainingWidget()
    widget.set_project(project)
    for model_id, asset_id in [('unet_resnet18', 'resnet18'), ('sam2_hiera_tiny', 'sam2_hiera_tiny')]:
        widget.model_id_combo.setCurrentIndex(widget.model_id_combo.findData(model_id))
        assert f'기본 가중치 준비 필요: {asset_id}' in widget.mode_help.text()
        assert 'prepare_model_weights.bat' in widget.mode_help.text()
    widget.deleteLater()


def test_training_reports_corrupt_weights_as_corrupt_not_absent(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox
    from core.project import ProjectManager
    from widgets.training_widget import TrainingWidget
    QApplication.instance() or QApplication([])
    monkeypatch.setenv('DVS_BUILTIN_ASSETS_DIR', str(tmp_path))
    path = tmp_path / assets.ASSET_FILES['resnet18']
    path.parent.mkdir()
    path.write_bytes(b'damaged weights')
    project = ProjectManager.create_new('test', 'segment', str(tmp_path / 'project'), ['background', 'chip'])
    widget = TrainingWidget()
    widget.set_project(project)
    widget.model_id_combo.setCurrentIndex(widget.model_id_combo.findData('unet_resnet18'))
    widget.mode_combo.setCurrentIndex(widget.mode_combo.findData('builtin_finetune'))
    assert '파일 있음' in widget.mode_help.text()
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args[1:]))
    widget._start_training()
    assert warnings[0][0] == '사전학습 가중치 확인 실패'
    assert 'SHA256 불일치' in warnings[0][1] and str(path) in warnings[0][1]
    assert widget._run_project is None
    widget.deleteLater()
