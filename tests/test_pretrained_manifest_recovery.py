"""A failed unrelated download must not strand complete pretrained weights."""
import hashlib
import json
from pathlib import Path

import pytest

import builtin_assets as assets
import prepare_builtin_assets as prepare


def _official_fixture(monkeypatch, tmp_path, model='resnet18', payload=b'official weight fixture'):
    digest = hashlib.sha256(payload).hexdigest()
    relative = f'torchvision/{model}-{digest[:8]}.pth'
    monkeypatch.setitem(assets.ASSET_FILES, model, relative)
    monkeypatch.setenv('DVS_BUILTIN_ASSETS_DIR', str(tmp_path))
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path, digest


def test_manual_official_file_loads_offline_without_manifest(tmp_path, monkeypatch):
    path, _ = _official_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr('model_download.download_context', lambda: pytest.fail('network attempted'))
    assert assets.builtin_asset_path('resnet18') == path
    assert not (tmp_path / 'manifest.json').exists(), 'Runtime must not write into an installed app.'


def test_manual_official_file_with_missing_manifest_entry_is_checked(tmp_path, monkeypatch):
    path, _ = _official_fixture(monkeypatch, tmp_path)
    manifest = tmp_path / 'manifest.json'
    original = json.dumps({'schema_version': 1, 'files': {}})
    manifest.write_text(original)
    assert assets.builtin_asset_path('resnet18') == path
    assert manifest.read_text() == original
    path.write_bytes(b'corrupt')
    with pytest.raises((RuntimeError, ValueError), match='SHA256|무결성'):
        assets.builtin_asset_path('resnet18')


@pytest.mark.parametrize('manifest', [
    'bad json', '[]', '{"schema_version":1,"files":[]}',
])
def test_corrupt_manifest_is_not_bypassed(tmp_path, monkeypatch, manifest):
    _official_fixture(monkeypatch, tmp_path)
    (tmp_path / 'manifest.json').write_text(manifest)
    with pytest.raises(RuntimeError, match='manifest'):
        assets.builtin_asset_path('resnet18')


def test_existing_manifest_hash_mismatch_is_not_bypassed(tmp_path, monkeypatch):
    path, _ = _official_fixture(monkeypatch, tmp_path)
    relative = path.relative_to(tmp_path).as_posix()
    (tmp_path / 'manifest.json').write_text(json.dumps({'schema_version': 1, 'files': {
        relative: {'sha256': '0' * 64, 'bytes': path.stat().st_size}}}))
    with pytest.raises(RuntimeError, match='무결성'):
        assets.builtin_asset_path('resnet18')


def test_non_official_filename_cannot_bypass_manifest(tmp_path, monkeypatch):
    relative = assets.ASSET_FILES['sam2_hiera_tiny']
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b'arbitrary checkpoint')
    monkeypatch.setenv('DVS_BUILTIN_ASSETS_DIR', str(tmp_path))
    with pytest.raises(RuntimeError, match='manifest'):
        assets.builtin_asset_path('sam2_hiera_tiny')


def test_each_download_success_is_published_even_when_another_model_fails(tmp_path, monkeypatch):
    import builtin_assets
    visited = []
    def download(url, cache):
        visited.append(Path(url).name)
        if 'resnet34-' in url:
            # Previous successful entry is usable before this next download completes.
            assert builtin_assets.builtin_asset_path('resnet18').is_file()
            raise OSError('TLS connection failed')
        path = Path(cache) / Path(url).name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'downloaded')
        return path
    monkeypatch.setenv('DVS_BUILTIN_ASSETS_DIR', str(tmp_path))
    monkeypatch.setattr(prepare, 'cached_imagenet_weights', download)
    with pytest.raises(RuntimeError, match='resnet34'):
        prepare.prepare(tmp_path, ['resnet18', 'resnet34', 'resnet50'])
    assert len(visited) == 3
    files = json.loads((tmp_path / 'manifest.json').read_text())['files']
    assert set(files) == {assets.ASSET_FILES['resnet18'], assets.ASSET_FILES['resnet50']}
    assert builtin_assets.builtin_asset_path('resnet50').is_file()


def test_manual_official_registration_does_not_download_other_models(tmp_path, monkeypatch):
    path, digest = _official_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(prepare, 'cached_imagenet_weights', lambda *a: pytest.fail('network attempted'))
    prepare.prepare(tmp_path, ['resnet18'], register_existing=True)
    entry = json.loads((tmp_path / 'manifest.json').read_text())['files'][path.relative_to(tmp_path).as_posix()]
    assert entry == {'sha256': digest, 'bytes': path.stat().st_size}
    assert assets.builtin_asset_path('resnet18') == path


def test_manual_registration_rejects_missing_or_corrupt_file(tmp_path, monkeypatch):
    path, _ = _official_fixture(monkeypatch, tmp_path)
    path.write_bytes(b'incomplete')
    with pytest.raises(RuntimeError, match='resnet18'):
        prepare.prepare(tmp_path, ['resnet18'], register_existing=True)
    assert not (tmp_path / 'manifest.json').exists()
    path.unlink()
    with pytest.raises(RuntimeError, match='resnet18'):
        prepare.prepare(tmp_path, ['resnet18'], register_existing=True)


def test_unknown_only_does_not_create_output_directory(tmp_path):
    output = tmp_path / 'not-created'
    with pytest.raises(SystemExit):
        prepare.prepare(output, ['unknown'])
    assert not output.exists()


def test_resnet_asset_revision_matches_the_training_recipe():
    from torchvision import models
    from builtin_models import IMAGENET_WEIGHTS
    for name in ('resnet18', 'resnet34', 'resnet50'):
        enum, revision = IMAGENET_WEIGHTS[name]
        assert Path(assets.ASSET_FILES[name]).name == Path(getattr(getattr(models, enum), revision).url).name


def test_atomic_manifest_write_failure_preserves_previous_file(tmp_path, monkeypatch):
    _official_fixture(monkeypatch, tmp_path)
    prepare.prepare(tmp_path, ['resnet18'], register_existing=True)
    original = (tmp_path / 'manifest.json').read_bytes()
    _official_fixture(monkeypatch, tmp_path, 'resnet34', b'new model')
    def fail_replace(*args):
        raise OSError('disk full')
    monkeypatch.setattr(prepare.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='disk full'):
        prepare.prepare(tmp_path, ['resnet34'], register_existing=True)
    assert (tmp_path / 'manifest.json').read_bytes() == original
    assert not list(tmp_path.glob('.manifest-*.tmp'))


def test_interrupted_download_keeps_previous_completed_entry(tmp_path, monkeypatch):
    visited = []
    def download(url, cache):
        visited.append(url)
        if len(visited) == 2:
            raise KeyboardInterrupt()
        path = Path(cache) / Path(url).name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'first downloaded file')
        return path
    monkeypatch.setattr(prepare, 'cached_imagenet_weights', download)
    with pytest.raises(KeyboardInterrupt):
        prepare.prepare(tmp_path, ['resnet18', 'resnet34'])
    assert set(json.loads((tmp_path / 'manifest.json').read_text())['files']) == {assets.ASSET_FILES['resnet18']}


def test_preparation_reuses_completed_manifest_files_without_redownloading(tmp_path, monkeypatch):
    _official_fixture(monkeypatch, tmp_path)
    prepare.prepare(tmp_path, ['resnet18'], register_existing=True)
    original_entry = json.loads((tmp_path / 'manifest.json').read_text())['files'][assets.ASSET_FILES['resnet18']]
    calls = []
    def download(url, cache):
        calls.append(url)
        path = Path(cache) / Path(url).name
        path.write_bytes(b'newly downloaded file')
        return path
    monkeypatch.setattr(prepare, 'cached_imagenet_weights', download)
    prepare.prepare(tmp_path, ['resnet18', 'resnet34'])
    assert len(calls) == 1 and 'resnet34-' in calls[0]
    assert json.loads((tmp_path / 'manifest.json').read_text())['files'][assets.ASSET_FILES['resnet18']] == original_entry


def test_package_readiness_still_requires_manifest(tmp_path, monkeypatch):
    path, digest = _official_fixture(monkeypatch, tmp_path)
    relative = path.relative_to(tmp_path).as_posix()
    monkeypatch.setattr(assets, 'ASSET_FILES', {'resnet18': relative})
    assert assets.builtin_asset_path('resnet18') == path
    assert not assets.builtin_assets_ready()
    (tmp_path / 'manifest.json').write_text(json.dumps({'schema_version': 1, 'files': {
        relative: {'sha256': digest, 'bytes': path.stat().st_size}}}))
    assert assets.builtin_assets_ready()


def test_prepare_cli_reports_failure_without_success(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(prepare, 'cached_imagenet_weights', lambda *args: (_ for _ in ()).throw(OSError('TLS error')))
    assert prepare.main(['--output', str(tmp_path), '--only', 'resnet18']) == 1
    out = capsys.readouterr()
    assert 'resnet18' in out.err and 'TLS error' in out.err
    assert '[ready]' not in out.out


def test_resnet_batch_can_run_from_another_working_directory():
    script = (Path(__file__).resolve().parents[1] / 'gui/prepare_resnet_weights.bat').read_text()
    assert 'cd /d "%~dp0"' in script
    assert '--output builtin_assets --only resnet18,resnet34,resnet50' in script
    assert '"%~1"=="offline"' in script and '--register-existing' in script
    assert 'if errorlevel 1' in script and 'exit /b 1' in script
    assert 'build_exe.py' not in script and 'powershell' not in script.lower()
