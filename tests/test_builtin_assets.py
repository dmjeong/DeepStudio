"""Offline basic-model asset bundle contracts."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest


def test_windows_same_size_and_timestamp_rewrite_is_rehashed(tmp_path, monkeypatch):
    import builtin_assets

    target = tmp_path / "weights.pt"
    target.write_bytes(b"official")
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": {
        target.name: {"sha256": hashlib.sha256(b"official").hexdigest(), "bytes": 8},
    }}), encoding="utf-8")
    # Reproduce the Windows metadata collision deterministically on every OS.
    frozen_stat, real_stat = target.stat(), Path.stat
    monkeypatch.setattr(Path, "stat", lambda path, *a, **kw: frozen_stat if path == target else real_stat(path, *a, **kw))
    monkeypatch.setattr(builtin_assets, "sys", types.SimpleNamespace(platform="win32"))
    builtin_assets._verify(target, tmp_path, target.name, target.name)
    target.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="무결성"):
        builtin_assets._verify(target, tmp_path, target.name, target.name)


def test_packaged_asset_requires_manifest_hash_and_never_downloads(tmp_path, monkeypatch):
    import builtin_assets

    root = tmp_path / "assets"
    relative = builtin_assets.ASSET_FILES["sam2_hiera_tiny"]
    target = root / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"official")
    (root / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": {
        relative: {"sha256": hashlib.sha256(b"official").hexdigest(), "bytes": 8},
    }}), encoding="utf-8")
    monkeypatch.setenv("DVS_BUILTIN_ASSETS_DIR", str(root))
    assert builtin_assets.builtin_asset_path("sam2_hiera_tiny") == target.resolve()
    target.write_bytes(b"tampered")
    try:
        builtin_assets.builtin_asset_path("sam2_hiera_tiny")
    except RuntimeError as exc:
        assert "무결성" in str(exc)
    else:
        raise AssertionError("tampered packaged weight was accepted")


def test_asset_manifest_covers_every_basic_pretrained_file():
    from builtin_assets import ASSET_FILES

    assert {"sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_base_plus", "sam2_hiera_large"} <= set(ASSET_FILES)
    assert {"efficientnet_b0", "efficientnet_b1", "resnet18", "resnet50", "convnext_v1_tiny"} <= set(ASSET_FILES)


def test_libreyolo_download_enables_windows_native_certificate_store(monkeypatch):
    import model_download

    calls = []
    fake_truststore = types.SimpleNamespace(inject_into_ssl=lambda: calls.append("injected"))
    monkeypatch.setitem(sys.modules, "truststore", fake_truststore)
    monkeypatch.setattr(model_download, "_REQUESTS_NATIVE_CA_ENABLED", False)
    model_download.enable_requests_native_ca()
    model_download.enable_requests_native_ca()
    assert calls == ["injected"]
    source = (Path(__file__).resolve().parents[1] / "python" / "prepare_builtin_assets.py").read_text(
        encoding="utf-8"
    )
    assert source.index("enable_requests_native_ca()") < source.index(
        "from libreyolo.utils.download import download_weights"
    )
