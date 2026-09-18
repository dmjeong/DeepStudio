"""Offline asset resolver contracts."""

from pathlib import Path
import hashlib

import pytest

from model_runtime.assets import AssetResolutionError, AssetResolver


def test_resolver_prefers_roots_and_verifies_manifest_hash(tmp_path: Path):
    project, pack, cache = (tmp_path / name for name in ("project", "pack", "cache"))
    for root in (project, pack, cache):
        root.mkdir()
    (pack / "model.onnx").write_bytes(b"graph")
    digest = hashlib.sha256(b"graph").hexdigest()
    resolver = AssetResolver((project, pack, cache))
    asset = resolver.resolve_manifest({"assets": [{"path": "model.onnx", "size": 5, "sha256": digest}]})[0]
    assert asset.path == (pack / "model.onnx").resolve()
    assert asset.source == "root-1"


def test_resolver_rejects_download_like_paths_and_hash_mismatch(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "model.onnx").write_bytes(b"graph")
    resolver = AssetResolver((root,))
    with pytest.raises(AssetResolutionError, match="relative"):
        resolver.resolve("../model.onnx")
    with pytest.raises(AssetResolutionError, match="hash mismatch"):
        resolver.resolve("model.onnx", sha256="0" * 64)


def test_resolver_rejects_symlink_assets(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"private")
    (root / "model.onnx").symlink_to(outside)
    resolver = AssetResolver((root,))
    with pytest.raises(AssetResolutionError, match="symlink"):
        resolver.resolve("model.onnx")
