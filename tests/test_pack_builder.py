"""Pack builder output is safe to install offline."""

import json
from pathlib import Path

import pytest

from model_runtime.pack_builder import PackBuildError, build_pack
from model_runtime.pack_installer import PackInstallError, PackInstaller


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "model_id": "vendor.example", "pack_version": "1.0.0",
    }), encoding="utf-8")
    (source / "model.onnx").write_bytes(b"graph")
    return source


def test_builder_requires_explicit_unsigned_development_mode(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(PackBuildError, match="unsigned"):
        build_pack(source, tmp_path / "model.dvmodel")
    output = build_pack(source, tmp_path / "model.dvmodel", allow_unsigned=True)
    installed = PackInstaller(tmp_path / "installed").install(output, allow_unsigned=True)
    assert installed.model_id == "vendor.example"
    assert (installed.path / "model.onnx").read_bytes() == b"graph"


def test_builder_does_not_follow_symlinks(tmp_path):
    source = _source(tmp_path)
    (source / "outside").symlink_to(tmp_path / "outside.txt")
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")
    with pytest.raises(PackBuildError, match="symlink"):
        build_pack(source, tmp_path / "model.dvmodel", allow_unsigned=True)


def test_builder_rejects_symlinked_source_root(tmp_path):
    source = _source(tmp_path)
    link = tmp_path / "source-link"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(PackBuildError, match="source symlink"):
        build_pack(link, tmp_path / "model.dvmodel", allow_unsigned=True)


def test_builder_accepts_manifest_outside_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"graph")
    manifest = tmp_path / "release-manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "model_id": "vendor.external", "pack_version": "2.0.0",
    }), encoding="utf-8")
    output = build_pack(source, tmp_path / "model.dvmodel", manifest=manifest, allow_unsigned=True)
    installed = PackInstaller(tmp_path / "installed").install(output, allow_unsigned=True)
    assert json.loads((installed.path / "manifest.json").read_text())['pack_version'] == "2.0.0"


def test_special_model_pack_requires_its_graph_contract(tmp_path):
    source = _source(tmp_path)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({"family": "Re-DETR v4", "variant": "Small", "task": "detect", "runtimes": ["onnx"]})
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PackBuildError, match="contracts"):
        build_pack(source, tmp_path / "redetr.dvmodel", allow_unsigned=True)


def test_container_pack_requires_declared_offline_image_archive_asset(tmp_path):
    source = _source(tmp_path)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({
        "runtimes": ["container"],
        "container_image": "registry.invalid/example@sha256:" + "a" * 64,
        "container_image_archive": "docker/image.tar",
    })
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PackBuildError, match="archive"):
        build_pack(source, tmp_path / "container.dvmodel", allow_unsigned=True)
    (source / "docker").mkdir()
    (source / "docker/image.tar").write_bytes(b"image")
    assert build_pack(source, tmp_path / "container.dvmodel", allow_unsigned=True).is_file()


def test_release_ready_pack_requires_redistribution_notices(tmp_path):
    source = _source(tmp_path)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["release_status"] = "release_ready"
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PackBuildError, match="THIRD_PARTY_NOTICES"):
        build_pack(source, tmp_path / "release.dvmodel", allow_unsigned=True)
    (source / "THIRD_PARTY_NOTICES.md").write_text("notice", encoding="utf-8")
    (source / "licenses").mkdir()
    (source / "licenses" / "model.txt").write_text("license", encoding="utf-8")
    assert build_pack(source, tmp_path / "release.dvmodel", allow_unsigned=True).is_file()


@pytest.mark.parametrize("field,value", [
    ("model_id", "vendor:escape"),
    ("model_id", "con"),
    ("pack_version", "../1.0.0"),
    ("pack_version", "1:0:0"),
])
def test_builder_rejects_windows_unsafe_manifest_identifiers(tmp_path, field, value):
    source = _source(tmp_path)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest[field] = value
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PackBuildError, match="manifest"):
        build_pack(source, tmp_path / "unsafe.dvmodel", allow_unsigned=True)
