"""Offline model pack extraction, checksums, and activation contracts."""

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from model_runtime.pack_installer import PackInstallError, PackInstaller


def _write_pack(path: Path, *, signed=False, corrupt=False):
    files = {
        "manifest.json": json.dumps({"schema_version": 1, "model_id": "vendor.example", "pack_version": "1.0.0"}).encode(),
        "README.ko.md": b"offline model",
        "assets/model.onnx": b"fixture",
    }
    checksums = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    if corrupt:
        checksums["assets/model.onnx"] = "0" * 64
    manifest = json.loads(files["manifest.json"])
    if signed:
        manifest["signature"] = {"key_id": "test", "value": "development"}
        files["manifest.json"] = json.dumps(manifest).encode()
        checksums["manifest.json"] = hashlib.sha256(files["manifest.json"]).hexdigest()
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
        archive.writestr("checksums.json", json.dumps({"files": checksums}))


def test_installer_requires_signature_unless_development_mode(tmp_path):
    pack = tmp_path / "model.dvmodel"
    _write_pack(pack)
    with pytest.raises(PackInstallError, match="unsigned"):
        PackInstaller(tmp_path / "installed").install(pack)
    installed = PackInstaller(tmp_path / "installed").install(pack, allow_unsigned=True)
    assert installed.model_id == "vendor.example"
    assert (installed.path / "assets/model.onnx").read_bytes() == b"fixture"
    assert json.loads((installed.path.parent / "current.json").read_text())["pack_version"] == "1.0.0"


def test_installer_rejects_checksum_tampering(tmp_path):
    pack = tmp_path / "model.dvmodel"
    _write_pack(pack, corrupt=True)
    with pytest.raises(PackInstallError, match="checksum mismatch"):
        PackInstaller(tmp_path / "installed").install(pack, allow_unsigned=True)


def test_installer_rejects_windows_drive_paths_inside_archive(tmp_path):
    pack = tmp_path / "drive-path.dvmodel"
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("C:/escape.txt", b"private")
    with pytest.raises(PackInstallError, match="unsafe|invalid"):
        PackInstaller(tmp_path / "installed").install(pack, allow_unsigned=True)


@pytest.mark.parametrize("field,value", [
    ("model_id", "vendor:escape"),
    ("model_id", "con"),
    ("pack_version", "../1.0.0"),
    ("pack_version", "1:0:0"),
])
def test_installer_rejects_windows_unsafe_manifest_identifiers(tmp_path, field, value):
    pack = tmp_path / "unsafe.dvmodel"
    manifest = {"schema_version": 1, "model_id": "vendor.example", "pack_version": "1.0.0"}
    manifest[field] = value
    files = {"manifest.json": json.dumps(manifest).encode(), "model.onnx": b"fixture"}
    checksums = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    with zipfile.ZipFile(pack, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        archive.writestr("checksums.json", json.dumps({"files": checksums}))
    with pytest.raises(PackInstallError, match="manifest"):
        PackInstaller(tmp_path / "installed").install(pack, allow_unsigned=True)


def test_installer_rejects_non_normalized_checksum_keys(tmp_path):
    pack = tmp_path / "checksum-path.dvmodel"
    files = {
        "manifest.json": json.dumps({
            "schema_version": 1, "model_id": "vendor.example", "pack_version": "1.0.0",
        }).encode(),
        "assets/model.onnx": b"fixture",
    }
    checksums = {"manifest.json": hashlib.sha256(files["manifest.json"]).hexdigest(),
                 "assets\\model.onnx": hashlib.sha256(files["assets/model.onnx"]).hexdigest()}
    with zipfile.ZipFile(pack, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        archive.writestr("checksums.json", json.dumps({"files": checksums}))
    with pytest.raises(PackInstallError, match="normalized"):
        PackInstaller(tmp_path / "installed").install(pack, allow_unsigned=True)


def test_signed_pack_is_idempotent(tmp_path):
    pack = tmp_path / "model.dvmodel"
    _write_pack(pack, signed=True)
    root = tmp_path / "installed"
    first = PackInstaller(root).install(pack)
    second = PackInstaller(root).install(pack)
    assert first == second


def test_installer_rejects_release_ready_pack_without_redistribution_notices(tmp_path):
    pack = tmp_path / "release.dvmodel"
    files = {
        "manifest.json": json.dumps({
            "schema_version": 1, "model_id": "vendor.release", "pack_version": "1.0.0",
            "release_status": "release_ready",
        }).encode(),
        "model.onnx": b"fixture",
    }
    checksums = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    with zipfile.ZipFile(pack, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
        archive.writestr("checksums.json", json.dumps({"files": checksums}))
    with pytest.raises(PackInstallError, match="THIRD_PARTY_NOTICES"):
        PackInstaller(tmp_path / "installed").install(pack, allow_unsigned=True)
