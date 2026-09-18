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


def test_signed_pack_is_idempotent(tmp_path):
    pack = tmp_path / "model.dvmodel"
    _write_pack(pack, signed=True)
    root = tmp_path / "installed"
    first = PackInstaller(root).install(pack)
    second = PackInstaller(root).install(pack)
    assert first == second
