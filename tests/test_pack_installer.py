"""Offline model pack extraction, checksums, and activation contracts."""

import hashlib
import base64
import json
from pathlib import Path
import zipfile

import pytest

from model_runtime.pack_installer import PackInstallError, PackInstaller


def _write_pack(path: Path, *, private_key=None, corrupt=False):
    files = {
        "manifest.json": json.dumps({"schema_version": 1, "model_id": "vendor.example", "pack_version": "1.0.0"}).encode(),
        "README.ko.md": b"offline model",
        "assets/model.onnx": b"fixture",
    }
    manifest = json.loads(files["manifest.json"])
    content_hashes = {
        name: hashlib.sha256(value).hexdigest()
        for name, value in files.items() if name != "manifest.json"
    }
    if corrupt:
        content_hashes["assets/model.onnx"] = "0" * 64
    if private_key is not None:
        from model_runtime.pack_signing import signature_payload
        signature = private_key.sign(signature_payload(manifest, content_hashes))
        manifest["signature"] = {
            "algorithm": "ed25519", "key_id": "test",
            "value": base64.b64encode(signature).decode("ascii"),
        }
        files["manifest.json"] = json.dumps(manifest).encode()
    checksums = {"manifest.json": hashlib.sha256(files["manifest.json"]).hexdigest(),
                 **content_hashes}
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
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    pack = tmp_path / "model.dvmodel"
    _write_pack(pack, private_key=private_key)
    root = tmp_path / "installed"
    first = PackInstaller(root, trusted_keys={"test": public_key}).install(pack)
    second = PackInstaller(root, trusted_keys={"test": public_key}).install(pack)
    assert first == second
    assert first.signed is True


def test_installer_loads_versioned_trust_store_from_environment(tmp_path, monkeypatch):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    trust_store = tmp_path / "model-pack-trust.json"
    trust_store.write_text(json.dumps({
        "schema_version": 1,
        "keys": {"test": {
            "algorithm": "ed25519",
            "public_key": base64.b64encode(public_key).decode("ascii"),
        }},
    }), encoding="utf-8")
    pack = tmp_path / "model.dvmodel"
    _write_pack(pack, private_key=private_key)
    monkeypatch.setenv("DEEPVISION_MODEL_PACK_TRUST_STORE", str(trust_store))
    installed = PackInstaller(tmp_path / "installed").install(pack)
    assert installed.signed is True


@pytest.mark.parametrize("payload,error", [
    ({"schema_version": 2, "keys": {}}, "schema_version"),
    ({"schema_version": 1, "keys": {}}, "at least one"),
    ({"schema_version": 1, "keys": {"bad": {
        "algorithm": "rsa", "public_key": "ignored",
    }}}, "Ed25519"),
])
def test_installer_rejects_invalid_trust_store(tmp_path, payload, error):
    trust_store = tmp_path / "bad-trust.json"
    trust_store.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PackInstallError, match=error):
        PackInstaller(tmp_path / "installed", trust_store=trust_store)


def test_installer_rejects_fake_and_unknown_signatures(tmp_path):
    pack = tmp_path / "fake.dvmodel"
    manifest = {
        "schema_version": 1, "model_id": "vendor.example", "pack_version": "1.0.0",
        "signature": {"algorithm": "ed25519", "key_id": "not-a-real-key",
                      "value": base64.b64encode(b"x" * 64).decode("ascii")},
    }
    files = {"manifest.json": json.dumps(manifest).encode(), "model.onnx": b"fixture"}
    checksums = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    with zipfile.ZipFile(pack, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
        archive.writestr("checksums.json", json.dumps({"files": checksums}))
    with pytest.raises(PackInstallError, match="unknown key"):
        PackInstaller(tmp_path / "installed").install(pack)

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    public_key = Ed25519PrivateKey.generate().public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    with pytest.raises(PackInstallError, match="verification failed"):
        PackInstaller(tmp_path / "installed", trusted_keys={
            "not-a-real-key": public_key,
        }).install(pack)


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


def test_installer_rejects_release_ready_pack_without_license_metadata(tmp_path):
    pack = tmp_path / "release-metadata.dvmodel"
    files = {
        "manifest.json": json.dumps({
            "schema_version": 1, "model_id": "vendor.release", "pack_version": "1.0.0",
            "release_status": "release_ready",
        }).encode(),
        "model.onnx": b"fixture",
        "THIRD_PARTY_NOTICES.md": b"notice",
        "licenses/model.txt": b"license",
    }
    checksums = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    with zipfile.ZipFile(pack, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
        archive.writestr("checksums.json", json.dumps({"files": checksums}))
    with pytest.raises(PackInstallError, match="license object"):
        PackInstaller(tmp_path / "installed").install(pack, allow_unsigned=True)


def test_installer_rejects_symlinked_model_directory(tmp_path):
    pack = tmp_path / "model.dvmodel"
    _write_pack(pack)
    root = tmp_path / "installed"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / "vendor.example").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    with pytest.raises(PackInstallError, match="symlink"):
        PackInstaller(root).install(pack, allow_unsigned=True)
    assert not (outside / "1.0.0").exists()


def test_installer_rejects_symlinked_version_directory(tmp_path):
    pack = tmp_path / "model.dvmodel"
    _write_pack(pack)
    root = tmp_path / "installed"
    model_root = root / "vendor.example"
    model_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (model_root / "1.0.0").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    with pytest.raises(PackInstallError, match="symlink"):
        PackInstaller(root).install(pack, allow_unsigned=True)
    assert not (outside / "pack.sha256").exists()


def test_installer_rejects_file_as_install_root(tmp_path):
    pack = tmp_path / "model.dvmodel"
    _write_pack(pack)
    root = tmp_path / "installed"
    root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(PackInstallError, match="root must be a directory"):
        PackInstaller(root).install(pack, allow_unsigned=True)
