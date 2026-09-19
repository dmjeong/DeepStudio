"""Offline `.dvmodel` staging, checksum validation, and atomic activation.

The installer never imports or executes files from a model pack. A pack may
contain a container image and arbitrary model assets, but activation only
publishes a verified directory and a small pointer file for the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Mapping
import uuid
import zipfile

from .special_contracts import (SpecialContractError, validate_container_image_asset,
                                validate_special_assets, validate_special_manifest)
from .pack_signing import load_trusted_keys, verify_pack_signature


class PackInstallError(ValueError):
    """Raised when a model pack is unsafe, corrupt, or incompatible."""


MAX_ENTRIES = 4096
MAX_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
SHA256_HEX = 64
MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,79}$")
PACK_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
WINDOWS_RESERVED_NAMES = frozenset({
    "con", "prn", "aux", "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
})


def _validate_release_notices(manifest: Mapping[str, Any], files) -> None:
    """Require redistribution notices before a pack can claim release-ready."""
    if manifest.get("release_status", "requested") != "release_ready":
        return
    names = set(files)
    if "THIRD_PARTY_NOTICES.md" not in names:
        raise PackInstallError("release-ready model pack requires THIRD_PARTY_NOTICES.md")
    if not any(name.startswith("licenses/") and name != "licenses/" for name in names):
        raise PackInstallError("release-ready model pack requires licenses/")


def _validate_release_metadata(manifest: Mapping[str, Any]) -> None:
    """Require auditable provenance fields before accepting a release pack."""
    if manifest.get("release_status", "requested") != "release_ready":
        return
    license_info = manifest.get("license")
    if not isinstance(license_info, Mapping):
        raise PackInstallError("release-ready model pack requires a license object")
    for field in ("spdx", "source", "revision"):
        value = license_info.get(field)
        if not isinstance(value, str) or not value.strip():
            raise PackInstallError(f"release-ready model pack license.{field} is required")


def _safe_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name:
        raise PackInstallError("model pack contains an invalid path")
    normalized = name.replace("\\", "/")
    if ":" in normalized:
        raise PackInstallError(f"unsafe model pack path: {name}")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or any(part == "" for part in path.parts):
        raise PackInstallError(f"unsafe model pack path: {name}")
    return "/".join(path.parts)


def _read_json(archive: zipfile.ZipFile, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(archive.read(name))
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackInstallError(f"invalid {name}") from exc
    if not isinstance(value, Mapping):
        raise PackInstallError(f"{name} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class InstalledPack:
    model_id: str
    pack_version: str
    content_hash: str
    path: Path
    signed: bool


class PackInstaller:
    """Install a verified pack below an app-owned local directory."""

    def __init__(self, root: str | Path, *, max_entries: int = MAX_ENTRIES,
                 max_uncompressed_bytes: int = MAX_UNCOMPRESSED_BYTES,
                 trusted_keys: Mapping[str, bytes | str] | None = None,
                 trust_store: str | Path | None = None):
        self.root = Path(root).expanduser()
        self.max_entries = max_entries
        self.max_uncompressed_bytes = max_uncompressed_bytes
        configured_store = trust_store or os.environ.get("DEEPVISION_MODEL_PACK_TRUST_STORE")
        if trusted_keys is not None and configured_store is not None:
            raise PackInstallError("provide trusted_keys or a model pack trust store, not both")
        try:
            self.trusted_keys = (load_trusted_keys(configured_store) if configured_store is not None
                                 else dict(trusted_keys or {}))
        except ValueError as exc:
            raise PackInstallError(str(exc)) from exc
        if self.max_entries < 1 or self.max_uncompressed_bytes < 1:
            raise ValueError("pack limits must be positive")

    def install(self, pack_path: str | Path, *, allow_unsigned: bool = False) -> InstalledPack:
        if not self.root.is_absolute():
            raise PackInstallError("installed model root must be an absolute directory")
        if self.root.exists() and self.root.is_symlink():
            raise PackInstallError("installed model root cannot be a symlink")
        self.root = self.root.resolve()
        if self.root.exists() and not self.root.is_dir():
            raise PackInstallError("installed model root must be a directory")
        source = Path(pack_path).expanduser()
        if source.suffix.lower() != ".dvmodel" or not source.is_file():
            raise PackInstallError("model pack must be an existing .dvmodel file")
        staging_parent = self.root.parent if self.root.parent.exists() else Path(tempfile.gettempdir())
        staging_parent.mkdir(parents=True, exist_ok=True)
        staging: Path | None = Path(tempfile.mkdtemp(prefix=".dvmodel-", dir=staging_parent))
        staged_target: Path | None = None
        temporary_pointer: Path | None = None
        target_committed = False
        pointer_committed = False
        try:
            manifest, checksums, signed = self._extract_verified(source, staging, allow_unsigned=allow_unsigned)
            model_id = self._required_text(manifest, "model_id")
            version = self._required_text(manifest, "pack_version")
            content_hash = _sha256(source)
            model_root = self.root / model_id
            if model_root.is_symlink():
                raise PackInstallError("model install directory cannot be a symlink")
            if model_root.exists() and not model_root.is_dir():
                raise PackInstallError("model install directory must be a directory")
            model_root.mkdir(parents=True, exist_ok=True)
            # Re-check after mkdir so a pre-existing or raced path cannot make
            # the subsequent move escape the application-owned root.
            if model_root.is_symlink() or not model_root.is_dir():
                raise PackInstallError("model install directory is unsafe")
            if model_root.resolve().parent != self.root:
                raise PackInstallError("model install directory escapes installed root")
            target = model_root / version
            if target.is_symlink():
                raise PackInstallError("model version directory cannot be a symlink")
            if target.exists():
                if not target.is_dir():
                    raise PackInstallError("model version path must be a directory")
                existing = target / "pack.sha256"
                if existing.is_file() and existing.read_text(encoding="ascii").strip() == content_hash:
                    return InstalledPack(model_id, version, content_hash, target, signed)
                raise PackInstallError(f"pack version already installed with different content: {model_id}/{version}")
            staged_target = target.parent / f".{version}.{uuid.uuid4().hex}.staging"
            shutil.move(str(staging), str(staged_target))
            staging = None
            (staged_target / "pack.sha256").write_text(content_hash + "\n", encoding="ascii")
            (staged_target / "installed.json").write_text(json.dumps({
                "model_id": model_id, "pack_version": version,
                "content_hash": content_hash, "signed": signed,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            staged_target.replace(target)
            target_committed = True
            staged_target = None
            pointer = target.parent / "current.json"
            temporary_pointer = target.parent / f".{pointer.name}.{uuid.uuid4().hex}.tmp"
            temporary_pointer.write_text(json.dumps({
                "model_id": model_id, "pack_version": version,
                "content_hash": content_hash, "path": str(target),
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary_pointer.replace(pointer)
            pointer_committed = True
            temporary_pointer = None
            return InstalledPack(model_id, version, content_hash, target, signed)
        except Exception:
            if staging is not None and staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            if staged_target is not None and staged_target.exists() and not staged_target.is_symlink():
                shutil.rmtree(staged_target, ignore_errors=True)
            if temporary_pointer is not None and temporary_pointer.exists() and not temporary_pointer.is_symlink():
                temporary_pointer.unlink(missing_ok=True)
            if target_committed and not pointer_committed and target.exists() and not target.is_symlink():
                shutil.rmtree(target, ignore_errors=True)
            raise

    def _extract_verified(self, source: Path, staging: Path, *, allow_unsigned: bool):
        try:
            archive = zipfile.ZipFile(source)
        except (OSError, zipfile.BadZipFile) as exc:
            raise PackInstallError("cannot open model pack") from exc
        with archive:
            infos = archive.infolist()
            if not infos or len(infos) > self.max_entries:
                raise PackInstallError("model pack entry limit exceeded")
            names: dict[str, zipfile.ZipInfo] = {}
            total = 0
            for info in infos:
                name = _safe_name(info.filename)
                folded = name.casefold()
                if folded in names:
                    raise PackInstallError(f"duplicate model pack path: {name}")
                names[folded] = info
                if info.is_dir():
                    continue
                mode = (info.external_attr >> 16) & 0xFFFF
                if mode and (mode & 0o170000) == 0o120000:
                    raise PackInstallError(f"symlink is not allowed in model pack: {name}")
                total += int(info.file_size)
                if total > self.max_uncompressed_bytes:
                    raise PackInstallError("model pack uncompressed size limit exceeded")
            if "manifest.json" not in names or "checksums.json" not in names:
                raise PackInstallError("model pack requires manifest.json and checksums.json")
            manifest = _read_json(archive, "manifest.json")
            checksums = _read_json(archive, "checksums.json")
            if manifest.get("schema_version") != 1:
                raise PackInstallError("unsupported model pack schema_version")
            try:
                validate_special_manifest(manifest)
            except SpecialContractError as exc:
                raise PackInstallError(str(exc)) from exc
            if not isinstance(checksums.get("files"), Mapping):
                raise PackInstallError("checksums.json.files must be an object")
            signature = manifest.get("signature")
            signature_present = signature is not None
            if not signature_present and not allow_unsigned:
                raise PackInstallError("unsigned development pack rejected")
            expected_files = {_safe_name(info.filename) for info in infos
                              if not info.is_dir() and _safe_name(info.filename) != "checksums.json"}
            _validate_release_notices(manifest, expected_files)
            _validate_release_metadata(manifest)
            try:
                validate_special_assets(manifest, expected_files)
                validate_container_image_asset(manifest, expected_files)
            except SpecialContractError as exc:
                raise PackInstallError(str(exc)) from exc
            checksum_values: dict[str, Any] = {}
            checksum_folds: set[str] = set()
            for raw_name, expected in checksums["files"].items():
                name = _safe_name(raw_name)
                if name != raw_name:
                    raise PackInstallError(f"checksums.json path is not normalized: {raw_name}")
                folded = name.casefold()
                if folded in checksum_folds:
                    raise PackInstallError(f"duplicate checksum path: {raw_name}")
                checksum_folds.add(folded)
                checksum_values[name] = expected
            listed_files = set(checksum_values)
            if listed_files != expected_files:
                raise PackInstallError("checksums.json does not cover exactly the pack files")
            for name, expected in checksum_values.items():
                if not isinstance(expected, str) or len(expected) != SHA256_HEX or any(c not in "0123456789abcdefABCDEF" for c in expected):
                    raise PackInstallError(f"invalid checksum for {name}")
            signed = False
            if signature_present:
                try:
                    verify_pack_signature(manifest, checksum_values, self.trusted_keys)
                except ValueError as exc:
                    raise PackInstallError(str(exc)) from exc
                signed = True
            for info in infos:
                name = _safe_name(info.filename)
                if info.is_dir():
                    continue
                target = staging / Path(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as input_stream, target.open("wb") as output_stream:
                    shutil.copyfileobj(input_stream, output_stream, 1024 * 1024)
                if name == "checksums.json":
                    continue
                if _sha256(target).lower() != checksum_values[name].lower():
                    raise PackInstallError(f"checksum mismatch: {name}")
            return manifest, checksums, signed

    @staticmethod
    def _required_text(manifest: Mapping[str, Any], key: str) -> str:
        value = manifest.get(key)
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise PackInstallError(f"manifest {key} must be a safe non-empty string")
        value = value.strip()
        pattern = MODEL_ID_RE if key == "model_id" else PACK_VERSION_RE if key == "pack_version" else None
        if pattern is None:
            if "/" in value or "\\" in value or ".." in value or ":" in value:
                raise PackInstallError(f"manifest {key} must be a safe non-empty string")
        elif (not pattern.fullmatch(value) or
              value.casefold() in WINDOWS_RESERVED_NAMES):
            raise PackInstallError(f"manifest {key} must be a safe non-empty string")
        return value
