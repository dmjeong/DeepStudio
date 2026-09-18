"""Offline `.dvmodel` staging, checksum validation, and atomic activation.

The installer never imports or executes files from a model pack. A pack may
contain a container image and arbitrary model assets, but activation only
publishes a verified directory and a small pointer file for the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Mapping
import uuid
import zipfile


class PackInstallError(ValueError):
    """Raised when a model pack is unsafe, corrupt, or incompatible."""


MAX_ENTRIES = 4096
MAX_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
SHA256_HEX = 64


def _safe_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name:
        raise PackInstallError("model pack contains an invalid path")
    normalized = name.replace("\\", "/")
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
                 max_uncompressed_bytes: int = MAX_UNCOMPRESSED_BYTES):
        self.root = Path(root).expanduser()
        self.max_entries = max_entries
        self.max_uncompressed_bytes = max_uncompressed_bytes
        if self.max_entries < 1 or self.max_uncompressed_bytes < 1:
            raise ValueError("pack limits must be positive")

    def install(self, pack_path: str | Path, *, allow_unsigned: bool = False) -> InstalledPack:
        source = Path(pack_path).expanduser()
        if source.suffix.lower() != ".dvmodel" or not source.is_file():
            raise PackInstallError("model pack must be an existing .dvmodel file")
        staging_parent = self.root.parent if self.root.parent.exists() else Path(tempfile.gettempdir())
        staging_parent.mkdir(parents=True, exist_ok=True)
        staging: Path | None = Path(tempfile.mkdtemp(prefix=".dvmodel-", dir=staging_parent))
        try:
            manifest, checksums, signed = self._extract_verified(source, staging, allow_unsigned=allow_unsigned)
            model_id = self._required_text(manifest, "model_id")
            version = self._required_text(manifest, "pack_version")
            content_hash = _sha256(source)
            target = self.root / model_id / version
            if target.exists():
                existing = target / "pack.sha256"
                if existing.is_file() and existing.read_text(encoding="ascii").strip() == content_hash:
                    return InstalledPack(model_id, version, content_hash, target, signed)
                raise PackInstallError(f"pack version already installed with different content: {model_id}/{version}")
            target.parent.mkdir(parents=True, exist_ok=True)
            staged_target = target.parent / f".{version}.{uuid.uuid4().hex}.staging"
            shutil.move(str(staging), str(staged_target))
            staging = None
            (staged_target / "pack.sha256").write_text(content_hash + "\n", encoding="ascii")
            (staged_target / "installed.json").write_text(json.dumps({
                "model_id": model_id, "pack_version": version,
                "content_hash": content_hash, "signed": signed,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            staged_target.replace(target)
            pointer = target.parent / "current.json"
            temporary_pointer = target.parent / f".{pointer.name}.{uuid.uuid4().hex}.tmp"
            temporary_pointer.write_text(json.dumps({
                "model_id": model_id, "pack_version": version,
                "content_hash": content_hash, "path": str(target),
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary_pointer.replace(pointer)
            return InstalledPack(model_id, version, content_hash, target, signed)
        except Exception:
            if staging is not None and staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
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
            if not isinstance(checksums.get("files"), Mapping):
                raise PackInstallError("checksums.json.files must be an object")
            signature = manifest.get("signature")
            signed = isinstance(signature, Mapping) and bool(signature.get("key_id")) and bool(signature.get("value"))
            if not signed and not allow_unsigned:
                raise PackInstallError("unsigned development pack rejected")
            expected_files = {_safe_name(info.filename) for info in infos
                              if not info.is_dir() and _safe_name(info.filename) != "checksums.json"}
            listed_files = {_safe_name(name) for name in checksums["files"]}
            if listed_files != expected_files:
                raise PackInstallError("checksums.json does not cover exactly the pack files")
            for name, expected in checksums["files"].items():
                if not isinstance(expected, str) or len(expected) != SHA256_HEX or any(c not in "0123456789abcdefABCDEF" for c in expected):
                    raise PackInstallError(f"invalid checksum for {name}")
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
                if _sha256(target).lower() != checksums["files"][name].lower():
                    raise PackInstallError(f"checksum mismatch: {name}")
            return manifest, checksums, signed

    @staticmethod
    def _required_text(manifest: Mapping[str, Any], key: str) -> str:
        value = manifest.get(key)
        if not isinstance(value, str) or not value.strip() or "/" in value or "\\" in value or ".." in value:
            raise PackInstallError(f"manifest {key} must be a safe non-empty string")
        return value.strip()
