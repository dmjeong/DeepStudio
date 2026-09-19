"""Deterministic offline ``.dvmodel`` pack builder.

The builder only archives files and writes their SHA-256 manifest.  It never
executes model code or resolves dependencies.  Release signing remains an
external step; unsigned output is deliberately opt-in for local development.
"""

from __future__ import annotations

import json
from pathlib import Path
import os
import tempfile
from typing import Any, Mapping
import zipfile

from .pack_installer import (MODEL_ID_RE, PACK_VERSION_RE, WINDOWS_RESERVED_NAMES,
                              _safe_name, _sha256, _validate_release_metadata,
                              _validate_release_notices)
from .special_contracts import (SpecialContractError, validate_container_image_asset,
                                validate_special_assets, validate_special_manifest)
from .pack_signing import verify_pack_signature


class PackBuildError(ValueError):
    """Raised when a source tree cannot become a safe model pack."""


def _required_text(manifest: Mapping[str, Any], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise PackBuildError(f"manifest {key} must be a safe non-empty string")
    value = value.strip()
    pattern = MODEL_ID_RE if key == "model_id" else PACK_VERSION_RE if key == "pack_version" else None
    if pattern is None:
        if "/" in value or "\\" in value or ".." in value or ":" in value:
            raise PackBuildError(f"manifest {key} must be a safe non-empty string")
    elif (not pattern.fullmatch(value) or
          value.casefold() in WINDOWS_RESERVED_NAMES):
        raise PackBuildError(f"manifest {key} must be a safe non-empty string")
    return value


def build_pack(source: str | Path, output: str | Path, *, manifest: str | Path | None = None,
               allow_unsigned: bool = False,
               trusted_keys: Mapping[str, bytes | str] | None = None) -> Path:
    """Create a reproducible pack from a directory and return its output path."""
    source_path = Path(source).expanduser()
    if source_path.is_symlink():
        raise PackBuildError("pack source symlink is not allowed")
    root = source_path.resolve()
    output_path = Path(output).expanduser()
    if output_path.exists() and output_path.is_symlink():
        raise PackBuildError("pack output symlink is not allowed")
    target = output_path.resolve()
    if not root.is_dir():
        raise PackBuildError(f"pack source is not a directory: {root}")
    if target.suffix.lower() != ".dvmodel":
        raise PackBuildError("pack output must use the .dvmodel suffix")
    manifest_input = Path(manifest).expanduser() if manifest else None
    if manifest_input is not None and manifest_input.is_symlink():
        raise PackBuildError("manifest symlink is not allowed")
    manifest_path = manifest_input.resolve() if manifest_input is not None else root / "manifest.json"
    if not manifest_path.is_file():
        raise PackBuildError("pack source requires manifest.json")
    if manifest_path.is_symlink():
        raise PackBuildError("manifest symlink is not allowed")
    try:
        definition = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackBuildError("invalid manifest.json") from exc
    if not isinstance(definition, Mapping) or definition.get("schema_version") != 1:
        raise PackBuildError("manifest schema_version must be 1")
    try:
        validate_special_manifest(definition)
    except SpecialContractError as exc:
        raise PackBuildError(str(exc)) from exc
    _required_text(definition, "model_id")
    _required_text(definition, "pack_version")
    signature = definition.get("signature")
    signature_present = signature is not None
    if not signature_present and not allow_unsigned:
        raise PackBuildError("unsigned output requires --allow-unsigned")

    files: list[tuple[str, Path]] = []
    names: set[str] = set()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PackBuildError(f"symlink is not allowed: {path.relative_to(root)}")
        if not path.is_file() or path.resolve() == target:
            continue
        name = _safe_name(path.relative_to(root).as_posix())
        if name == "checksums.json":
            continue
        # An explicitly supplied manifest replaces a stale source/manifest.json.
        if name == "manifest.json" and manifest_path != path.resolve():
            continue
        folded = name.casefold()
        if folded in names:
            raise PackBuildError(f"duplicate pack path: {name}")
        names.add(folded)
        files.append((name, path))
    if manifest_path != (root / "manifest.json").resolve():
        files.append(("manifest.json", manifest_path))
        names.add("manifest.json")
    if "manifest.json" not in {name for name, _ in files}:
        raise PackBuildError("pack source requires manifest.json")
    try:
        validate_special_assets(definition, {name for name, _ in files})
        validate_container_image_asset(definition, {name for name, _ in files})
    except SpecialContractError as exc:
        raise PackBuildError(str(exc)) from exc
    try:
        _validate_release_notices(definition, {name for name, _ in files})
        _validate_release_metadata(definition)
    except ValueError as exc:
        raise PackBuildError(str(exc)) from exc

    checksums = {name: _sha256(path) for name, path in files}
    if signature_present:
        try:
            verify_pack_signature(definition, checksums, dict(trusted_keys or {}))
        except ValueError as exc:
            raise PackBuildError(str(exc)) from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, path in files:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
            info = zipfile.ZipInfo("checksums.json", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, json.dumps({"schema_version": 1, "files": checksums},
                                               ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target
