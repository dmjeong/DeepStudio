"""Validate the built-in model catalog against a staged Windows payload.

The catalog is intentionally separate from model weights.  A normal contract
build may ship the catalog while entries are still ``requested`` during
development.  A production build opts into the stricter release-ready mode,
which requires every catalog entry to declare the staged implementation or
model-pack files that make the entry usable offline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Mapping
import zipfile

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model_runtime.pack_signing import load_trusted_keys, verify_pack_signature


_STATUSES = frozenset({
    "requested", "scoped", "runtime_verified", "trained_verified",
    "export_verified", "sdk_verified", "release_ready",
})
_PAYLOAD_KINDS = frozenset({"builtin", "pack"})
_MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,79}$")
REQUIRED_BUILTIN_MODEL_IDS = frozenset({
    "efficientnet_b0", "efficientnet_b1", "resnet18", "resnet50",
    "convnext_v1_tiny", "libreyolo_classify_mobilenetv4_small",
    "patchcore_wide_resnet50_2", "patchcore_resnet18",
    "re_detr_v4_small", "re_detr_v4_medium", "re_detr_v4_large",
    "libreyolo_detect_9t", "deeplabv3plus_resnet34", "unet_resnet18",
    "sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_base_plus",
    "sam2_hiera_large",
})


class ModelCatalogPayloadError(ValueError):
    """The catalog or its staged model payload contract is invalid."""


def _safe_relative(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelCatalogPayloadError(f"{field} must be a non-empty relative path")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (":" in normalized or path.is_absolute() or "//" in normalized or
            any(part in {"", ".", ".."} for part in path.parts)):
        raise ModelCatalogPayloadError(f"{field} is unsafe: {value}")
    return "/".join(path.parts)


def _catalog_path(root: Path, value: str | Path) -> Path:
    relative = _safe_relative(str(value), "catalog path")
    path = root / Path(*PurePosixPath(relative).parts)
    if path.is_symlink() or not path.is_file():
        raise ModelCatalogPayloadError(f"model catalog is missing: {relative}")
    return path


def _read_catalog(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelCatalogPayloadError("model catalog is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ModelCatalogPayloadError("model catalog must be a JSON object")
    return value


def _validate_entry_payload(root: Path, model: Mapping[str, Any], model_id: str,
                            trusted_keys: Mapping[str, bytes | str]) -> None:
    metadata = model.get("metadata")
    payload = model.get("payload")
    if payload is None and isinstance(metadata, Mapping):
        payload = metadata.get("payload")
    if not isinstance(payload, Mapping):
        raise ModelCatalogPayloadError(
            f"release-ready model {model_id} must declare payload.kind and payload.paths"
        )
    kind = payload.get("kind")
    if kind not in _PAYLOAD_KINDS:
        raise ModelCatalogPayloadError(f"release-ready model {model_id} has unsupported payload.kind")
    paths = payload.get("paths")
    if (not isinstance(paths, list) or not paths or
            any(not isinstance(item, str) for item in paths)):
        raise ModelCatalogPayloadError(f"release-ready model {model_id} must declare payload.paths")
    normalized: list[str] = []
    for index, value in enumerate(paths):
        normalized.append(_safe_relative(value, f"model {model_id} payload.paths[{index}]"))
    if len(set(normalized)) != len(normalized):
        raise ModelCatalogPayloadError(f"release-ready model {model_id} has duplicate payload paths")
    for relative in normalized:
        path = root / Path(*PurePosixPath(relative).parts)
        if path.is_symlink() or not path.is_file():
            raise ModelCatalogPayloadError(
                f"release-ready model {model_id} payload file is missing: {relative}"
            )
    if kind == "pack":
        pack_paths = [path for path in normalized if path.casefold().endswith(".dvmodel")]
        if not pack_paths:
            raise ModelCatalogPayloadError(
                f"release-ready model {model_id} pack payload must include a .dvmodel file"
            )
        _validate_model_pack(root / Path(*PurePosixPath(pack_paths[0]).parts), model_id,
                             trusted_keys)


def _validate_model_pack(path: Path, model_id: str,
                         trusted_keys: Mapping[str, bytes | str]) -> None:
    """Check release metadata without extracting or executing a model pack."""
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ModelCatalogPayloadError(f"release-ready model {model_id} pack is not a ZIP archive") from exc
    with archive:
        infos = archive.infolist()
        names: dict[str, zipfile.ZipInfo] = {}
        for info in infos:
            if info.is_dir():
                continue
            try:
                normalized = _safe_relative(info.filename, "model pack path")
            except ModelCatalogPayloadError:
                raise
            mode = (info.external_attr >> 16) & 0xFFFF
            if mode and (mode & 0o170000) == 0o120000:
                raise ModelCatalogPayloadError(f"release-ready model {model_id} pack contains a symlink")
            if info.filename != normalized:
                raise ModelCatalogPayloadError(f"release-ready model {model_id} pack path is not normalized")
            folded = normalized.casefold()
            if any(name.casefold() == folded for name in names):
                raise ModelCatalogPayloadError(f"release-ready model {model_id} pack contains duplicate paths")
            names[normalized] = info
        name_set = set(names)
        required = {"manifest.json", "checksums.json", "THIRD_PARTY_NOTICES.md"}
        if not required.issubset(name_set) or not any(name.startswith("licenses/") for name in name_set):
            raise ModelCatalogPayloadError(
                f"release-ready model {model_id} pack must include manifest, checksums, notices and licenses"
            )
        manifest_info = names["manifest.json"]
        if manifest_info.file_size > 1024 * 1024:
            raise ModelCatalogPayloadError(f"release-ready model {model_id} pack manifest is too large")
        try:
            manifest = json.loads(archive.read("manifest.json"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
            raise ModelCatalogPayloadError(f"release-ready model {model_id} pack manifest is invalid") from exc
        pack_version = manifest.get("pack_version") if isinstance(manifest, Mapping) else None
        if (not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1 or
                manifest.get("release_status") != "release_ready" or
                manifest.get("model_id") != model_id or
                not isinstance(pack_version, str) or not pack_version.strip()):
            raise ModelCatalogPayloadError(
                f"release-ready model {model_id} pack manifest must use schema 1, match model_id, "
                "declare a pack_version and be release_ready"
            )
        license_info = manifest.get("license")
        if not isinstance(license_info, Mapping):
            raise ModelCatalogPayloadError(f"release-ready model {model_id} pack requires a license object")
        for field in ("spdx", "source", "revision"):
            value = license_info.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ModelCatalogPayloadError(
                    f"release-ready model {model_id} pack license.{field} is required"
                )
        try:
            checksum_document = json.loads(archive.read("checksums.json"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as exc:
            raise ModelCatalogPayloadError(
                f"release-ready model {model_id} pack checksums are invalid"
            ) from exc
        checksums = checksum_document.get("files") if isinstance(checksum_document, Mapping) else None
        if not isinstance(checksums, Mapping):
            raise ModelCatalogPayloadError(
                f"release-ready model {model_id} pack checksums.files is required"
            )
        expected_names = name_set - {"checksums.json"}
        if set(checksums) != expected_names:
            raise ModelCatalogPayloadError(
                f"release-ready model {model_id} pack checksums must cover every file"
            )
        for name, expected in checksums.items():
            if (not isinstance(expected, str) or len(expected) != 64 or
                    any(character not in "0123456789abcdefABCDEF" for character in expected)):
                raise ModelCatalogPayloadError(
                    f"release-ready model {model_id} pack has an invalid checksum: {name}"
                )
            actual = hashlib.sha256(archive.read(name)).hexdigest()
            if actual.lower() != expected.lower():
                raise ModelCatalogPayloadError(
                    f"release-ready model {model_id} pack checksum mismatch: {name}"
                )
        try:
            verify_pack_signature(manifest, checksums, trusted_keys)
        except ValueError as exc:
            raise ModelCatalogPayloadError(
                f"release-ready model {model_id} pack signature failed: {exc}"
            ) from exc


def validate_model_catalog_payload(
    root: str | Path,
    catalog: str | Path = "models/default-model-catalog.json",
    *,
    require_release_ready: bool = False,
    required_model_ids: set[str] | frozenset[str] | None = None,
    trusted_keys: Mapping[str, bytes | str] | None = None,
    trust_store: str | Path | None = None,
) -> Mapping[str, Any]:
    """Validate a staged catalog and optionally its production release gate.

    The returned mapping is the parsed catalog.  No model code, weights, or
    network resource is loaded while validating it.
    """
    if trusted_keys is not None and trust_store is not None:
        raise ModelCatalogPayloadError("provide trusted_keys or trust_store, not both")
    try:
        resolved_keys = load_trusted_keys(trust_store) if trust_store is not None else dict(trusted_keys or {})
    except ValueError as exc:
        raise ModelCatalogPayloadError(str(exc)) from exc
    root_path = Path(root).expanduser()
    if root_path.is_symlink() or not root_path.is_dir():
        raise ModelCatalogPayloadError("model payload root must be a real directory")
    base = root_path.resolve()
    catalog_path = _catalog_path(base, catalog)
    value = _read_catalog(catalog_path)
    if value.get("schema_version") != 1 or value.get("offline") is not True:
        raise ModelCatalogPayloadError("model catalog must be an offline schema_version 1 catalog")
    policy = value.get("redistribution_policy")
    if (not isinstance(policy, Mapping) or policy.get("weights_included") is not True or
            policy.get("require_third_party_notices") is not True or
            policy.get("require_license_files_for_release_packs") is not True):
        raise ModelCatalogPayloadError("model catalog redistribution policy is incomplete")
    models = value.get("models")
    if not isinstance(models, list) or not models:
        raise ModelCatalogPayloadError("model catalog must contain at least one model")
    seen: set[str] = set()
    for model in models:
        if not isinstance(model, Mapping):
            raise ModelCatalogPayloadError("model catalog entry must be an object")
        model_id = model.get("model_id")
        if (not isinstance(model_id, str) or not _MODEL_ID_RE.fullmatch(model_id) or
                model_id in seen):
            raise ModelCatalogPayloadError(f"model catalog has an invalid or duplicate model_id: {model_id!r}")
        seen.add(model_id)
        status = model.get("release_status", "requested")
        if status not in _STATUSES:
            raise ModelCatalogPayloadError(f"model {model_id} has an unsupported release_status")
        if require_release_ready:
            if status != "release_ready":
                raise ModelCatalogPayloadError(
                    f"model {model_id} is not release_ready (status={status})"
                )
            _validate_entry_payload(base, model, model_id, resolved_keys)
    if require_release_ready and value.get("release_ready_only") is not True:
        raise ModelCatalogPayloadError(
            "production model catalog must be generated with release_ready_only=true"
        )
    if require_release_ready:
        required = REQUIRED_BUILTIN_MODEL_IDS if required_model_ids is None else frozenset(required_model_ids)
        missing = sorted(required - seen)
        unexpected = sorted(seen - required)
        if missing or unexpected:
            raise ModelCatalogPayloadError(
                f"production model catalog set mismatch: missing={missing}, unexpected={unexpected}"
            )
    return value


__all__ = ["ModelCatalogPayloadError", "REQUIRED_BUILTIN_MODEL_IDS",
           "validate_model_catalog_payload"]
