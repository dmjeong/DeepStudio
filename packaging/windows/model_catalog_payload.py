"""Validate the built-in model catalog against a staged Windows payload.

The catalog is intentionally separate from model weights.  A normal contract
build may ship the catalog while entries are still ``requested`` during
development.  A production build opts into the stricter release-ready mode,
which requires every catalog entry to declare the staged implementation or
model-pack files that make the entry usable offline.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


_STATUSES = frozenset({
    "requested", "scoped", "runtime_verified", "trained_verified",
    "export_verified", "sdk_verified", "release_ready",
})
_PAYLOAD_KINDS = frozenset({"builtin", "pack"})


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


def _validate_entry_payload(root: Path, model: Mapping[str, Any], model_id: str) -> None:
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
    if kind == "pack" and not any(path.casefold().endswith(".dvmodel") for path in normalized):
        raise ModelCatalogPayloadError(
            f"release-ready model {model_id} pack payload must include a .dvmodel file"
        )


def validate_model_catalog_payload(
    root: str | Path,
    catalog: str | Path = "models/default-model-catalog.json",
    *,
    require_release_ready: bool = False,
) -> Mapping[str, Any]:
    """Validate a staged catalog and optionally its production release gate.

    The returned mapping is the parsed catalog.  No model code, weights, or
    network resource is loaded while validating it.
    """
    root_path = Path(root).expanduser()
    if root_path.is_symlink() or not root_path.is_dir():
        raise ModelCatalogPayloadError("model payload root must be a real directory")
    base = root_path.resolve()
    catalog_path = _catalog_path(base, catalog)
    value = _read_catalog(catalog_path)
    if value.get("schema_version") != 1 or value.get("offline") is not True:
        raise ModelCatalogPayloadError("model catalog must be an offline schema_version 1 catalog")
    policy = value.get("redistribution_policy")
    if (not isinstance(policy, Mapping) or policy.get("weights_included") is not False or
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
        if not isinstance(model_id, str) or not model_id.strip() or model_id in seen:
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
            _validate_entry_payload(base, model, model_id)
    if require_release_ready and value.get("release_ready_only") is not True:
        raise ModelCatalogPayloadError(
            "production model catalog must be generated with release_ready_only=true"
        )
    return value


__all__ = ["ModelCatalogPayloadError", "validate_model_catalog_payload"]
