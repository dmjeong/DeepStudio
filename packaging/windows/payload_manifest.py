"""Build and verify an offline Windows payload manifest.

This is intentionally platform-neutral so CI can check the release payload
before a Windows-only WiX/PyInstaller build. It never downloads dependencies.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
from pathlib import PurePosixPath
import re
from typing import Any


MAX_SIGNED_PE_BYTES = 4 * 1024 * 1024 * 1024
DEFAULT_BUDGET_BYTES = int(3.5 * 1024 * 1024 * 1024)
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_payload(root: str | Path, *, version: str, commit: str = "",
                    budget_bytes: int = DEFAULT_BUDGET_BYTES) -> dict[str, Any]:
    root_path = Path(root)
    if root_path.is_symlink():
        raise ValueError("payload root symlink is not allowed")
    base = root_path.resolve()
    if not base.is_dir():
        raise ValueError(f"payload root is not a directory: {base}")
    if budget_bytes <= 0 or budget_bytes >= MAX_SIGNED_PE_BYTES:
        raise ValueError("payload budget must be positive and below the signed PE limit")
    files: list[dict[str, Any]] = []
    total = 0
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"payload symlink is not allowed: {path.relative_to(base)}")
        if not path.is_file():
            continue
        relative = path.relative_to(base).as_posix()
        size = path.stat().st_size
        total += size
        if total > budget_bytes:
            raise ValueError(f"payload exceeds budget: {total} > {budget_bytes}")
        files.append({"path": relative, "size": size, "sha256": sha256(path)})
    return {"schema_version": 1, "version": version, "commit": commit,
            "root": base.name, "file_count": len(files), "payload_bytes": total,
            "budget_bytes": budget_bytes, "files": files,
            "requirements": {"offline": True, "windows_arch": "x64",
                              "external_downloads": False}}


def verify_payload(root: str | Path, manifest: dict[str, Any], *, required_paths=()) -> None:
    root_path = Path(root)
    if root_path.is_symlink():
        raise ValueError("payload root symlink is not allowed")
    base = root_path.resolve()
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError("unsupported payload manifest")
    if manifest.get("payload_bytes", -1) > manifest.get("budget_bytes", 0):
        raise ValueError("manifest exceeds payload budget")
    seen: set[str] = set()
    total = 0
    for item in manifest["files"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("invalid payload file entry")
        relative = item["path"].replace("\\", "/")
        if (relative in seen or relative.startswith("/") or "//" in relative or
                any(part in {"", ".", ".."} for part in Path(relative).parts)):
            raise ValueError(f"unsafe or duplicate payload path: {relative}")
        seen.add(relative)
        path = base / Path(*Path(relative).parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"payload file missing: {relative}")
        actual_size = path.stat().st_size
        if actual_size != item.get("size") or sha256(path).lower() != str(item.get("sha256", "")).lower():
            raise ValueError(f"payload hash/size mismatch: {relative}")
        total += actual_size
    if total != manifest.get("payload_bytes") or len(seen) != manifest.get("file_count"):
        raise ValueError("payload manifest totals do not match files")
    actual_paths = {path.relative_to(base).as_posix() for path in base.rglob("*") if path.is_file()}
    if actual_paths != seen:
        raise ValueError("payload contains files missing from manifest")
    for required in required_paths:
        relative = str(required).replace("\\", "/").lstrip("/")
        if relative not in actual_paths:
            raise ValueError(f"required payload file is missing: {relative}")


def verify_offline_wsl_payload(root: str | Path, payload_manifest: dict[str, Any]) -> None:
    """Verify the signed WSL/Docker payload's own provenance inventory.

    The inventory is deliberately data-only.  It does not download or execute
    anything; it binds the WSL installer, owned distro archive, and notice
    files to the exact bytes already covered by the release payload manifest.
    """
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ValueError(f"payload root is not a directory: {base}")
    files = payload_manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("payload manifest files are required before WSL verification")
    payload_paths = {item.get("path") for item in files if isinstance(item, dict)}
    required = {
        "runtime/wsl/wsl-offline.msi",
        "runtime/wsl/owned-distro.tar",
        "runtime/wsl/licenses/manifest.json",
    }
    missing = sorted(required - payload_paths)
    if missing:
        raise ValueError("offline WSL payload is missing: " + ", ".join(missing))
    inventory_path = base / "runtime" / "wsl" / "licenses" / "manifest.json"
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("offline WSL license inventory is invalid") from exc
    if (not isinstance(inventory, dict) or inventory.get("schema_version") != 1 or
            inventory.get("platform") != "windows-x64" or
            not isinstance(inventory.get("components"), list)):
        raise ValueError("offline WSL license inventory schema is invalid")
    seen: set[str] = set()
    for component in inventory["components"]:
        if not isinstance(component, dict):
            raise ValueError("offline WSL component entry is invalid")
        component_id = component.get("id")
        artifact = str(component.get("artifact", "")).replace("\\", "/")
        digest = component.get("sha256")
        license_name = component.get("license")
        notice = str(component.get("notice", "")).replace("\\", "/")
        artifact_path = PurePosixPath(artifact)
        notice_path = PurePosixPath(notice)
        if (not isinstance(component_id, str) or not component_id.strip() or component_id in seen or
                not artifact or ":" in artifact or artifact_path.is_absolute() or
                any(part in {"", ".", ".."} for part in artifact_path.parts) or
                artifact not in payload_paths or not SHA256_RE.fullmatch(str(digest or "")) or
                not isinstance(license_name, str) or not license_name.strip() or
                not notice or ":" in notice or notice_path.is_absolute() or
                any(part in {"", ".", ".."} for part in notice_path.parts) or
                notice not in payload_paths):
            raise ValueError(f"offline WSL component inventory is invalid: {component_id!r}")
        seen.add(component_id)
        artifact_file = base / Path(*artifact_path.parts)
        notice_file = base / Path(*notice_path.parts)
        if (artifact_file.is_symlink() or not artifact_file.is_file() or
                notice_file.is_symlink() or not notice_file.is_file()):
            raise ValueError(f"offline WSL component files are missing: {component_id}")
        if sha256(artifact_file).lower() != str(digest).lower():
            raise ValueError(f"offline WSL artifact hash mismatch: {artifact}")
    required_components = {"wsl", "owned_distro", "docker_engine"}
    if not required_components.issubset(seen):
        raise ValueError("offline WSL inventory must list wsl, owned_distro, and docker_engine")


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Build or verify a Windows offline payload manifest")
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--version", default="dev")
    parser.add_argument("--commit", default="")
    parser.add_argument("--budget-bytes", type=int, default=DEFAULT_BUDGET_BYTES)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--require-offline-wsl", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        verify_payload(args.root, manifest)
        if args.require_offline_wsl:
            verify_offline_wsl_payload(args.root, manifest)
    else:
        result = collect_payload(args.root, version=args.version, commit=args.commit,
                                 budget_bytes=args.budget_bytes)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
