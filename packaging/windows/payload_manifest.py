"""Build and verify an offline Windows payload manifest.

This is intentionally platform-neutral so CI can check the release payload
before a Windows-only WiX/PyInstaller build. It never downloads dependencies.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
from typing import Any


MAX_SIGNED_PE_BYTES = 4 * 1024 * 1024 * 1024
DEFAULT_BUDGET_BYTES = int(3.5 * 1024 * 1024 * 1024)


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


def main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Build or verify a Windows offline payload manifest")
    parser.add_argument("root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--version", default="dev")
    parser.add_argument("--commit", default="")
    parser.add_argument("--budget-bytes", type=int, default=DEFAULT_BUDGET_BYTES)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        verify_payload(args.root, json.loads(args.manifest.read_text(encoding="utf-8")))
    else:
        result = collect_payload(args.root, version=args.version, commit=args.commit,
                                 budget_bytes=args.budget_bytes)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
