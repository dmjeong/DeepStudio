"""Offline model/runtime asset resolution with hash and size verification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping


class AssetResolutionError(ValueError):
    """An asset is missing, outside an allowed root, or does not match its contract."""


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_name(value: str | Path) -> str:
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts or any(part == "" for part in path.parts):
        raise AssetResolutionError(f"asset path must be a relative safe path: {value}")
    return "/".join(path.parts)


@dataclass(frozen=True)
class Asset:
    name: str
    path: Path
    size: int
    sha256: str
    source: str


class AssetResolver:
    """Resolve project, installed-pack, and cache roots in that order.

    No resolver method performs a download.  A missing asset is an explicit
    error so offline Windows installs cannot silently reach a model hub.
    """

    def __init__(self, roots: Iterable[str | Path]):
        self.roots: tuple[Path, ...] = tuple(self._root(value) for value in roots)
        if not self.roots:
            raise AssetResolutionError("at least one asset root is required")

    @staticmethod
    def _root(value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise AssetResolutionError("asset roots must be absolute")
        if path.exists() and path.is_symlink():
            raise AssetResolutionError("asset roots cannot be symlinks")
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    @staticmethod
    def _candidate(root: Path, name: str) -> Path:
        raw = root / Path(*PurePosixPath(name).parts)
        if raw.is_symlink():
            raise AssetResolutionError(f"asset cannot be a symlink: {name}")
        candidate = raw.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise AssetResolutionError("asset path escapes its root") from exc
        return candidate

    def resolve(self, name: str | Path, *, sha256: str | None = None,
                size: int | None = None) -> Asset:
        safe_name = _relative_name(name)
        expected_hash = None if sha256 is None else str(sha256).lower()
        if expected_hash is not None and (len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash)):
            raise AssetResolutionError("sha256 must be a 64-character hexadecimal value")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
            raise AssetResolutionError("asset size must be a non-negative integer")
        for index, root in enumerate(self.roots):
            candidate = self._candidate(root, safe_name)
            if not candidate.is_file():
                continue
            actual_size = candidate.stat().st_size
            actual_hash = _digest(candidate)
            if size is not None and actual_size != size:
                raise AssetResolutionError(f"asset size mismatch: {safe_name}")
            if expected_hash is not None and actual_hash.lower() != expected_hash:
                raise AssetResolutionError(f"asset hash mismatch: {safe_name}")
            return Asset(safe_name, candidate, actual_size, actual_hash, f"root-{index}")
        raise AssetResolutionError(f"offline asset not found: {safe_name}")

    def resolve_manifest(self, manifest: Mapping[str, object], key: str = "assets") -> tuple[Asset, ...]:
        entries = manifest.get(key)
        if not isinstance(entries, list):
            raise AssetResolutionError(f"manifest {key} must be a list")
        result: list[Asset] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise AssetResolutionError(f"manifest {key} contains an invalid entry")
            name = entry.get("path")
            if not isinstance(name, str):
                raise AssetResolutionError(f"manifest {key} entry requires path")
            result.append(self.resolve(name, sha256=entry.get("sha256"), size=entry.get("size")))
        return tuple(result)
