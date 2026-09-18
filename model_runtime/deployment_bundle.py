"""Build and verify a portable ONNX deployment bundle.

The native SDK consumes a deployment JSON next to its graph files.  Keeping
those files together with an exact checksum manifest prevents the common
failure mode where an ONNX file is copied without its preprocessing contract,
SAM2 companion graph, PatchCore assets, or external tensor data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any, Mapping
import zipfile


class DeploymentBundleError(ValueError):
    """The deployment bundle is missing, unsafe, or internally inconsistent."""


def _safe_name(value: str | Path) -> str:
    text = value.as_posix() if isinstance(value, Path) else str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if (not text or ":" in text or path.is_absolute() or ".." in path.parts or
            any(part == "" for part in path.parts)):
        raise DeploymentBundleError(f"unsafe deployment path: {value}")
    return "/".join(path.parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeploymentBundleError(f"cannot read deployment config: {path}") from exc
    if not isinstance(value, dict):
        raise DeploymentBundleError("deployment config must be a JSON object")
    if value.get("schema_version") != 5:
        raise DeploymentBundleError("deployment config schema_version 5 is required")
    backend = value.get("backend")
    task = value.get("task")
    if not isinstance(backend, str) or not backend or not isinstance(task, str) or not task:
        raise DeploymentBundleError("deployment config requires backend and task")
    if value.get("cpp_supported") is not True:
        raise DeploymentBundleError("deployment config is not supported by the native SDK")
    return value


def _config_candidates(root: Path) -> list[Path]:
    preferred = ("inference_config.json", "sam2.json")
    candidates = [root / name for name in preferred if (root / name).is_file()]
    if candidates:
        return candidates
    return sorted(path for path in root.glob("*.json") if path.name != "manifest.json")


def _validate_referenced_files(root: Path, config: Mapping[str, Any]) -> None:
    references: set[str] = set()
    model_path = config.get("model_path")
    if isinstance(model_path, str):
        references.add(_safe_name(model_path))
    contracts = config.get("contracts")
    if isinstance(contracts, Mapping):
        graphs = contracts.get("graphs")
        if isinstance(graphs, Mapping):
            for graph in graphs.values():
                if isinstance(graph, Mapping) and isinstance(graph.get("file"), str):
                    references.add(_safe_name(graph["file"]))
    for name in references:
        path = root / Path(*PurePosixPath(name).parts)
        if not path.is_file() or path.is_symlink():
            raise DeploymentBundleError(f"deployment graph is missing: {name}")


def _copy_tree(source: Path, target: Path) -> list[str]:
    names: list[str] = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise DeploymentBundleError(f"deployment source symlink is not allowed: {path}")
        if not path.is_file():
            continue
        relative = _safe_name(path.relative_to(source))
        if relative == "manifest.json":
            continue
        destination = target / Path(*PurePosixPath(relative).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        names.append(relative)
    return names


def _write_manifest(root: Path, config_name: str) -> dict[str, Any]:
    config = _read_config(root / config_name)
    _validate_referenced_files(root, config)
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise DeploymentBundleError(f"deployment bundle symlink is not allowed: {path}")
        if path.is_file() and path.name != "manifest.json":
            name = _safe_name(path.relative_to(root))
            files[name] = {"size": path.stat().st_size, "sha256": _sha256(path)}
    if config_name not in files:
        raise DeploymentBundleError("deployment bundle config is not included")
    manifest = {
        "schema_version": 1,
        "bundle_type": "onnx-deployment",
        "config": config_name,
        "backend": config["backend"],
        "task": config["task"],
        "verification": config.get("verification", "unknown"),
        "files": files,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_deployment_bundle(source: str | Path, output: str | Path) -> Path:
    """Create an atomic directory bundle from an ONNX file or export folder."""
    source_path = Path(source).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if not source_path.exists() or source_path.is_symlink():
        raise DeploymentBundleError(f"deployment source does not exist: {source_path}")
    if destination.suffix.lower() != ".dvdeploy":
        raise DeploymentBundleError("deployment bundle output must use .dvdeploy")
    if destination.exists():
        raise DeploymentBundleError(f"deployment bundle already exists: {destination}")
    if source_path.is_file():
        if source_path.suffix.lower() != ".onnx":
            raise DeploymentBundleError("deployment source file must be .onnx")
        config_path = source_path.with_suffix(".json")
        if not config_path.is_file():
            raise DeploymentBundleError(f"deployment config is missing: {config_path}")
        source_root = source_path.parent
        names = [source_path.name, config_path.name]
        config_name = config_path.name
    elif source_path.is_dir():
        candidates = _config_candidates(source_path)
        if len(candidates) != 1:
            raise DeploymentBundleError("deployment source must contain exactly one config JSON")
        config_name = candidates[0].name
        names = None
        source_root = source_path
    else:
        raise DeploymentBundleError("deployment source must be a file or directory")

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
    try:
        if names is None:
            _copy_tree(source_root, staging)
        else:
            for name in names:
                source_file = source_root / name
                shutil.copy2(source_file, staging / name)
        _write_manifest(staging, config_name)
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def verify_deployment_bundle(bundle: str | Path) -> dict[str, Any]:
    """Verify every bundle byte and return its manifest."""
    path = Path(bundle).expanduser().resolve()
    temporary: Path | None = None
    if path.is_file() and path.suffix.lower() == ".zip":
        temporary = Path(tempfile.mkdtemp(prefix=".dvdeploy-verify-"))
        try:
            with zipfile.ZipFile(path) as archive:
                extracted: set[str] = set()
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    name = _safe_name(info.filename)
                    mode = (info.external_attr >> 16) & 0xFFFF
                    if info.create_system == 3 and stat.S_ISLNK(mode):
                        raise DeploymentBundleError(f"deployment archive symlink is not allowed: {name}")
                    if name in extracted:
                        raise DeploymentBundleError(f"deployment archive contains duplicate file: {name}")
                    extracted.add(name)
                    target = temporary / Path(*PurePosixPath(name).parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
            path = temporary
        except (OSError, zipfile.BadZipFile) as exc:
            raise DeploymentBundleError("cannot read deployment bundle archive") from exc
    if not path.is_dir() or path.is_symlink():
        raise DeploymentBundleError("deployment bundle must be a directory or zip archive")
    try:
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise DeploymentBundleError("deployment bundle manifest.json is missing")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DeploymentBundleError("deployment bundle manifest is invalid") from exc
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise DeploymentBundleError("unsupported deployment bundle manifest")
        config_name = manifest.get("config")
        files = manifest.get("files")
        if not isinstance(config_name, str) or not isinstance(files, Mapping):
            raise DeploymentBundleError("deployment bundle manifest is incomplete")
        config_name = _safe_name(config_name)
        actual: set[str] = set()
        manifest_names: set[str] = set()
        for relative, expected in files.items():
            name = _safe_name(relative)
            if name in manifest_names:
                raise DeploymentBundleError(f"duplicate deployment checksum: {name}")
            manifest_names.add(name)
            if not isinstance(expected, Mapping):
                raise DeploymentBundleError(f"invalid deployment checksum: {name}")
            file_path = path / Path(*PurePosixPath(name).parts)
            if not file_path.is_file() or file_path.is_symlink():
                raise DeploymentBundleError(f"deployment file is missing: {name}")
            if file_path.stat().st_size != expected.get("size") or _sha256(file_path) != expected.get("sha256"):
                raise DeploymentBundleError(f"deployment file checksum mismatch: {name}")
            actual.add(name)
        actual_files = {_safe_name(file.relative_to(path)) for file in path.rglob("*")
                        if file.is_file() and file.name != "manifest.json"}
        if actual != actual_files:
            raise DeploymentBundleError("deployment manifest does not cover every file")
        config = _read_config(path / Path(*PurePosixPath(config_name).parts))
        _validate_referenced_files(path, config)
        return manifest
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


__all__ = ["DeploymentBundleError", "build_deployment_bundle", "verify_deployment_bundle"]
