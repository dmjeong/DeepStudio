"""Entrypoint contract for an offline `.dvmodel` Docker worker.

The model pack may declare ``worker_entrypoint`` as ``module:factory``.  The
factory is imported only inside the isolated container and must return a
mapping of DVW1 command names to handlers.  A pack without a plugin still has
the protocol healthcheck, but model commands return explicit unsupported
responses rather than pretending to run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Mapping

from .worker_server import WorkerServer


class ContainerEntrypointError(RuntimeError):
    pass


def _load_pack_module(root: Path, module_name: str):
    """Load a plugin module from the pack root, never from ambient sys.path."""
    relative = Path(*module_name.split("."))
    candidates = (root / relative.with_suffix(".py"), root / relative / "__init__.py")
    module_path = None
    package_root = None
    for candidate in candidates:
        if candidate.is_symlink():
            raise ContainerEntrypointError("worker entrypoint module cannot be a symlink")
        if candidate.is_file():
            resolved = candidate.resolve(strict=True)
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ContainerEntrypointError("worker entrypoint module escapes pack root") from exc
            module_path = resolved
            if candidate.name == "__init__.py":
                package_root = str(resolved.parent)
            break
    if module_path is None:
        raise ContainerEntrypointError(f"worker entrypoint module is missing from pack: {module_name}")
    spec = importlib.util.spec_from_file_location(
        module_name, str(module_path),
        submodule_search_locations=[package_root] if package_root else None)
    if spec is None or spec.loader is None:
        raise ContainerEntrypointError(f"cannot load worker entrypoint: {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise ContainerEntrypointError(f"cannot load worker entrypoint: {module_name}") from exc
    return module


def _load_handlers(manifest_path: Path):
    manifest_path = Path(manifest_path).expanduser()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ContainerEntrypointError("model manifest must be an existing regular file")
    try:
        manifest_path = manifest_path.resolve(strict=True)
    except OSError as exc:
        raise ContainerEntrypointError(f"cannot resolve model manifest: {manifest_path}") from exc
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContainerEntrypointError(f"cannot read model manifest: {manifest_path}") from exc
    if not isinstance(manifest, Mapping):
        raise ContainerEntrypointError("model manifest must be an object")
    entrypoint = manifest.get("worker_entrypoint")
    if entrypoint is None:
        return {}
    if not isinstance(entrypoint, str) or entrypoint.count(":") != 1:
        raise ContainerEntrypointError("worker_entrypoint must use module:factory")
    module_name, factory_name = entrypoint.split(":", 1)
    module_parts = module_name.split(".")
    if (not module_name or not factory_name or
            any(not part.isidentifier() for part in module_parts) or
            not factory_name.isidentifier() or
            any(char.isspace() or char == "\x00" for char in entrypoint)):
        raise ContainerEntrypointError("worker_entrypoint module is unsafe")
    models_root = manifest_path.parent.resolve(strict=True)
    if str(models_root) not in sys.path:
        sys.path.insert(0, str(models_root))
    try:
        factory = getattr(_load_pack_module(models_root, module_name), factory_name)
    except AttributeError as exc:
        raise ContainerEntrypointError(f"cannot load worker entrypoint: {entrypoint}") from exc
    handlers = factory(manifest)
    if not isinstance(handlers, Mapping):
        raise ContainerEntrypointError("worker entrypoint must return a command mapping")
    return handlers


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a model-pack stdin/stdout worker")
    parser.add_argument("--worker-stdin-stdout", action="store_true")
    parser.add_argument("--manifest", type=Path, default=Path("/models/manifest.json"))
    args = parser.parse_args(argv)
    if not args.worker_stdin_stdout:
        parser.error("--worker-stdin-stdout is required")
    try:
        handlers = _load_handlers(args.manifest)
        return WorkerServer(handlers).serve(sys.stdin.buffer, sys.stdout.buffer, error_stream=sys.stderr.buffer)
    except ContainerEntrypointError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
