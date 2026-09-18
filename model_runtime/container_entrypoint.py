"""Entrypoint contract for an offline `.dvmodel` Docker worker.

The model pack may declare ``worker_entrypoint`` as ``module:factory``.  The
factory is imported only inside the isolated container and must return a
mapping of DVW1 command names to handlers.  A pack without a plugin still has
the protocol healthcheck, but model commands return explicit unsupported
responses rather than pretending to run.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Mapping

from .worker_server import WorkerServer


class ContainerEntrypointError(RuntimeError):
    pass


def _load_handlers(manifest_path: Path):
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
    if not module_name or not factory_name or any(part in module_name for part in ("/", "\\", "..")):
        raise ContainerEntrypointError("worker_entrypoint module is unsafe")
    models_root = manifest_path.parent.resolve()
    if str(models_root) not in sys.path:
        sys.path.insert(0, str(models_root))
    try:
        factory = getattr(importlib.import_module(module_name), factory_name)
    except (ImportError, AttributeError) as exc:
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
