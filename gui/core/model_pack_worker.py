"""Host-side adapter for an installed Docker model pack.

The desktop/web worker owns the container lifecycle while the pack owns model
code.  Only the DVW1 protocol crosses the boundary; the manifest is read and
validated before Docker is started.  This keeps adding a model independent of
the host Python imports and gives training, inference, and export the same
request contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence
import uuid

from core.container_worker import (ContainerWorker, ContainerWorkerError,
                                   build_container_command)
from model_runtime.special_contracts import (SpecialContractError,
                                             validate_container_entrypoint,
                                             validate_special_manifest)
from model_runtime.managed_wsl import (ManagedWslError,
                                       configured_docker_command)
from model_runtime.worker_protocol import Frame, request_frame


class ModelPackWorkerError(RuntimeError):
    """The pack manifest, worker process, or response violated the contract."""


def _read_manifest(pack_dir: str | Path) -> tuple[Path, Mapping[str, Any]]:
    root = Path(pack_dir).expanduser()
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ModelPackWorkerError("pack_dir must be an existing absolute directory")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ModelPackWorkerError("installed model pack manifest.json is missing")
    manifest_path = manifest_path.resolve()
    if manifest_path.parent != root.resolve():
        raise ModelPackWorkerError("installed model pack manifest.json escapes its root")
    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise ModelPackWorkerError("installed model pack cannot contain symlinks")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelPackWorkerError("installed model pack manifest is invalid") from exc
    if not isinstance(manifest, Mapping):
        raise ModelPackWorkerError("installed model pack manifest must be an object")
    if manifest.get("schema_version") != 1:
        raise ModelPackWorkerError("unsupported model pack manifest schema")
    model_id = manifest.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ModelPackWorkerError("model pack model_id is required")
    runtimes = manifest.get("runtimes", ())
    if not isinstance(runtimes, (list, tuple)) or "container" not in runtimes:
        raise ModelPackWorkerError("model pack does not declare the container runtime")
    try:
        validate_special_manifest(manifest)
        validate_container_entrypoint(manifest)
    except SpecialContractError as exc:
        raise ModelPackWorkerError(str(exc)) from exc
    return root.resolve(), manifest


def _container_image(manifest: Mapping[str, Any]) -> str:
    requirements = manifest.get("runtime_requirements", {})
    if not isinstance(requirements, Mapping):
        raise ModelPackWorkerError("runtime_requirements must be an object")
    image = manifest.get("container_image", requirements.get("container_image"))
    if not isinstance(image, str) or not image.strip() or any(char.isspace() for char in image):
        raise ModelPackWorkerError("container_image must be a fixed image reference")
    # A mutable tag would make an offline installed pack change underneath a
    # project.  Local development can still use a content ID or digest.
    if "@sha256:" not in image and not image.startswith("sha256:"):
        raise ModelPackWorkerError("container_image must be pinned by digest")
    return image


def _container_image_archive(root: Path, manifest: Mapping[str, Any]) -> Path | None:
    value = manifest.get("container_image_archive")
    if value is None:
        return None
    if (not isinstance(value, str) or not value.strip() or value.startswith("/") or
            ":" in value or "\\" in value or "//" in value or ".." in value.split("/")):
        raise ModelPackWorkerError("container_image_archive must be a safe relative path")
    relative = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ModelPackWorkerError("container_image_archive must be a safe relative path")
    archive = (root / Path(*relative.parts)).resolve()
    if archive.parent != root and root not in archive.parents:
        raise ModelPackWorkerError("container_image_archive escapes the installed pack")
    if archive.is_symlink() or not archive.is_file():
        raise ModelPackWorkerError("container_image_archive is missing from the installed pack")
    return archive


class ModelPackWorker:
    """Long-lived worker for one installed `.dvmodel` directory."""

    # ``close`` is part of the DVW1 lifecycle.  Keeping it in the host
    # allowlist lets ``close()`` perform a graceful protocol shutdown before
    # the process fallback terminates the container.
    COMMANDS = frozenset({"hello", "describe", "prepare", "train", "infer", "export", "cancel", "close"})

    def __init__(self, worker: ContainerWorker, model_id: str, manifest: Mapping[str, Any]):
        self._worker = worker
        self.model_id = model_id
        self.manifest = dict(manifest)

    @classmethod
    def from_installed_pack(cls, pack_dir: str | Path, *, data_dir: str | Path,
                            work_dir: str | Path, cpus: int = 4,
                            memory: str = "8g", name: str | None = None,
                            docker_command: Sequence[str] | None = None) -> "ModelPackWorker":
        root, manifest = _read_manifest(pack_dir)
        image = _container_image(manifest)
        image_archive = _container_image_archive(root, manifest)
        if docker_command is None:
            try:
                docker_command = configured_docker_command()
            except ManagedWslError as exc:
                raise ModelPackWorkerError(str(exc)) from exc
        command = build_container_command(image, model_dir=root, data_dir=data_dir,
                                          work_dir=work_dir, cpus=cpus,
                                          memory=memory, name=name,
                                          image_archive=image_archive,
                                          docker_command=docker_command)
        return cls(ContainerWorker(command), str(manifest["model_id"]), manifest)

    def start(self) -> None:
        try:
            self._worker.start()
        except ContainerWorkerError as exc:
            raise ModelPackWorkerError(str(exc)) from exc

    def request(self, command: str, *, fields: Mapping[str, Any] | None = None,
                payload: bytes = b"") -> tuple[Mapping[str, Any], bytes]:
        if command not in self.COMMANDS:
            raise ModelPackWorkerError(f"unsupported model pack command: {command}")
        request_id = uuid.uuid4().hex
        frame = request_frame(request_id, command, model_id=self.model_id,
                              **dict(fields or {}), payload=payload)
        try:
            response = self._worker.request(frame)
        except (ContainerWorkerError, ValueError) as exc:
            raise ModelPackWorkerError(str(exc)) from exc
        if response.header.get("type") == "error" or response.header.get("status") == "error":
            raise ModelPackWorkerError(str(response.header.get("message", "model pack worker error")))
        return response.header, response.payload

    def json_request(self, command: str, value: Mapping[str, Any] | None = None) -> dict[str, Any]:
        header, payload = self.request(command, fields={"encoding": "json"},
                                       payload=json.dumps(value or {}, ensure_ascii=False,
                                                          allow_nan=False).encode("utf-8"))
        if payload:
            try:
                decoded = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ModelPackWorkerError("model pack returned invalid JSON") from exc
            if not isinstance(decoded, dict):
                raise ModelPackWorkerError("model pack JSON response must be an object")
            return {**dict(header), "payload": decoded}
        return dict(header)

    def close(self) -> None:
        try:
            if self._worker.process is not None and self._worker.process.poll() is None:
                try:
                    self.request("close")
                except ModelPackWorkerError:
                    pass
        finally:
            self._worker.stop()

    def __enter__(self) -> "ModelPackWorker":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["ModelPackWorker", "ModelPackWorkerError"]
