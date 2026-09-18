"""Select and own a native or container model worker for one job stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .windows_worker import WindowsWorker, WindowsWorkerCommand, WindowsWorkerError


class WorkerManagerError(RuntimeError):
    pass


@dataclass
class WorkerSlot:
    model_id: str
    runtime_id: str
    worker: Any


class WorkerManager:
    """Keep worker lifecycle separate from UI and model implementation code.

    A manager owns at most one active worker per model/runtime key.  The
    manager does not install dependencies, pull images, or fall back from a
    failed verified runtime to another runtime silently.
    """

    def __init__(self) -> None:
        self._slots: dict[tuple[str, str], WorkerSlot] = {}

    def start_native(self, model_id: str, command: WindowsWorkerCommand) -> WindowsWorker:
        key = (self._id(model_id), command.runtime_id)
        if key in self._slots:
            raise WorkerManagerError(f"worker already active: {model_id}/{command.runtime_id}")
        worker = WindowsWorker(command)
        worker.start()
        self._slots[key] = WorkerSlot(model_id, command.runtime_id, worker)
        return worker

    def register_container(self, model_id: str, runtime_id: str, worker: Any) -> Any:
        key = (self._id(model_id), self._id(runtime_id))
        if key in self._slots:
            raise WorkerManagerError(f"worker already active: {model_id}/{runtime_id}")
        starter = getattr(worker, "start", None)
        if not callable(starter):
            raise WorkerManagerError("container worker must expose start()")
        starter()
        self._slots[key] = WorkerSlot(model_id, runtime_id, worker)
        return worker

    def get(self, model_id: str, runtime_id: str) -> Any:
        try:
            return self._slots[(self._id(model_id), self._id(runtime_id))].worker
        except KeyError as exc:
            raise WorkerManagerError(f"worker is not active: {model_id}/{runtime_id}") from exc

    def stop(self, model_id: str, runtime_id: str) -> None:
        key = (self._id(model_id), self._id(runtime_id))
        slot = self._slots.pop(key, None)
        if slot is None:
            return
        stopper = getattr(slot.worker, "stop", None)
        if not callable(stopper):
            raise WorkerManagerError("worker must expose stop()")
        stopper()

    def stop_all(self) -> None:
        errors: list[BaseException] = []
        for key in tuple(self._slots):
            slot = self._slots.pop(key)
            try:
                slot.worker.stop()
            except BaseException as exc:  # preserve cleanup of every slot
                errors.append(exc)
        if errors:
            raise WorkerManagerError(f"{len(errors)} worker(s) failed to stop") from errors[0]

    @staticmethod
    def _id(value: str) -> str:
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise WorkerManagerError("worker identifiers must be non-empty strings")
        return value.strip()
