"""Compatibility exports for the model worker manager."""

from .worker_manager import WorkerManager, WorkerManagerError, WorkerSlot

__all__ = ["WorkerManager", "WorkerManagerError", "WorkerSlot"]
