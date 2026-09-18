"""Windows native worker launcher and lifecycle.

The launcher never invokes a shell and never resolves a worker through PATH.
Release payloads pass absolute paths to a frozen worker executable (or the
bundled Python launcher during development).  This module is platform-neutral
for command construction, while process startup uses Windows process-group
flags when available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import BinaryIO, Mapping, Sequence

from .worker_protocol import Frame, WorkerProtocolError


class WindowsWorkerError(RuntimeError):
    """Worker command, startup, protocol, or shutdown failure."""


def _absolute_file(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise WindowsWorkerError(f"{name} must be an absolute path")
    # Development virtual environments commonly expose Python through a
    # symlink. Resolve it before checking the executable; release callers can
    # additionally pin the resolved path in their payload manifest.
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise WindowsWorkerError(f"{name} must be an existing regular file") from exc
    if not path.is_file():
        raise WindowsWorkerError(f"{name} must be an existing regular file")
    return path


def _absolute_dir(value: str | Path, name: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise WindowsWorkerError(f"{name} must be an absolute path")
    if path.is_symlink() or not path.is_dir():
        raise WindowsWorkerError(f"{name} must be an existing directory")
    return path.resolve()


def _safe_token(value: str, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise WindowsWorkerError(f"{name} must be a non-empty argument")
    return value


@dataclass(frozen=True)
class WindowsWorkerCommand:
    executable: Path
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    runtime_id: str

    @property
    def command(self) -> tuple[str, ...]:
        return (str(self.executable), *self.argv)


def build_worker_command(executable: str | Path, *, runtime_id: str,
                         cwd: str | Path, args: Sequence[str] = (),
                         env: Mapping[str, str] | None = None) -> WindowsWorkerCommand:
    """Build an absolute, shell-free command for a bundled worker."""
    binary = _absolute_file(executable, "worker executable")
    workdir = _absolute_dir(cwd, "worker cwd")
    runtime = _safe_token(runtime_id, "runtime_id")
    tokens = tuple(_safe_token(str(value), "worker argument") for value in args)
    merged = {str(key): str(value) for key, value in (env or {}).items()}
    if any("\x00" in key or "\x00" in value for key, value in merged.items()):
        raise WindowsWorkerError("worker environment contains NUL")
    # Do not inherit an externally selected Python/conda runtime in a frozen
    # release.  The caller may explicitly provide a PATH for bundled DLLs.
    if "PYTHONHOME" in merged or "PYTHONPATH" in merged:
        raise WindowsWorkerError("worker environment cannot override Python search paths")
    return WindowsWorkerCommand(binary, tokens, workdir, merged, runtime)


@dataclass
class WindowsWorker:
    command: WindowsWorkerCommand
    process: subprocess.Popen[bytes] | None = field(default=None, init=False)
    _write_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def start(self) -> None:
        if self.process is not None:
            raise WindowsWorkerError("worker is already started")
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        try:
            self.process = subprocess.Popen(
                self.command.command,
                cwd=str(self.command.cwd),
                env={**os.environ, **self.command.env},
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                creationflags=flags,
            )
        except OSError as exc:
            raise WindowsWorkerError(f"cannot start worker: {exc}") from exc

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def send(self, frame: Frame) -> None:
        process = self.process
        if process is None or process.stdin is None or not self.running:
            raise WindowsWorkerError("worker is not running")
        raw = frame.encode()
        with self._write_lock:
            try:
                process.stdin.write(raw)
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise WindowsWorkerError("worker stdin closed") from exc

    def receive(self) -> Frame:
        process = self.process
        if process is None or process.stdout is None:
            raise WindowsWorkerError("worker is not started")
        try:
            frame = Frame.read(process.stdout)
        except WorkerProtocolError as exc:
            raise WindowsWorkerError(str(exc)) from exc
        if frame is None:
            error = self.stderr_text()
            raise WindowsWorkerError(error or "worker exited without a response")
        return frame

    def stderr_text(self) -> str:
        process = self.process
        if process is None or process.stderr is None:
            return ""
        # Reading stderr to EOF is safe after process exit and avoids mixing it
        # into the binary stdout protocol while preserving diagnostics.
        if process.poll() is None:
            return ""
        try:
            return process.stderr.read().decode("utf-8", errors="replace")[-8192:]
        except OSError:
            return ""

    def stop(self, timeout: float = 10.0) -> None:
        process = self.process
        if process is None:
            return
        if timeout <= 0:
            raise WindowsWorkerError("timeout must be positive")
        try:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=min(5.0, timeout))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
        finally:
            self.process = None

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "WindowsWorker":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()
