"""Windows native worker launcher and lifecycle.

The launcher never invokes a shell and never resolves a worker through PATH.
Release payloads pass absolute paths to a frozen worker executable (or the
bundled Python launcher during development).  This module is platform-neutral
for command construction.  Windows startup uses a kill-on-close Job Object
for the worker tree and process-group flags when available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import subprocess
import threading
from typing import BinaryIO, Mapping, Sequence

from .worker_protocol import Frame, WorkerProtocolError


class WindowsWorkerError(RuntimeError):
    """Worker command, startup, protocol, or shutdown failure."""


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    """The small part of the Win32 Job Object contract we need.

    The structure is declared on every platform so importing this module stays
    cheap and dependency-free.  The API is called only when ``os.name ==
    "nt"``.
    """

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("per_process_user_time_limit", ctypes.c_longlong),
            ("per_job_user_time_limit", ctypes.c_longlong),
            ("limit_flags", wintypes.DWORD),
            ("minimum_working_set_size", ctypes.c_size_t),
            ("maximum_working_set_size", ctypes.c_size_t),
            ("active_process_limit", wintypes.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority_class", wintypes.DWORD),
            ("scheduling_class", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [("values", ctypes.c_ulonglong * 6)]

    _fields_ = [
        ("basic_limit_information", _BasicLimitInformation),
        ("io_info", _IoCounters),
        ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory_used", ctypes.c_size_t),
        ("peak_job_memory_used", ctypes.c_size_t),
    ]


class _WindowsJobObject:
    """Own a worker process tree through a Windows Job Object.

    ``Popen`` does not expose an extended STARTUPINFO attribute that assigns a
    job atomically at process creation.  We assign immediately after creation,
    before the worker receives its first frame, and enable
    ``KILL_ON_JOB_CLOSE`` so an application crash or forced close cannot leave
    descendants behind.  Non-Windows development hosts return ``None`` from
    :meth:`create` and use the existing process-group path.
    """

    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

    def __init__(self, kernel32, handle):
        self._kernel32 = kernel32
        self._handle = handle

    @classmethod
    def create(cls):
        if os.name != "nt":
            return None
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
            kernel32.CreateJobObjectW.restype = wintypes.HANDLE
            kernel32.SetInformationJobObject.argtypes = (
                wintypes.HANDLE, wintypes.INT, wintypes.LPVOID, wintypes.DWORD)
            kernel32.SetInformationJobObject.restype = wintypes.BOOL
            kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
            kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
            kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
            kernel32.TerminateJobObject.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            info = _JobObjectExtendedLimitInformation()
            info.basic_limit_information.limit_flags = cls._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(
                    handle, cls._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                    ctypes.byref(info), ctypes.sizeof(info)):
                error = ctypes.WinError(ctypes.get_last_error())
                kernel32.CloseHandle(handle)
                raise error
            return cls(kernel32, handle)
        except (OSError, AttributeError) as exc:
            raise WindowsWorkerError(f"cannot create Windows Job Object: {exc}") from exc

    def assign(self, process_handle) -> None:
        raw_handle = getattr(process_handle, "handle", process_handle)
        if not self._kernel32.AssignProcessToJobObject(self._handle, raw_handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def terminate(self) -> None:
        if self._handle and not self._kernel32.TerminateJobObject(self._handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle:
            self._kernel32.CloseHandle(handle)


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
    _job: _WindowsJobObject | None = field(default=None, init=False, repr=False)
    _write_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _stderr_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stderr_tail: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _stderr_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def start(self) -> None:
        if self.process is not None:
            raise WindowsWorkerError("worker is already started")
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
        job = _WindowsJobObject.create()
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
            if job is not None:
                # Popen has already created the process by this point. Attach
                # before the worker can receive any protocol frame; the job's
                # kill-on-close policy then owns descendants as well.
                # ``_handle`` is the Windows Popen handle.  The pid fallback
                # keeps this path easy to exercise with a portable fake job;
                # real Windows workers always expose ``_handle``.
                job.assign(getattr(self.process, "_handle", self.process.pid))
                self._job = job
            self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
            self._stderr_thread.start()
        except (OSError, WindowsWorkerError) as exc:
            if job is not None and self._job is None:
                job.close()
            if self.process is not None:
                try:
                    self.process.kill()
                    self.process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                self.process = None
            if isinstance(exc, WindowsWorkerError):
                raise
            raise WindowsWorkerError(f"cannot start worker: {exc}") from exc

    def _drain_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        try:
            while True:
                chunk = process.stderr.read(4096)
                if not chunk:
                    return
                with self._stderr_lock:
                    self._stderr_tail.extend(chunk)
                    del self._stderr_tail[:-8192]
        except OSError:
            return

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
        with self._stderr_lock:
            return bytes(self._stderr_tail).decode("utf-8", errors="replace")

    def stop(self, timeout: float = 10.0) -> None:
        process = self.process
        if process is None:
            if self._job is not None:
                self._job.close()
                self._job = None
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
                if self._job is not None:
                    try:
                        self._job.terminate()
                    except OSError:
                        process.terminate()
                else:
                    process.terminate()
                try:
                    process.wait(timeout=min(5.0, timeout))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
        finally:
            self.process = None
            if self._job is not None:
                self._job.close()
                self._job = None
            thread = self._stderr_thread
            self._stderr_thread = None
            if thread is not None:
                thread.join(timeout=1.0)

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "WindowsWorker":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()
