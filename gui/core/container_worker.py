"""안전한 Docker 모델 worker 실행과 stdin/stdout 프레임 계약.

추가 모델 팩만 이 경로를 사용한다. 기본 모델은 Windows native worker가 담당한다.
명령은 shell 문자열로 조립하지 않으며, 컨테이너를 매 이미지마다 만들지 않고
worker 하나를 준비해 여러 요청을 처리하는 것을 전제로 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
import threading
import uuid
from typing import Sequence

from model_runtime.worker_protocol import (  # noqa: E402
    Frame as _ProtocolFrame,
    HEADER,
    MAGIC,
    MAX_HEADER_BYTES,
    MAX_PAYLOAD_BYTES,
    WorkerProtocolError,
    response_request_id,
)


class ContainerWorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Frame(_ProtocolFrame):
    """Compatibility wrapper preserving the historical container error type."""

    def encode(self) -> bytes:
        try:
            return super().encode()
        except WorkerProtocolError as exc:
            raise ContainerWorkerError(str(exc)) from exc

    @classmethod
    def read(cls, stream):
        try:
            value = _ProtocolFrame.read(stream)
        except WorkerProtocolError as exc:
            raise ContainerWorkerError(str(exc)) from exc
        return None if value is None else cls(value.header, value.payload)


def _mount_path(value: str | Path, name: str, *, writable: bool = False) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ContainerWorkerError(f"{name} must be an absolute path")
    if not path.exists():
        raise ContainerWorkerError(f"{name} does not exist: {path}")
    if path.is_symlink():
        raise ContainerWorkerError(f"{name} cannot be a symlink: {path}")
    # All three mounts are directories. ``writable`` only documents the
    # container's access mode; it must never allow a host file to be mounted
    # at a directory endpoint such as /work.
    if not path.is_dir():
        raise ContainerWorkerError(f"{name} must be a directory: {path}")
    return path.resolve()


@dataclass(frozen=True)
class ContainerCommand:
    image: str
    name: str
    argv: tuple[str, ...]
    label: str
    image_archive: str | None = None
    docker_command: tuple[str, ...] = ("docker",)


def build_container_command(
    image: str,
    *,
    model_dir: str | Path,
    data_dir: str | Path,
    work_dir: str | Path,
    name: str | None = None,
    cpus: int = 4,
    memory: str = "8g",
    image_archive: str | Path | None = None,
    docker_command: Sequence[str] = ("docker",),
) -> ContainerCommand:
    """검증된 이미지의 장기 실행 worker 명령을 만든다.

    이 함수는 Docker를 실행하지 않는다. 실행은 `ContainerWorker.start`가 담당한다.
    caller가 넘기는 image는 registry가 서명/hash 검증을 끝낸 고정 image ID여야 한다.
    """
    if not image or image.startswith("-") or any(char.isspace() for char in image):
        raise ContainerWorkerError("image must be a fixed Docker image reference")
    if isinstance(cpus, bool) or not isinstance(cpus, int) or not 1 <= cpus <= 128:
        raise ContainerWorkerError("cpus must be between 1 and 128")
    if not isinstance(memory, str) or not memory or any(char.isspace() for char in memory):
        raise ContainerWorkerError("memory must be a Docker memory limit")
    prefix = tuple(str(value) for value in docker_command)
    if (not prefix or any(not value or "\x00" in value for value in prefix)):
        raise ContainerWorkerError("docker_command must be a non-empty argv")
    model = _mount_path(model_dir, "model_dir")
    data = _mount_path(data_dir, "data_dir")
    work = _mount_path(work_dir, "work_dir", writable=True)
    archive: Path | None = None
    if image_archive is not None:
        archive = Path(image_archive).expanduser()
        if not archive.is_absolute() or archive.is_symlink() or not archive.is_file():
            raise ContainerWorkerError("image_archive must be an existing absolute file")
        archive = archive.resolve()
    container_name = name or f"deepvision-worker-{uuid.uuid4().hex[:12]}"
    if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for char in container_name):
        raise ContainerWorkerError("invalid container name")
    label = f"deepvision.owner={container_name}"
    argv = (
        *prefix, "run", "--rm", "--name", container_name,
        "--label", label, "--network", "none", "--pull=never", "--read-only",
        "--cap-drop=ALL", "--security-opt=no-new-privileges", "--log-driver=none", "--init", "-i",
        "--cpus", str(cpus), "--memory", memory,
        "--mount", f"type=bind,src={model},dst=/models,readonly",
        "--mount", f"type=bind,src={data},dst=/data,readonly",
        "--mount", f"type=bind,src={work},dst=/work",
        image,
        "--worker-stdin-stdout", "--manifest", "/models/manifest.json",
    )
    return ContainerCommand(image, container_name, argv, label,
                            None if archive is None else str(archive), prefix)


class ContainerWorker:
    """소유한 컨테이너 하나의 lifecycle만 관리하는 얇은 실행기."""

    def __init__(self, command: ContainerCommand):
        self.command = command
        self.process: subprocess.Popen[bytes] | None = None
        self._write_lock = threading.Lock()
        self._stderr_thread: threading.Thread | None = None
        self._stderr_tail = bytearray()
        self._stderr_lock = threading.Lock()

    def start(self) -> None:
        if self.process is not None:
            raise ContainerWorkerError("worker already started")
        self._load_image_archive()
        self.process = subprocess.Popen(
            self.command.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _load_image_archive(self) -> None:
        archive = self.command.image_archive
        if not archive:
            return
        loaded = subprocess.run(
            (*self.command.docker_command, "load", "--input", archive), check=False,
            capture_output=True, text=True,
        )
        if loaded.returncode != 0:
            detail = (loaded.stderr or loaded.stdout or "docker load failed").strip()
            raise ContainerWorkerError(f"offline container image load failed: {detail}")
        inspected = subprocess.run(
            (*self.command.docker_command, "image", "inspect", "--format", "{{json .RepoDigests}}",
             self.command.image), check=False, capture_output=True, text=True,
        )
        if inspected.returncode != 0:
            raise ContainerWorkerError("loaded container image cannot be inspected")
        references = []
        try:
            value = json.loads((inspected.stdout or "").strip())
            if isinstance(value, list):
                references = [item for item in value if isinstance(item, str)]
        except json.JSONDecodeError:
            references = []
        if self.command.image.startswith("sha256:"):
            id_result = subprocess.run(
                (*self.command.docker_command, "image", "inspect", "--format", "{{.Id}}", self.command.image),
                check=False, capture_output=True, text=True,
            )
            if id_result.returncode != 0 or id_result.stdout.strip() != self.command.image:
                raise ContainerWorkerError("loaded container image digest does not match the pack")
        elif self.command.image not in references:
            raise ContainerWorkerError("loaded container image digest does not match the pack")

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

    def send(self, frame: Frame) -> None:
        process = self.process
        if process is None or process.stdin is None or process.poll() is not None:
            raise ContainerWorkerError("container worker is not running")
        raw = frame.encode()
        with self._write_lock:
            try:
                process.stdin.write(raw)
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise ContainerWorkerError("container worker stdin closed") from exc

    def receive(self) -> Frame:
        process = self.process
        if process is None or process.stdout is None:
            raise ContainerWorkerError("container worker is not started")
        try:
            frame = Frame.read(process.stdout)
        except ContainerWorkerError:
            raise
        if frame is None:
            raise ContainerWorkerError(self.stderr_text() or "container worker exited without a response")
        return frame

    def request(self, frame: Frame) -> Frame:
        """Send one request and require the matching response request_id."""
        self.send(frame)
        response = self.receive()
        if response_request_id(response) != response_request_id(frame):
            raise ContainerWorkerError("container worker response request_id mismatch")
        return response

    def stderr_text(self) -> str:
        with self._stderr_lock:
            return bytes(self._stderr_tail).decode("utf-8", errors="replace")

    def _remove_owned_container(self) -> bool:
        """Remove the named container only after checking our ownership label.

        Docker names are user-visible and can be reused after an application
        restart.  Never issue ``docker rm`` for a name until inspect proves
        that the label belongs to this worker instance.
        """
        key, separator, value = self.command.label.partition("=")
        if not separator or not key or not value:
            return False
        format_arg = "{{{{index .Config.Labels \"{}\"}}}}".format(key)
        inspected = subprocess.run(
            (*self.command.docker_command, "inspect", "--format", format_arg, self.command.name),
            check=False,
            capture_output=True,
            text=True,
        )
        if inspected.returncode != 0 or inspected.stdout.strip() != value:
            return False
        subprocess.run(
            (*self.command.docker_command, "rm", "--force", self.command.name),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True

    def stop(self, timeout: float = 10.0) -> None:
        if self.process is None:
            return
        # Closing the pipe is not a container lifecycle guarantee. Ask Docker to stop
        # the owned name and then wait for the attached client process.
        subprocess.run((*self.command.docker_command, "stop", "--time", str(max(1, int(timeout))), self.command.name),
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            self.process.wait(timeout=max(1.0, timeout))
        except subprocess.TimeoutExpired:
            subprocess.run((*self.command.docker_command, "kill", self.command.name), check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.process.kill()
            self.process.wait(timeout=5)
        finally:
            # ``--rm`` normally handles this, but an interrupted docker.exe or
            # a daemon restart can leave an exited container behind.  Inspect
            # and remove it explicitly while preserving the ownership check.
            self._remove_owned_container()
            self.process = None
            thread = self._stderr_thread
            self._stderr_thread = None
            if thread is not None:
                thread.join(timeout=1.0)
