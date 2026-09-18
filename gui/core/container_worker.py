"""안전한 Docker 모델 worker 실행과 stdin/stdout 프레임 계약.

추가 모델 팩만 이 경로를 사용한다. 기본 모델은 Windows native worker가 담당한다.
명령은 shell 문자열로 조립하지 않으며, 컨테이너를 매 이미지마다 만들지 않고
worker 하나를 준비해 여러 요청을 처리하는 것을 전제로 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
import subprocess
from typing import BinaryIO, Mapping
import uuid


MAGIC = b"DVW1"
HEADER = struct.Struct("<4sIQ")
MAX_HEADER_BYTES = 64 * 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


class ContainerWorkerError(RuntimeError):
    pass


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    """Pipes may return a partial read even when more bytes are available."""
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@dataclass(frozen=True)
class Frame:
    header: Mapping[str, object]
    payload: bytes = b""

    def encode(self) -> bytes:
        header = json.dumps(self.header, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(header) > MAX_HEADER_BYTES:
            raise ContainerWorkerError("worker frame header too large")
        if len(self.payload) > MAX_PAYLOAD_BYTES:
            raise ContainerWorkerError("worker frame payload too large")
        return HEADER.pack(MAGIC, len(header), len(self.payload)) + header + self.payload

    @classmethod
    def read(cls, stream: BinaryIO) -> "Frame | None":
        prefix = _read_exact(stream, HEADER.size)
        if not prefix:
            return None
        if len(prefix) != HEADER.size:
            raise ContainerWorkerError("truncated worker frame header")
        magic, header_size, payload_size = HEADER.unpack(prefix)
        if magic != MAGIC or header_size > MAX_HEADER_BYTES or payload_size > MAX_PAYLOAD_BYTES:
            raise ContainerWorkerError("invalid worker frame limits or magic")
        raw_header = _read_exact(stream, header_size)
        payload = _read_exact(stream, payload_size)
        if len(raw_header) != header_size or len(payload) != payload_size:
            raise ContainerWorkerError("truncated worker frame payload")
        try:
            header = json.loads(raw_header.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContainerWorkerError("invalid worker frame JSON") from exc
        if not isinstance(header, dict) or not isinstance(header.get("type"), str):
            raise ContainerWorkerError("worker frame requires an object and type")
        return cls(header, payload)


def _mount_path(value: str | Path, name: str, *, writable: bool = False) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ContainerWorkerError(f"{name} must be an absolute path")
    if not path.exists():
        raise ContainerWorkerError(f"{name} does not exist: {path}")
    if path.is_symlink():
        raise ContainerWorkerError(f"{name} cannot be a symlink: {path}")
    if not writable and not path.is_dir():
        raise ContainerWorkerError(f"{name} must be a directory: {path}")
    return path.resolve()


@dataclass(frozen=True)
class ContainerCommand:
    image: str
    name: str
    argv: tuple[str, ...]
    label: str


def build_container_command(
    image: str,
    *,
    model_dir: str | Path,
    data_dir: str | Path,
    work_dir: str | Path,
    name: str | None = None,
    cpus: int = 4,
    memory: str = "8g",
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
    model = _mount_path(model_dir, "model_dir")
    data = _mount_path(data_dir, "data_dir")
    work = _mount_path(work_dir, "work_dir", writable=True)
    container_name = name or f"deepvision-worker-{uuid.uuid4().hex[:12]}"
    if any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for char in container_name):
        raise ContainerWorkerError("invalid container name")
    label = f"deepvision.owner={container_name}"
    argv = (
        "docker", "run", "--rm", "--name", container_name,
        "--label", label, "--network", "none", "--pull=never", "--read-only",
        "--cap-drop=ALL", "--security-opt=no-new-privileges", "--log-driver=none", "--init", "-i",
        "--cpus", str(cpus), "--memory", memory,
        "--mount", f"type=bind,src={model},dst=/models,readonly",
        "--mount", f"type=bind,src={data},dst=/data,readonly",
        "--mount", f"type=bind,src={work},dst=/work",
        image,
        "--worker-stdin-stdout",
    )
    return ContainerCommand(image, container_name, argv, label)


class ContainerWorker:
    """소유한 컨테이너 하나의 lifecycle만 관리하는 얇은 실행기."""

    def __init__(self, command: ContainerCommand):
        self.command = command
        self.process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        if self.process is not None:
            raise ContainerWorkerError("worker already started")
        self.process = subprocess.Popen(
            self.command.argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )

    def stop(self, timeout: float = 10.0) -> None:
        if self.process is None:
            return
        # Closing the pipe is not a container lifecycle guarantee. Ask Docker to stop
        # the owned name and then wait for the attached client process.
        subprocess.run(("docker", "stop", "--time", str(max(1, int(timeout))), self.command.name),
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            self.process.wait(timeout=max(1.0, timeout))
        except subprocess.TimeoutExpired:
            subprocess.run(("docker", "kill", self.command.name), check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.process.kill()
            self.process.wait(timeout=5)
        finally:
            self.process = None
