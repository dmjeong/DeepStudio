"""Shared framed protocol for native and container model workers.

The host and worker communicate over stdin/stdout.  stdout is reserved for
``DVW1`` frames; human-readable diagnostics belong on stderr.  Keeping this
module free of Qt, Docker and model imports lets the Windows launcher and the
container worker use the same wire contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import struct
from typing import BinaryIO, Mapping


MAGIC = b"DVW1"
HEADER = struct.Struct("<4sIQ")
MAX_HEADER_BYTES = 64 * 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


class WorkerProtocolError(ValueError):
    """A frame is malformed or exceeds the negotiated safety limits."""


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _json_header(header: Mapping[str, object]) -> bytes:
    if not isinstance(header, Mapping):
        raise WorkerProtocolError("worker frame header must be an object")
    try:
        raw = json.dumps(header, ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WorkerProtocolError("worker frame header is not JSON serializable") from exc
    if len(raw) > MAX_HEADER_BYTES:
        raise WorkerProtocolError("worker frame header too large")
    return raw


@dataclass(frozen=True)
class Frame:
    """One request, progress event, response, or error frame."""

    header: Mapping[str, object]
    payload: bytes = b""

    def encode(self) -> bytes:
        if not isinstance(self.payload, (bytes, bytearray, memoryview)):
            raise WorkerProtocolError("worker frame payload must be bytes")
        payload = bytes(self.payload)
        if len(payload) > MAX_PAYLOAD_BYTES:
            raise WorkerProtocolError("worker frame payload too large")
        header = _json_header(self.header)
        return HEADER.pack(MAGIC, len(header), len(payload)) + header + payload

    @classmethod
    def read(cls, stream: BinaryIO) -> "Frame | None":
        prefix = _read_exact(stream, HEADER.size)
        if not prefix:
            return None
        if len(prefix) != HEADER.size:
            raise WorkerProtocolError("truncated worker frame header")
        magic, header_size, payload_size = HEADER.unpack(prefix)
        if magic != MAGIC:
            raise WorkerProtocolError("invalid worker frame magic")
        if header_size > MAX_HEADER_BYTES or payload_size > MAX_PAYLOAD_BYTES:
            raise WorkerProtocolError("worker frame exceeds negotiated limits")
        raw_header = _read_exact(stream, header_size)
        payload = _read_exact(stream, payload_size)
        if len(raw_header) != header_size or len(payload) != payload_size:
            raise WorkerProtocolError("truncated worker frame payload")
        try:
            header = json.loads(raw_header.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkerProtocolError("invalid worker frame JSON") from exc
        if not isinstance(header, dict) or not isinstance(header.get("type"), str):
            raise WorkerProtocolError("worker frame requires an object and type")
        return cls(header, payload)


def request_frame(request_id: str, command: str, *, model_id: str = "",
                  payload: bytes = b"", **fields: object) -> Frame:
    """Create a validated host-to-worker command frame."""
    if not isinstance(request_id, str) or not request_id.strip():
        raise WorkerProtocolError("request_id is required")
    if not isinstance(command, str) or not command.strip():
        raise WorkerProtocolError("worker command is required")
    header: dict[str, object] = {"type": command, "request_id": request_id}
    if model_id:
        header["model_id"] = model_id
    header.update(fields)
    return Frame(header, payload)


def response_request_id(frame: Frame) -> str:
    value = frame.header.get("request_id")
    if not isinstance(value, str) or not value:
        raise WorkerProtocolError("worker response is missing request_id")
    return value
