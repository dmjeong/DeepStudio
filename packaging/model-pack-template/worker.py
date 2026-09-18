"""Small, dependency-free DVW1 model-pack worker template.

Replace the command handlers with the model implementation and keep stdout
reserved for framed responses.  The template deliberately does not import
Deep Vision Studio or a host-side Python package, so the Docker image owns all
model dependencies.
"""

from __future__ import annotations

import json
import struct
import sys
from typing import Any, Mapping


MAGIC = b"DVW1"
HEADER = struct.Struct("<4sIQ")
MAX_HEADER_BYTES = 64 * 1024
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
COMMANDS = {"hello", "describe", "prepare", "train", "infer", "export", "cancel", "close"}


def _read_exact(size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        block = sys.stdin.buffer.read(size - len(result))
        if not block:
            return bytes(result)
        result.extend(block)
    return bytes(result)


def _frame(header: Mapping[str, Any], payload: bytes = b"") -> bytes:
    encoded = json.dumps(dict(header), ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_HEADER_BYTES or len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("DVW1 response exceeds the negotiated limits")
    return HEADER.pack(MAGIC, len(encoded), len(payload)) + encoded + payload


def _response(request_id: str, command: str, **fields: Any) -> bytes:
    return _frame({"type": "response", "request_id": request_id, "status": "ok",
                   "command": command, **fields})


def _error(request_id: str, code: str, message: str) -> bytes:
    return _frame({"type": "error", "request_id": request_id, "status": "error",
                   "code": code, "message": message})


def _dispatch(header: Mapping[str, Any], payload: bytes) -> bytes:
    request_id = header.get("request_id")
    command = header.get("type")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id is required")
    if command == "hello":
        return _response(request_id, "hello", protocol="DVW1", protocol_version=1,
                         commands=sorted(COMMANDS))
    if command not in COMMANDS:
        return _error(request_id, "unsupported_command", f"unsupported command: {command}")
    if command == "describe":
        return _response(request_id, "describe", model_id="vendor.example_classifier",
                         task="classify", input_size=[224, 224], input_channels=[3])
    if command == "prepare":
        return _response(request_id, "prepare", ready=True)
    if command in {"train", "infer", "export"}:
        # This is intentionally a safe placeholder.  A real pack should copy
        # artifacts under /work and return paths relative to that mount.
        return _response(request_id, command, accepted=True, payload_bytes=len(payload))
    if command == "cancel":
        return _response(request_id, "cancel", cancelled=True)
    return _response(request_id, command)


def serve() -> int:
    while True:
        prefix = _read_exact(HEADER.size)
        if not prefix:
            return 0
        if len(prefix) != HEADER.size:
            print("truncated DVW1 header", file=sys.stderr)
            return 2
        magic, header_size, payload_size = HEADER.unpack(prefix)
        if magic != MAGIC or header_size > MAX_HEADER_BYTES or payload_size > MAX_PAYLOAD_BYTES:
            print("invalid DVW1 frame", file=sys.stderr)
            return 2
        raw_header = _read_exact(header_size)
        payload = _read_exact(payload_size)
        if len(raw_header) != header_size or len(payload) != payload_size:
            print("truncated DVW1 frame", file=sys.stderr)
            return 2
        try:
            header = json.loads(raw_header.decode("utf-8"))
            if not isinstance(header, dict):
                raise ValueError("header must be an object")
            output = _dispatch(header, payload)
        except Exception as exc:  # Keep protocol errors on the wire, never stdout text.
            request_id = header.get("request_id", "") if isinstance(header, dict) else ""
            output = _error(request_id if isinstance(request_id, str) else "", "handler_error", str(exc))
        sys.stdout.buffer.write(output)
        sys.stdout.buffer.flush()
        if isinstance(header, dict) and header.get("type") == "close":
            return 0


def factory(manifest: Mapping[str, Any]):
    """Document the plugin shape expected by container_entrypoint."""
    if manifest.get("model_id") != "vendor.example_classifier":
        raise ValueError("template worker received an unexpected model_id")
    return {}


if __name__ == "__main__":
    raise SystemExit(serve())
