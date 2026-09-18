"""Minimal stdin/stdout worker host for model-pack Docker images.

Model packs provide handlers for the logical commands; this module owns the
wire framing, request IDs, error shape, and shutdown behavior.  It never logs
to stdout, so binary payloads cannot be corrupted by progress text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import sys
from typing import BinaryIO, Callable, Mapping

from .worker_protocol import Frame, WorkerProtocolError, response_request_id


COMMANDS = frozenset({"hello", "describe", "prepare", "train", "infer", "export", "cancel", "close"})
Handler = Callable[[Mapping[str, object], bytes], Frame | Mapping[str, object] | None]


class WorkerServerError(RuntimeError):
    pass


def _error_frame(request_id: str, code: str, message: str) -> Frame:
    return Frame({"type": "error", "request_id": request_id, "status": "error",
                  "code": code, "message": message})


@dataclass
class WorkerServer:
    handlers: Mapping[str, Handler] = field(default_factory=dict)
    protocol_version: int = 1
    _closed: bool = field(default=False, init=False)

    def dispatch(self, frame: Frame) -> Frame:
        request_id = response_request_id(frame)
        command = frame.header.get("type")
        if command not in COMMANDS:
            return _error_frame(request_id, "unsupported_command", f"unsupported command: {command}")
        if command == "close":
            self._closed = True
        if command == "hello" and "hello" not in self.handlers:
            return Frame({"type": "response", "request_id": request_id, "status": "ok",
                          "protocol": "DVW1", "protocol_version": self.protocol_version,
                          "commands": sorted(COMMANDS)})
        handler = self.handlers.get(command)
        if handler is None:
            if command in {"describe", "prepare", "train", "infer", "export"}:
                return _error_frame(
                    request_id,
                    "not_implemented",
                    f"worker does not implement {command}; install a model-pack handler",
                )
            return Frame({"type": "response", "request_id": request_id, "status": "ok",
                          "command": command})
        try:
            value = handler(frame.header, frame.payload)
        except Exception as exc:  # worker errors are data, never stdout tracebacks
            return _error_frame(request_id, "handler_error", str(exc))
        if value is None:
            return Frame({"type": "response", "request_id": request_id, "status": "ok",
                          "command": command})
        if isinstance(value, Frame):
            if value.header.get("request_id") != request_id:
                raise WorkerServerError("handler response request_id mismatch")
            return value
        if not isinstance(value, Mapping):
            raise WorkerServerError("worker handler must return a frame or mapping")
        header = {"type": "response", "request_id": request_id, "status": "ok", **value}
        return Frame(header)

    def serve(self, source: BinaryIO, sink: BinaryIO, *, error_stream: BinaryIO | None = None) -> int:
        """Serve until EOF or ``close``; return processed frame count."""
        count = 0
        while not self._closed:
            try:
                frame = Frame.read(source)
                if frame is None:
                    break
                response = self.dispatch(frame)
                sink.write(response.encode())
                sink.flush()
                count += 1
            except (WorkerProtocolError, WorkerServerError) as exc:
                if error_stream is not None:
                    error_stream.write((f"worker protocol error: {exc}\n").encode("utf-8"))
                    error_stream.flush()
                return count
        return count


def main() -> int:
    """A protocol-only worker useful as a Docker image healthcheck."""
    return WorkerServer().serve(sys.stdin.buffer, sys.stdout.buffer, error_stream=sys.stderr.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
