"""Shared worker protocol, bundled Windows launcher, and owned WSL contracts."""

from __future__ import annotations

from pathlib import Path
import struct
import sys
from io import BytesIO
from unittest.mock import patch

import pytest

from model_runtime.managed_wsl import ManagedWsl, ManagedWslError, WslCommandResult
from model_runtime.worker_manager import WorkerManager, WorkerManagerError
from model_runtime.worker_server import WorkerServer
from model_runtime.container_entrypoint import ContainerEntrypointError, _load_handlers
from model_runtime.windows_worker import WindowsWorker, WindowsWorkerError, build_worker_command
from model_runtime.worker_protocol import Frame, WorkerProtocolError, request_frame
from gui.core.container_worker import ContainerCommand, ContainerWorker


def test_shared_protocol_rejects_nan_and_roundtrips_partial_reads():
    with pytest.raises(WorkerProtocolError, match="JSON"):
        Frame({"type": "response", "value": float("nan")}).encode()

    frame = request_frame("r1", "hello", model_id="resnet18", payload=b"abc")
    raw = frame.encode()

    class Partial:
        def __init__(self, value: bytes):
            self.value = value

        def read(self, size: int) -> bytes:
            if not self.value:
                return b""
            chunk, self.value = self.value[:1], self.value[1:]
            return chunk

    assert Frame.read(Partial(raw)) == frame


def test_windows_worker_uses_absolute_shell_free_process_and_frames(tmp_path: Path):
    code = (
        "import sys,struct,json; p=sys.stdin.buffer.read(16); "
        "m,h,n=struct.unpack('<4sIQ',p); b=sys.stdin.buffer.read(h+n); "
        "header=json.loads(b[:h]); out=json.dumps({'type':'response','request_id':header['request_id']}).encode(); "
        "sys.stdout.buffer.write(struct.pack('<4sIQ',b'DVW1',len(out),0)+out); sys.stdout.buffer.flush()"
    )
    command = build_worker_command(sys.executable, runtime_id="builtin-cpu-v1", cwd=tmp_path,
                                   args=("-u", "-c", code), env={"PYTHONUNBUFFERED": "1"})
    assert command.command[0] == str(Path(sys.executable).resolve())
    assert command.command[1:] == ("-u", "-c", code)
    assert "PYTHONPATH" not in command.env

    worker = WindowsWorker(command)
    worker.start()
    try:
        worker.send(request_frame("r1", "hello"))
        assert worker.receive().header == {"type": "response", "request_id": "r1"}
    finally:
        worker.stop()
    assert worker.process is None


def test_windows_worker_rejects_external_python_overrides(tmp_path: Path):
    with pytest.raises(WindowsWorkerError, match="Python search"):
        build_worker_command(sys.executable, runtime_id="runtime", cwd=tmp_path,
                             env={"PYTHONPATH": "/tmp"})


def test_managed_wsl_imports_only_local_owned_distro(tmp_path: Path):
    tarball = tmp_path / "distro.tar"
    tarball.write_bytes(b"tar")
    install = tmp_path / "distro"
    install.mkdir()
    calls: list[tuple[str, ...]] = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        if argv[1:3] == ("--list", "--quiet"):
            return WslCommandResult(tuple(argv), 0, "", "")
        return WslCommandResult(tuple(argv), 0, "", "")

    manager = ManagedWsl("DeepVisionStudio", tmp_path, runner=fake_run)
    manager.import_offline(tarball, install, version="1.0.0")
    assert manager.is_owned()
    assert any(command[1] == "--import" for command in calls)
    result = manager.docker(("info",), check=False)
    assert result.argv[:4] == ("wsl.exe", "-d", "DeepVisionStudio", "--")


def test_managed_wsl_refuses_foreign_existing_distro(tmp_path: Path):
    tarball = tmp_path / "distro.tar"
    tarball.write_bytes(b"tar")
    install = tmp_path / "distro"
    install.mkdir()

    def fake_run(argv, **kwargs):
        if argv[1:3] == ("--list", "--quiet"):
            return WslCommandResult(tuple(argv), 0, "DeepVisionStudio\n", "")
        raise AssertionError("foreign distro must not be imported")

    manager = ManagedWsl("DeepVisionStudio", tmp_path, runner=fake_run)
    with pytest.raises(ManagedWslError, match="not owned"):
        manager.import_offline(tarball, install, version="1.0.0")


def test_worker_manager_owns_and_stops_container_lifecycle():
    class FakeWorker:
        def __init__(self):
            self.started = 0
            self.stopped = 0

        def start(self):
            self.started += 1

        def stop(self):
            self.stopped += 1

    manager = WorkerManager()
    worker = FakeWorker()
    manager.register_container("vendor.model", "container-v1", worker)
    assert worker.started == 1
    assert manager.get("vendor.model", "container-v1") is worker
    with pytest.raises(WorkerManagerError, match="already active"):
        manager.register_container("vendor.model", "container-v1", FakeWorker())
    manager.stop_all()
    assert worker.stopped == 1


def test_worker_server_keeps_stdout_framed_and_returns_handler_errors():
    def infer(header, payload):
        assert payload == b"image"
        return {"artifact": "result.bin"}

    source = BytesIO(b"".join([
        request_frame("hello-1", "hello").encode(),
        request_frame("infer-1", "infer", payload=b"image").encode(),
        request_frame("bad-1", "not-a-command").encode(),
        request_frame("close-1", "close").encode(),
    ]))
    sink = BytesIO()
    count = WorkerServer({"infer": infer}).serve(source, sink)
    assert count == 4
    responses = []
    sink.seek(0)
    while (frame := Frame.read(sink)) is not None:
        responses.append(frame)
    assert responses[0].header["commands"]
    assert responses[1].header["artifact"] == "result.bin"
    assert responses[2].header["code"] == "unsupported_command"
    assert responses[3].header["status"] == "ok"


def test_container_worker_sends_and_receives_shared_dvw1_frames():
    code = (
        "import sys\nfrom model_runtime.worker_protocol import Frame\n"
        "while True:\n"
        "  f=Frame.read(sys.stdin.buffer)\n"
        "  if f is None: break\n"
        "  out=Frame({'type':'response','request_id':f.header['request_id'],'status':'ok'}, f.payload)\n"
        "  sys.stdout.buffer.write(out.encode()); sys.stdout.buffer.flush()\n"
        "  if f.header['type']=='close': break\n"
    )
    command = ContainerCommand("fixture", "fixture", (sys.executable, "-u", "-c", code),
                               "deepvision.owner=fixture")
    worker = ContainerWorker(command)
    worker.start()
    try:
        response = worker.request(request_frame("container-1", "infer", payload=b"image"))
        assert response.payload == b"image"
        close = worker.request(request_frame("container-close", "close"))
        assert close.header["status"] == "ok"
    finally:
        with patch("gui.core.container_worker.subprocess.run"):
            worker.stop()
    assert worker.process is None


def test_container_entrypoint_loads_plugin_inside_pack_root_only(tmp_path: Path):
    (tmp_path / "plugin.py").write_text(
        "def factory(manifest):\n"
        "    return {'describe': lambda header, payload: {'model': manifest['model_id']}}\n",
        encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"model_id":"vendor.plugin","worker_entrypoint":"plugin:factory"}', encoding="utf-8")
    handlers = _load_handlers(manifest)
    assert handlers["describe"]({}, {})["model"] == "vendor.plugin"

    manifest.write_text('{"worker_entrypoint":"../escape:factory"}', encoding="utf-8")
    with pytest.raises(ContainerEntrypointError, match="unsafe"):
        _load_handlers(manifest)
