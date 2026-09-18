"""Installed Docker model-pack host adapter contracts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT)]

from core.model_pack_worker import ModelPackWorker, ModelPackWorkerError
from model_runtime.worker_protocol import Frame, response_request_id


def _manifest(root: Path, image: str = "registry.invalid/model@sha256:" + "a" * 64) -> None:
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "model_id": "extra_model", "pack_version": "1.0.0",
        "runtimes": ["container"], "container_image": image,
    }), encoding="utf-8")


class FakeProcess:
    def poll(self):
        return None


class FakeWorker:
    def __init__(self, command):
        self.command = command
        self.process = FakeProcess()
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def request(self, frame):
        assert self.started
        if frame.header["type"] == "hello":
            return Frame({"type": "response", "request_id": response_request_id(frame),
                          "status": "ok", "commands": ["train", "infer", "export"]},
                         b'{"ready":true}')
        return Frame({"type": "response", "request_id": response_request_id(frame),
                      "status": "ok", "command": frame.header["type"]},
                     b'{"accepted":true}')

    def stop(self):
        self.closed = True


def test_pack_requires_pinned_container_image(tmp_path):
    _manifest(tmp_path, "registry.invalid/model:latest")
    with pytest.raises(ModelPackWorkerError, match="pinned"):
        ModelPackWorker.from_installed_pack(tmp_path, data_dir=tmp_path, work_dir=tmp_path)


@pytest.mark.parametrize("image", [
    "registry.invalid/model@sha256:short",
    "registry.invalid/model@sha256:" + "g" * 64,
    "sha256:" + "a" * 63,
])
def test_pack_rejects_malformed_container_digest(tmp_path, image):
    _manifest(tmp_path, image)
    with pytest.raises(ModelPackWorkerError, match="64-character"):
        ModelPackWorker.from_installed_pack(tmp_path, data_dir=tmp_path, work_dir=tmp_path)


def test_pack_passes_offline_image_archive_to_container_command(tmp_path):
    _manifest(tmp_path)
    archive = tmp_path / "docker-image.tar"
    archive.write_bytes(b"oci archive")
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["container_image_archive"] = archive.name
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (patch("core.model_pack_worker.build_container_command", return_value=object()) as build,
          patch("core.model_pack_worker.ContainerWorker", FakeWorker)):
        ModelPackWorker.from_installed_pack(tmp_path, data_dir=tmp_path, work_dir=tmp_path)
    assert build.call_args.kwargs["image_archive"] == archive.resolve()


def test_pack_uses_configured_owned_wsl_docker_command(tmp_path):
    _manifest(tmp_path)
    prefix = ("wsl.exe", "-d", "DeepVisionStudio", "--", "docker")
    with (patch("core.model_pack_worker.configured_docker_command", return_value=prefix),
          patch("core.model_pack_worker.build_container_command", return_value=object()) as build,
          patch("core.model_pack_worker.ContainerWorker", FakeWorker)):
        ModelPackWorker.from_installed_pack(tmp_path, data_dir=tmp_path, work_dir=tmp_path)
    assert build.call_args.kwargs["docker_command"] == prefix


def test_pack_rejects_missing_offline_image_archive(tmp_path):
    _manifest(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["container_image_archive"] = "docker-image.tar"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelPackWorkerError, match="archive"):
        ModelPackWorker.from_installed_pack(tmp_path, data_dir=tmp_path, work_dir=tmp_path)


def test_pack_rejects_symlinked_manifest(tmp_path):
    external = tmp_path.parent / "external-manifest.json"
    external.write_text("{}", encoding="utf-8")
    (tmp_path / "manifest.json").symlink_to(external)
    with pytest.raises(ModelPackWorkerError, match="manifest"):
        ModelPackWorker.from_installed_pack(tmp_path, data_dir=tmp_path, work_dir=tmp_path)


def test_pack_train_infer_export_share_dvw1_adapter(tmp_path):
    _manifest(tmp_path)
    with (patch("core.model_pack_worker.build_container_command", return_value=object()) as build,
          patch("core.model_pack_worker.ContainerWorker", FakeWorker)):
        worker = ModelPackWorker.from_installed_pack(tmp_path, data_dir=tmp_path, work_dir=tmp_path)
    worker.start()
    hello = worker.json_request("hello")
    result = worker.json_request("train", {"dataset": "manifest.json"})
    assert hello["payload"] == {"ready": True}
    assert result["payload"] == {"accepted": True}
    assert build.call_args.kwargs["model_dir"] == tmp_path.resolve()
    worker.close()
    assert worker._worker.closed


def test_pack_rejects_unknown_command(tmp_path):
    _manifest(tmp_path)
    with patch("core.model_pack_worker.build_container_command", return_value=object()), \
         patch("core.model_pack_worker.ContainerWorker", FakeWorker):
        worker = ModelPackWorker.from_installed_pack(tmp_path, data_dir=tmp_path, work_dir=tmp_path)
    with pytest.raises(ModelPackWorkerError, match="unsupported"):
        worker.request("unknown")


def test_pack_job_runs_prepare_before_operation(tmp_path):
    from webapp.worker import model_pack_operation

    class Context:
        directory = tmp_path / "job"
        failure = ""

        def cancelled(self):
            return False

        def emit(self, event, args):
            events.append((event, args))

    class FakePack:
        model_id = "extra_model"
        last = None

        @classmethod
        def from_installed_pack(cls, *args, **kwargs):
            return cls()

        def __init__(self):
            self.commands = []
            type(self).last = self

        def start(self):
            self.commands.append("start")

        def json_request(self, command, value=None):
            self.commands.append((command, value))
            return {"command": command, "payload": {"ready": True}}

        def close(self):
            self.commands.append("close")

    events = []
    with patch("core.model_pack_worker.ModelPackWorker", FakePack):
        result = model_pack_operation(Context(), {
            "operation": "train", "pack_dir": str(tmp_path),
            "data_dir": str(tmp_path), "work_dir": str(tmp_path), "request": {"seed": 4},
        })
    assert result["status"] == "completed"
    assert [event for event, _ in events] == ["log_message", "model_pack_prepared", "model_pack_result"]
    assert [item[0] if isinstance(item, tuple) else item for item in FakePack.last.commands] == [
        "start", "hello", "prepare", "train", "close"]
