"""The checked-in Docker model-pack template is buildable and speaks DVW1."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from model_runtime.pack_builder import build_pack
from model_runtime.worker_protocol import Frame, request_frame


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "packaging" / "model-pack-template"


def _read_frame(process: subprocess.Popen[bytes]) -> Frame:
    assert process.stdout is not None
    frame = Frame.read(process.stdout)
    assert frame is not None
    return frame


def test_template_manifest_is_digest_pinned_and_pack_builds(tmp_path):
    manifest = json.loads((TEMPLATE / "manifest.json").read_text(encoding="utf-8"))
    # The checked-in template runs its independent DVW1 worker directly from
    # Dockerfile.  worker_entrypoint is optional for packs that use the host's
    # generic container_entrypoint inside their own image.
    assert "worker_entrypoint" not in manifest
    assert "@sha256:" in manifest["container_image"]
    output = build_pack(TEMPLATE, tmp_path / "template.dvmodel", allow_unsigned=True)
    assert output.is_file()


def test_template_worker_round_trips_shared_dvw1_protocol():
    process = subprocess.Popen([sys.executable, str(TEMPLATE / "worker.py")],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in (
            request_frame("hello", "hello"),
            request_frame("describe", "describe"),
            request_frame("infer", "infer", payload=b"image"),
            request_frame("close", "close"),
        ):
            process.stdin.write(frame.encode())
            process.stdin.flush()
            response = _read_frame(process)
            assert response.header["request_id"] == frame.header["request_id"]
            assert response.header["status"] == "ok"
        assert process.wait(timeout=3) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
