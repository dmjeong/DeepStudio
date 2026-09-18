"""Offline payload hash/size/budget validation."""

import json
import sys
from pathlib import Path

import pytest

# The Windows packaging helper is intentionally kept outside the application
# package so it can be copied into a release build without Python imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packaging" / "windows"))
from payload_manifest import collect_payload, verify_payload


def test_payload_manifest_roundtrip_and_tamper_detection(tmp_path):
    root = tmp_path / "payload"
    root.mkdir()
    (root / "app.txt").write_text("app", encoding="utf-8")
    (root / "nested").mkdir()
    (root / "nested/model.onnx").write_bytes(b"onnx")
    manifest = collect_payload(root, version="1.0.0", commit="abc", budget_bytes=1024)
    verify_payload(root, manifest)
    (root / "app.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        verify_payload(root, manifest)


def test_payload_manifest_rejects_budget_and_symlink(tmp_path):
    root = tmp_path / "payload"
    root.mkdir()
    (root / "large.bin").write_bytes(b"12345")
    with pytest.raises(ValueError, match="budget"):
        collect_payload(root, version="dev", budget_bytes=4)
    target = tmp_path / "target.txt"
    target.write_text("x", encoding="utf-8")
    (root / "link").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        collect_payload(root, version="dev", budget_bytes=1024)
    root_link = tmp_path / "root-link"
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="root symlink"):
        collect_payload(root_link, version="dev", budget_bytes=1024)
