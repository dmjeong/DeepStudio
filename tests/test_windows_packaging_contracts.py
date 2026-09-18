"""Static contracts for the Windows-only WiX release entry point."""

from pathlib import Path
import xml.etree.ElementTree as ET
import json
import os
import runpy
import sys
import pytest


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "packaging" / "windows"


def test_wix_sources_are_well_formed_and_use_payload_contract():
    msi = (WINDOWS / "bootstrapper" / "DeepVisionStudio.msi.wxs").read_text(encoding="utf-8")
    bundle = (WINDOWS / "bootstrapper" / "DeepVisionStudio.bundle.wxs").read_text(encoding="utf-8")
    ET.fromstring(msi)
    ET.fromstring(bundle)
    assert 'Files Include="$(var.PayloadRoot)\\**"' in msi
    assert "MsiPackage SourceFile=\"$(var.MsiPath)\"" in bundle
    assert "Condition=\"VersionNT64\"" in bundle


def test_release_script_verifies_payload_before_wix_build():
    script = (WINDOWS / "build_release.ps1").read_text(encoding="utf-8")
    assert "collect_payloads.py" in script and "validate_payloads.py" in script
    assert "--manifest" in script
    assert "WixToolset.Bal.wixext" in script
    assert "THIRD_PARTY_NOTICES.md" in script
    assert "external_downloads" not in script


def test_pyinstaller_build_includes_model_pack_runtime_and_optional_native_sdk():
    script = (ROOT / "gui" / "build_exe.py").read_text(encoding="utf-8")
    assert "model_sdk', 'schemas" in script
    for module in ("model_runtime.container_entrypoint", "model_runtime.pack_installer",
                   "model_runtime.worker_protocol", "model_runtime.windows_worker"):
        assert f'"--hidden-import", "{module}"' in script
    assert "VISION_NATIVE_RUNTIME_DIR" in script
    assert "--add-binary" in script


def test_pyinstaller_native_runtime_argument_is_opt_in_and_filters_library_files(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    suffix = ".dll" if sys.platform == "win32" else ".dylib"
    (runtime / f"vision_runtime{suffix}").write_bytes(b"dll")
    (runtime / f"onnxruntime{suffix}").write_bytes(b"ort")
    (runtime / "README.txt").write_text("ignored", encoding="utf-8")
    build = runpy.run_path(str(ROOT / "gui" / "build_exe.py"))
    monkeypatch.setenv("VISION_NATIVE_RUNTIME_DIR", str(runtime))
    args = build["_native_runtime_arguments"]()
    assert args == ["--add-binary", f"{runtime / f'onnxruntime{suffix}'}{os.pathsep}.",
                    "--add-binary", f"{runtime / f'vision_runtime{suffix}'}{os.pathsep}."]


def test_payload_wrapper_scripts_collect_and_verify_exact_bytes(tmp_path):
    root = tmp_path / "payload"
    root.mkdir()
    (root / "THIRD_PARTY_NOTICES.md").write_text("notice", encoding="utf-8")
    (root / "worker.exe").write_bytes(b"worker")
    manifest_path = tmp_path / "release-manifest.json"
    collect = runpy.run_path(str(WINDOWS / "collect_payloads.py"))
    manifest = collect["collect_payload"](root, version="1.0.0")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    verify = runpy.run_path(str(WINDOWS / "validate_payloads.py"))
    verify["verify_payload"](root, json.loads(manifest_path.read_text(encoding="utf-8")))


def test_payload_stager_copies_artifacts_and_rejects_symlinks(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "worker.exe").write_bytes(b"worker")
    notice = tmp_path / "NOTICE.md"
    notice.write_text("notice", encoding="utf-8")
    stager = runpy.run_path(str(WINDOWS / "stage_payload.py"))
    output = stager["stage_payload"](tmp_path / "payload", [("workers", source)], notice=notice)
    assert (output / "workers/worker.exe").read_bytes() == b"worker"
    assert (output / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8") == "notice"

    outside = tmp_path / "outside"
    outside.write_text("private", encoding="utf-8")
    (source / "escape").symlink_to(outside)
    with pytest.raises(stager["PayloadStageError"], match="symlink"):
        stager["stage_payload"](tmp_path / "bad", [("workers", source)], notice=notice)
