"""Static contracts for the Windows-only WiX release entry point."""

from pathlib import Path
import xml.etree.ElementTree as ET
import base64
import hashlib
import json
import os
import runpy
import sys
import zipfile
import pytest

from core.model_registry import builtin_model_specs


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
    assert "WixQuietExec" in msi
    assert 'SetProperty Id="RunDeepVisionWslBootstrap"' in msi
    assert 'SetProperty Id="WixQuietExecCmdLine"' not in msi
    assert "Wix4UtilCA_$(sys.BUILDARCHSHORT)" in msi
    assert "bootstrap_wsl.ps1" in msi
    assert "NOT REMOVE" in msi
    assert "WslMsiPath" in bundle
    assert 'Permanent="yes"' in bundle


def test_release_script_verifies_payload_before_wix_build():
    script = (WINDOWS / "build_release.ps1").read_text(encoding="utf-8")
    assert "collect_payloads.py" in script and "validate_payloads.py" in script
    assert "--manifest" in script
    assert "WixToolset.Bal.wixext" in script
    assert "THIRD_PARTY_NOTICES.md" in script
    assert "external_downloads" not in script
    assert "WixToolset.Util.wixext" in script
    assert '"-d", "Version=$Version"' in script
    assert '"-d", "PayloadRoot=$root"' in script
    assert '"-d", "RequireOfflineWsl=$requireWslValue"' in script
    assert '$wix.Source --version' in script
    assert '$WixVersion -notmatch' in script
    assert 'WiX version mismatch' in script
    assert '[string] $WixVersion = "7.0.0"' in script


def test_third_party_notice_names_optional_model_sources():
    notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for component in ("LibreYOLO", "RT-DETRv4", "SAM2 1.0", "ONNX Runtime"):
        assert component in notice
    assert "exact model card" in notice.lower()


def test_pyinstaller_build_includes_model_pack_runtime_and_optional_native_sdk():
    script = (ROOT / "gui" / "build_exe.py").read_text(encoding="utf-8")
    assert "model_sdk', 'schemas" in script
    for module in ("core.model_pack_worker", "model_runtime.container_entrypoint", "model_runtime.deployment_bundle", "model_runtime.pack_installer",
                   "model_runtime.worker_protocol", "model_runtime.windows_worker"):
        assert f'"--hidden-import", "{module}"' in script
    assert "VISION_NATIVE_RUNTIME_DIR" in script
    assert "--add-binary" in script
    assert "packaging" in script and "windows" in script and "models" in script
    assert "check_runtime_dependencies" in script
    assert '"--collect-all", "PySide6"' in script
    assert '"--collect-all", "shiboken6"' in script
    assert '"--collect-all", "libreyolo"' in script
    assert '"--collect-all", "sam2"' in script
    assert '"--hidden-import", "huggingface_hub"' in script
    assert '"--add-data", f"{BUILTIN_ASSETS_DIR}{os.pathsep}builtin_assets"' in script


def test_windows_builder_installs_the_gui_runtime_with_the_build_interpreter():
    script = (ROOT / "gui" / "build.bat").read_text(encoding="utf-8")
    assert "prepare_torch_runtime.py --accelerator %DVS_ACCELERATOR%" in script
    assert 'set "DVS_ACCELERATOR=auto"' in script
    assert "python -m pip install -r requirements.txt" in script
    assert "SAM2_BUILD_CUDA=0" in script
    assert "prepare_builtin_assets.py --output builtin_assets" in script
    assert "pip show pyinstaller" not in script


def test_windows_installer_contract_freezes_cuda_training_runtime():
    workflow = (ROOT / ".github/workflows/windows-native-sdk.yml").read_text(encoding="utf-8")
    assert "torch==2.14.0+cu130" in workflow
    assert "torchvision==0.29.0+cu130" in workflow
    assert "torch_cuda.dll" in workflow and "c10_cuda.dll" in workflow


def test_simple_installer_only_stages_the_app_and_public_examples(tmp_path):
    simple = runpy.run_path(str(WINDOWS / "stage_simple_payload.py"))
    app = tmp_path / "app"
    examples = tmp_path / "example"
    cpp = tmp_path / "cpp"
    csharp = tmp_path / "csharp"
    for directory in (app, examples / "assets", examples / "cpp", examples / "csharp",
                      cpp / "include", cpp / "src", csharp):
        directory.mkdir(parents=True, exist_ok=True)
    (app / "DeepVisionStudio.exe").write_bytes(b"app")
    (examples / "README.md").write_text("example", encoding="utf-8")
    (examples / "assets" / "test.onnx").write_bytes(b"onnx")
    (examples / "cpp" / "CMakeLists.txt").write_text("cmake", encoding="utf-8")
    (examples / "cpp" / "main.cpp").write_text("int main() {}", encoding="utf-8")
    for name in ("classifier.h", "example_paths.h", "self_test.cpp"):
        (examples / "cpp" / name).write_text("example code", encoding="utf-8")
    for name in ("with_opencv/CMakeLists.txt", "with_evision/CMakeLists.txt",
                 "with_evision/classifier.h", "with_evision/bw8_preprocess.h"):
        path = examples / "cpp" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("example code", encoding="utf-8")
    (examples / "csharp" / "OnnxExample.csproj").write_text("project", encoding="utf-8")
    (examples / "csharp" / "Program.cs").write_text("code", encoding="utf-8")
    (cpp / "CMakeLists.txt").write_text("runtime", encoding="utf-8")
    (cpp / "include" / "vision_inference.h").write_text("header", encoding="utf-8")
    (cpp / "src" / "vision_inference.cpp").write_text("source", encoding="utf-8")
    (csharp / "VisionRuntime.csproj").write_text("runtime", encoding="utf-8")
    (csharp / "VisionRuntime.cs").write_text("source", encoding="utf-8")
    notice = tmp_path / "THIRD_PARTY_NOTICES.md"
    notice.write_text("notice", encoding="utf-8")

    payload = simple["stage_simple_payload"](
        tmp_path / "payload", app=app, example_root=examples, cpp_runtime_root=cpp,
        csharp_runtime_root=csharp, notice=notice,
    )
    assert {item.name for item in payload.iterdir()} == {"app", "Examples", "THIRD_PARTY_NOTICES.md"}
    assert (payload / "Examples/cpp/vision-runtime/src/vision_inference.cpp").is_file()
    assert (payload / "Examples/cpp/with_evision/bw8_preprocess.h").is_file()
    assert (payload / "Examples/cpp/with_opencv/CMakeLists.txt").is_file()
    for name in ("classifier.h", "example_paths.h", "self_test.cpp"):
        assert (payload / "Examples/cpp" / name).read_text(encoding="utf-8") == "example code"
    assert (payload / "Examples/csharp/vision-runtime/VisionRuntime.cs").is_file()
    simple["validate_simple_payload"](payload)
    (payload / "models").mkdir()
    with pytest.raises(simple["PayloadStageError"], match="top level"):
        simple["validate_simple_payload"](payload)


def test_simple_installer_build_path_is_explicit_and_skips_model_catalog():
    release = (WINDOWS / "build_release.ps1").read_text(encoding="utf-8")
    simple = (WINDOWS / "build_simple_installer.ps1").read_text(encoding="utf-8")
    batch = (ROOT / "gui" / "build.bat").read_text(encoding="utf-8")
    validator = (WINDOWS / "validate_payloads.py").read_text(encoding="utf-8")
    assert "[switch] $Simple" in release
    assert "stage_simple_payload.py" in release
    assert "--skip-model-catalog" in release
    assert "stage_simple_payload.py" in simple
    assert "-Simple" in simple
    assert '"installer"' in batch
    assert "build_simple_installer.ps1" in batch
    assert "--skip-model-catalog" in validator


def test_csharp_sdk_exposes_directory_bundle_open():
    source = (ROOT / "sdk" / "csharp" / "VisionRuntime.cs").read_text(encoding="utf-8")
    assert "OpenBundle" in source
    assert "dv_create_session_from_bundle" in source
    sample = ROOT / "sdk" / "csharp" / "sample"
    project = (sample / "VisionRuntime.Sample.csproj").read_text(encoding="utf-8")
    program = (sample / "Program.cs").read_text(encoding="utf-8")
    assert '<RuntimeIdentifier>win-x64</RuntimeIdentifier>' in project
    assert '<SelfContained>true</SelfContained>' in project
    assert "VisionSession.OpenBundle" in program
    assert "DangerousGetHandle" not in source
    assert "dv_infer(VisionSession session" in source
    assert "dv_sam_segment(VisionSession session, SamImageContext context" in source


def test_offline_default_model_catalog_matches_registry():
    catalog = json.loads((WINDOWS / "models" / "default-model-catalog.json").read_text(encoding="utf-8"))
    assert catalog["schema_version"] == 1
    assert catalog["offline"] is True
    assert catalog["redistribution_policy"]["weights_included"] is True
    expected = {spec.model_id for spec in builtin_model_specs()}
    actual = {item["model_id"] for item in catalog["models"]}
    assert actual == expected


def test_offline_default_model_catalog_is_deterministically_generated():
    exporter = runpy.run_path(str(ROOT / "tools" / "export_model_catalog.py"))
    packaged = json.loads((WINDOWS / "models" / "default-model-catalog.json").read_text(encoding="utf-8"))
    assert packaged == exporter["build_catalog"]()


def test_model_catalog_payload_gate_is_optional_for_development_and_strict_for_release(tmp_path):
    api = runpy.run_path(str(WINDOWS / "model_catalog_payload.py"))
    root = tmp_path / "payload"
    (root / "models").mkdir(parents=True)
    (root / "app").mkdir()
    (root / "app" / "DeepVisionStudio.exe").write_bytes(b"gui")
    catalog = {
        "schema_version": 1,
        "catalog_id": "test",
        "offline": True,
        "release_ready_only": False,
        "redistribution_policy": {
            "weights_included": True,
            "require_third_party_notices": True,
            "require_license_files_for_release_packs": True,
        },
        "models": [{"model_id": "demo", "release_status": "requested"}],
    }
    catalog_path = root / "models" / "default-model-catalog.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    api["validate_model_catalog_payload"](root)
    with pytest.raises(api["ModelCatalogPayloadError"], match="not release_ready"):
        api["validate_model_catalog_payload"](root, require_release_ready=True)

    catalog["release_ready_only"] = True
    catalog["models"] = [{
        "model_id": "demo",
        "release_status": "release_ready",
        "metadata": {"payload": {"kind": "builtin", "paths": ["app/DeepVisionStudio.exe"]}},
    }]
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    api["validate_model_catalog_payload"](
        root, require_release_ready=True, required_model_ids={"demo"}
    )

    catalog["models"][0]["metadata"]["payload"] = {
        "kind": "pack", "paths": ["models/demo.bin"]
    }
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    (root / "models" / "demo.bin").write_bytes(b"pack")
    with pytest.raises(api["ModelCatalogPayloadError"], match=".dvmodel"):
        api["validate_model_catalog_payload"](
            root, require_release_ready=True, required_model_ids={"demo"}
        )

    catalog["models"][0]["metadata"]["payload"] = {
        "kind": "pack", "paths": ["models/demo.dvmodel"]
    }
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from model_runtime.pack_signing import signature_payload
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    manifest = {
            "schema_version": 1,
            "model_id": "demo",
            "pack_version": "1.0.0",
            "release_status": "release_ready",
            "license": {"spdx": "MIT", "source": "https://example.invalid/model", "revision": "v1"},
        }
    pack_files = {
        "THIRD_PARTY_NOTICES.md": b"notice",
        "licenses/model.txt": b"license",
    }
    signed_hashes = {name: hashlib.sha256(content).hexdigest()
                     for name, content in pack_files.items()}
    manifest["signature"] = {
        "algorithm": "ed25519", "key_id": "release-key",
        "value": base64.b64encode(
            private_key.sign(signature_payload(manifest, signed_hashes))
        ).decode("ascii"),
    }
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    checksums = {"manifest.json": hashlib.sha256(manifest_bytes).hexdigest(), **signed_hashes}
    with zipfile.ZipFile(root / "models" / "demo.dvmodel", "w") as archive:
        archive.writestr("manifest.json", manifest_bytes)
        archive.writestr("checksums.json", json.dumps({"files": checksums}))
        for name, content in pack_files.items():
            archive.writestr(name, content)
    api["validate_model_catalog_payload"](
        root, require_release_ready=True, required_model_ids={"demo"},
        trusted_keys={"release-key": public_key},
    )


def test_release_catalog_requires_every_builtin_model(tmp_path):
    api = runpy.run_path(str(WINDOWS / "model_catalog_payload.py"))
    root = tmp_path / "payload"
    (root / "models").mkdir(parents=True)
    (root / "app").mkdir()
    (root / "app" / "DeepVisionStudio.exe").write_bytes(b"gui")
    catalog = {
        "schema_version": 1, "offline": True, "release_ready_only": True,
        "redistribution_policy": {
            "weights_included": True,
            "require_third_party_notices": True,
            "require_license_files_for_release_packs": True,
        },
        "models": [{
            "model_id": "efficientnet_b0", "release_status": "release_ready",
            "payload": {"kind": "builtin", "paths": ["app/DeepVisionStudio.exe"]},
        }],
    }
    path = root / "models" / "default-model-catalog.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(api["ModelCatalogPayloadError"], match="set mismatch"):
        api["validate_model_catalog_payload"](root, require_release_ready=True)


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
    with pytest.raises(ValueError, match="required payload"):
        verify["verify_payload"](root, json.loads(manifest_path.read_text(encoding="utf-8")),
                                  required_paths=["models/default-model-catalog.json"])


def test_offline_wsl_inventory_binds_artifacts_and_notices(tmp_path):
    root = tmp_path / "payload"
    wsl = root / "runtime" / "wsl"
    licenses = wsl / "licenses"
    licenses.mkdir(parents=True)
    (root / "THIRD_PARTY_NOTICES.md").write_text("notice", encoding="utf-8")
    (wsl / "wsl-offline.msi").write_bytes(b"wsl-msi")
    (wsl / "owned-distro.tar").write_bytes(b"distro")
    for name in ("wsl.txt", "docker.txt", "distro.txt"):
        (licenses / name).write_text(name, encoding="utf-8")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    inventory = {
        "schema_version": 1,
        "platform": "windows-x64",
        "components": [
            {"id": "wsl", "artifact": "runtime/wsl/wsl-offline.msi",
             "sha256": digest(wsl / "wsl-offline.msi"), "license": "Microsoft",
             "notice": "runtime/wsl/licenses/wsl.txt"},
            {"id": "owned_distro", "artifact": "runtime/wsl/owned-distro.tar",
             "sha256": digest(wsl / "owned-distro.tar"), "license": "distro-license",
             "notice": "runtime/wsl/licenses/distro.txt"},
            {"id": "docker_engine", "artifact": "runtime/wsl/owned-distro.tar",
             "sha256": digest(wsl / "owned-distro.tar"), "license": "Apache-2.0",
             "notice": "runtime/wsl/licenses/docker.txt"},
        ],
    }
    (licenses / "manifest.json").write_text(json.dumps(inventory), encoding="utf-8")
    collector = runpy.run_path(str(WINDOWS / "collect_payloads.py"))
    manifest = collector["collect_payload"](root, version="1.0.0")
    verifier = runpy.run_path(str(WINDOWS / "validate_payloads.py"))
    verifier["verify_payload"](root, manifest)
    verifier["verify_offline_wsl_payload"](root, manifest)
    (wsl / "owned-distro.tar").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        verifier["verify_offline_wsl_payload"](root, manifest)
    root_link = tmp_path / "payload-link"
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="root symlink"):
        verifier["verify_offline_wsl_payload"](root_link, manifest)


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
    source_link = tmp_path / "source-link"
    source_link.symlink_to(source, target_is_directory=True)
    with pytest.raises(stager["PayloadStageError"], match="symlink"):
        stager["stage_payload"](tmp_path / "bad-root-link", [("workers", source_link)], notice=notice)


def test_payload_paths_reject_drive_and_duplicate_separator_forms(tmp_path):
    stager = runpy.run_path(str(WINDOWS / "stage_payload.py"))
    source = tmp_path / "source"
    source.mkdir()
    (source / "worker.exe").write_bytes(b"worker")
    with pytest.raises(stager["PayloadStageError"], match="unsafe"):
        stager["stage_payload"](tmp_path / "bad-drive", [("C:/workers", source)])
    with pytest.raises(stager["PayloadStageError"], match="unsafe"):
        stager["stage_payload"](tmp_path / "bad-separator", [("workers//nested", source)])

    manifest_api = runpy.run_path(str(WINDOWS / "payload_manifest.py"))
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "worker.exe").write_bytes(b"worker")
    manifest = manifest_api["collect_payload"](payload, version="1.0.0")
    manifest["files"][0]["path"] = "C:/worker.exe"
    with pytest.raises(ValueError, match="unsafe"):
        manifest_api["verify_payload"](payload, manifest)


def test_release_script_requires_app_catalog_and_csharp_sdk_payload_files():
    script = (WINDOWS / "build_release.ps1").read_text(encoding="utf-8")
    assert 'app\\DeepVisionStudio.exe' in script
    assert 'models\\default-model-catalog.json' in script
    assert 'sdk\\VisionRuntime.dll' in script
    assert 'sdk\\native\\vision_runtime.dll' in script
    assert '[switch] $RequireReleaseReadyModels' in script
    assert '--require-release-ready-models' in script
    assert '[string] $ModelPackTrustStore' in script
    assert '--model-pack-trust-store $modelTrustStore' in script


def test_release_script_has_optional_authenticode_sign_and_verify_gate():
    script = (WINDOWS / "build_release.ps1").read_text(encoding="utf-8")
    assert '[switch] $RequireSignature' in script
    assert '[string] $CertificatePath' in script
    assert 'signtool.exe' in script
    assert 'Sign-AndVerify $msi' in script
    assert 'Sign-AndVerify $setup' in script
    assert 'verify /pa /all' in script
    assert 'SHA256' in script


def test_release_script_removes_partial_installer_artifacts_on_failure():
    script = (WINDOWS / "build_release.ps1").read_text(encoding="utf-8")
    assert "$releaseSucceeded = $false" in script
    assert "trap" in script
    assert "Never let a" in script
    assert "Remove-Item -LiteralPath $artifact -Force" in script
    assert "$releaseSucceeded = $true" in script


def test_release_script_fails_closed_when_python_payload_contracts_fail():
    script = (WINDOWS / "build_release.ps1").read_text(encoding="utf-8")
    assert 'throw "Payload manifest collection failed."' in script
    assert 'throw "Payload contract validation failed."' in script
    assert script.index('throw "Payload manifest collection failed."') > script.index('python $collector')
    assert script.index('throw "Payload contract validation failed."') > script.index('python $validator')


def test_release_script_has_production_offline_wsl_payload_gate():
    script = (WINDOWS / "build_release.ps1").read_text(encoding="utf-8")
    assert '[switch] $RequireOfflineWsl' in script
    assert 'runtime\\wsl\\wsl-offline.msi' in script
    assert 'runtime\\wsl\\owned-distro.tar' in script
    assert 'runtime\\wsl\\licenses\\manifest.json' in script
    assert 'runtime\\wsl\\bootstrap_wsl.ps1' in script
    workflow = (ROOT / ".github" / "workflows" / "windows-native-sdk.yml").read_text(encoding="utf-8")
    assert '-RequireOfflineWsl' in workflow
    assert 'DEEPVISION_WSL_PAYLOAD_ROOT' in workflow
    assert 'bootstrap_wsl.ps1' in workflow


def test_windows_workflow_keeps_release_ready_models_as_an_explicit_gate():
    workflow = (ROOT / ".github" / "workflows" / "windows-native-sdk.yml").read_text(encoding="utf-8")
    assert "require_release_ready_models" in workflow
    assert 'type: boolean' in workflow
    assert 'DEEPVISION_MODEL_PAYLOAD_ROOT' in workflow
    assert '$requireModels = $env:REQUIRE_RELEASE_READY_MODELS -eq "true"' in workflow
    assert '$releaseArgs += @("-RequireReleaseReadyModels", "-ModelPackTrustStore", $env:DEEPVISION_MODEL_PACK_TRUST_STORE)' in workflow
    assert 'DEEPVISION_MODEL_PACK_TRUST_STORE must point to an Ed25519 public-key trust store' in workflow


def test_offline_wsl_bootstrap_is_shell_free_and_never_downloads():
    script = (WINDOWS / "wsl" / "bootstrap_wsl.ps1").read_text(encoding="utf-8")
    assert '"--import"' in script and '"--version"' in script
    assert '"--web-download"' not in script
    assert '"--" "docker" "info"' in script
    assert 'owned-distro.json' in script
    assert 'Start-Process -FilePath "msiexec.exe"' in script
    assert script.index('Start-Process -FilePath "msiexec.exe"') < script.index('Get-Command "wsl.exe"')


def test_offline_wsl_bootstrap_rolls_back_only_a_failed_new_import():
    script = (WINDOWS / "wsl" / "bootstrap_wsl.ps1").read_text(encoding="utf-8")
    assert "$importAttempted = $false" in script
    assert "$bootstrapCommitted = $false" in script
    assert '"--unregister"' in script
    assert 'Remove-Item -LiteralPath $installDir -Recurse -Force' in script
    assert 'if (-not $bootstrapCommitted -and $importAttempted' in script
    assert script.index('"--unregister"') > script.index('} catch {')
    assert script.index('$bootstrapCommitted = $true') > script.index('"--" "docker" "info"')


def test_windows_workflow_builds_real_gui_and_native_runtime():
    workflow = (ROOT / ".github" / "workflows" / "windows-native-sdk.yml").read_text(encoding="utf-8")
    assert '"cpp/**"' in workflow
    assert '"gui/widgets/**"' in workflow
    assert "vcpkg.exe" in workflow
    assert 'version = "1.29.0"' in workflow
    assert "onnxruntime-win-x64-$version.zip" in workflow
    assert '$version = "1.29.0"' in workflow
    assert "python gui/build_exe.py" in workflow
    assert '"--source", "app=$app"' in workflow
    assert '"--source", "sdk/native=$native"' in workflow
    assert '"python/export_onnx.py"' in workflow
    assert '"python/export_sam2_onnx.py"' in workflow
    assert '"tests/test_export_contracts.py"' in workflow
    assert "release-contract" not in workflow


def test_ci_cpp_runtime_pin_matches_measured_onnx_runtime():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "onnxruntime-linux-x64-1.29.0.tgz" in workflow
    assert "onnxruntime-linux-x64-1.29.0\"" in workflow


def test_python_requirements_pin_the_same_onnx_runtime_release():
    for relative in ("gui/requirements.txt", "python/requirements.txt",
                     "python/requirements-efficientnet-cpu.txt", "webapp/requirements.txt"):
        requirements = (ROOT / relative).read_text(encoding="utf-8")
        assert "onnxruntime==1.29.0" in requirements, relative


def test_windows_workflow_keeps_offline_wsl_as_an_explicit_release_gate():
    workflow = (ROOT / ".github" / "workflows" / "windows-native-sdk.yml").read_text(encoding="utf-8")
    assert "require_offline_wsl" in workflow
    assert 'type: boolean' in workflow
    assert '$requireWsl = $env:REQUIRE_OFFLINE_WSL -eq "true"' in workflow
    assert 'if ($requireWsl)' in workflow
    assert 'if ($env:REQUIRE_OFFLINE_WSL -eq "true") { $releaseArgs += "-RequireOfflineWsl" }' in workflow
    assert 'packaging/model-pack-template/**' in workflow
    assert 'tests/test_model_pack_template.py' in workflow


def test_real_payload_layout_contains_managed_and_native_sdk_runtime(tmp_path):
    stager = runpy.run_path(str(WINDOWS / "stage_payload.py"))
    collector = runpy.run_path(str(WINDOWS / "collect_payloads.py"))
    verifier = runpy.run_path(str(WINDOWS / "validate_payloads.py"))
    app = tmp_path / "app"
    sdk = tmp_path / "sdk"
    native = tmp_path / "native"
    models = tmp_path / "models"
    for directory in (app, sdk, native, models):
        directory.mkdir()
    (app / "DeepVisionStudio.exe").write_bytes(b"frozen-gui")
    (sdk / "VisionRuntime.dll").write_bytes(b"managed-sdk")
    (native / "vision_runtime.dll").write_bytes(b"native-sdk")
    (models / "default-model-catalog.json").write_text("{}", encoding="utf-8")
    notice = tmp_path / "THIRD_PARTY_NOTICES.md"
    notice.write_text("notice", encoding="utf-8")
    payload = stager["stage_payload"](
        tmp_path / "payload",
        [("app", app), ("sdk", sdk), ("sdk/native", native), ("models", models)],
        notice=notice,
    )
    manifest = collector["collect_payload"](payload, version="1.0.0")
    verifier["verify_payload"](
        payload, manifest,
        required_paths=["app/DeepVisionStudio.exe", "sdk/VisionRuntime.dll",
                        "sdk/native/vision_runtime.dll", "models/default-model-catalog.json"],
    )
