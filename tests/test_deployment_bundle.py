"""Portable ONNX deployment bundle integrity contracts."""

import json
from pathlib import Path
import stat
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_runtime.deployment_bundle import (  # noqa: E402
    DeploymentBundleError,
    build_deployment_bundle,
    verify_deployment_bundle,
)


def _config(model_name="model.onnx"):
    return {
        "schema_version": 5, "backend": "builtin", "task": "classify",
        "model_path": model_name, "input_name": "input_image",
        "output_name": "class_logits", "output_names": ["class_logits"],
        "input_channels": 3, "input_height": 32, "input_width": 32,
        "num_classes": 2, "cpp_supported": True, "verification": "passed",
    }


def test_build_and_verify_single_onnx_bundle(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx-fixture")
    model.with_suffix(".json").write_text(json.dumps(_config()), encoding="utf-8")
    bundle = build_deployment_bundle(model, tmp_path / "release.dvdeploy")
    manifest = verify_deployment_bundle(bundle)
    assert manifest["backend"] == "builtin"
    assert (bundle / "model.onnx").read_bytes() == b"onnx-fixture"
    assert json.loads((bundle / "manifest.json").read_text())["config"] == "model.json"


def test_export_checkpoint_can_emit_sdk_bundle(tmp_path):
    import torch
    sys.path.insert(0, str(ROOT / "python"))
    from builtin_models import build_builtin_model, make_builtin_checkpoint
    from export_onnx import export_checkpoint

    model = build_builtin_model("resnet18", 2, 3).eval()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(make_builtin_checkpoint("resnet18", model, num_classes=2,
                                       input_size=32, class_names=["ok", "ng"]), checkpoint)
    bundle = tmp_path / "model.dvdeploy"
    result = export_checkpoint(checkpoint, tmp_path / "model.onnx", verify=True,
                               log=lambda _: None, bundle_output=bundle)
    assert result["bundle_path"] == str(bundle)
    assert verify_deployment_bundle(bundle)["backend"] == "builtin"


def test_build_renames_config_to_model_json_and_copies_companion_graphs(tmp_path):
    source = tmp_path / "sam-export"
    source.mkdir()
    (source / "sam2_encoder.onnx").write_bytes(b"encoder")
    (source / "sam2_decoder.onnx").write_bytes(b"decoder")
    config = _config("sam2_encoder.onnx")
    config.update({"backend": "sam2", "task": "segment", "contracts": {
        "graphs": {"encoder": {"file": "sam2_encoder.onnx"},
                    "decoder": {"file": "sam2_decoder.onnx"}}}})
    (source / "sam2.json").write_text(json.dumps(config), encoding="utf-8")
    bundle = build_deployment_bundle(source, tmp_path / "sam.dvdeploy")
    assert (bundle / "sam2.json").is_file()
    assert verify_deployment_bundle(bundle)["task"] == "segment"


def test_patchcore_export_can_emit_sdk_bundle(tmp_path):
    import torch
    sys.path.insert(0, str(ROOT / "python"))
    from export_patchcore_onnx import export_patchcore_model
    from patchcore import PatchCore

    model = PatchCore(input_size=32, n_neighbors=1, backbone_name="resnet18", device="cpu",
                      pretrained=False)
    with torch.inference_mode():
        feature_channels = model._avg_pool(model.backbone(torch.zeros(1, 3, 32, 32))).shape[1]
    model.memory_bank = torch.randn(2, feature_channels)
    model.anomaly_threshold = .5
    bundle = tmp_path / "patchcore.dvdeploy"
    result = export_patchcore_model(model, tmp_path / "patchcore.onnx", verify=True,
                                    log=lambda _: None, bundle_output=bundle)
    assert result["bundle_path"] == str(bundle)
    assert verify_deployment_bundle(bundle)["backend"] == "patchcore"


def test_verify_rejects_tampering_and_zip_slip(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx-fixture")
    model.with_suffix(".json").write_text(json.dumps(_config()), encoding="utf-8")
    bundle = build_deployment_bundle(model, tmp_path / "release.dvdeploy")
    (bundle / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(DeploymentBundleError, match="checksum"):
        verify_deployment_bundle(bundle)
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../manifest.json", "{}")
    with pytest.raises(DeploymentBundleError, match="unsafe"):
        verify_deployment_bundle(archive)


def test_verify_rejects_zip_symlink_and_duplicate_entries(tmp_path):
    symlink_archive = tmp_path / "symlink.zip"
    with zipfile.ZipFile(symlink_archive, "w") as output:
        info = zipfile.ZipInfo("manifest.json")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        output.writestr(info, "target")
    with pytest.raises(DeploymentBundleError, match="symlink"):
        verify_deployment_bundle(symlink_archive)

    duplicate_archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate_archive, "w") as output:
        output.writestr("manifest.json", "{}")
        output.writestr("manifest.json", "{}")
    with pytest.raises(DeploymentBundleError, match="duplicate"):
        verify_deployment_bundle(duplicate_archive)


def test_verify_rejects_drive_paths_and_duplicate_normalized_manifest_keys(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx-fixture")
    model.with_suffix(".json").write_text(json.dumps(_config()), encoding="utf-8")
    bundle = build_deployment_bundle(model, tmp_path / "release.dvdeploy")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["C:/escape.onnx"] = manifest["files"]["model.onnx"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DeploymentBundleError, match="unsafe"):
        verify_deployment_bundle(bundle)
    del manifest["files"]["C:/escape.onnx"]
    (bundle / "nested").mkdir()
    (bundle / "nested" / "model.onnx").write_bytes(b"onnx-fixture")
    manifest["files"]["nested/model.onnx"] = manifest["files"]["model.onnx"]
    manifest["files"]["nested\\model.onnx"] = manifest["files"]["model.onnx"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DeploymentBundleError, match="duplicate"):
        verify_deployment_bundle(bundle)


def test_build_and_verify_reject_symlinked_bundle_paths(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx-fixture")
    model.with_suffix(".json").write_text(json.dumps(_config()), encoding="utf-8")
    source_link = tmp_path / "source-link"
    try:
        source_link.symlink_to(model, target_is_directory=False)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")
    with pytest.raises(DeploymentBundleError, match="source symlink"):
        build_deployment_bundle(source_link, tmp_path / "source-link.dvdeploy")

    destination = tmp_path / "destination.dvdeploy"
    destination_link = tmp_path / "destination-link.dvdeploy"
    destination_link.symlink_to(destination, target_is_directory=True)
    with pytest.raises(DeploymentBundleError, match="output symlink"):
        build_deployment_bundle(model, destination_link)


def test_verify_rejects_symlinked_bundle_directory_and_non_normalized_key(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx-fixture")
    model.with_suffix(".json").write_text(json.dumps(_config()), encoding="utf-8")
    bundle = build_deployment_bundle(model, tmp_path / "release.dvdeploy")
    link = tmp_path / "release-link.dvdeploy"
    try:
        link.symlink_to(bundle, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    with pytest.raises(DeploymentBundleError, match="bundle symlink"):
        verify_deployment_bundle(link)

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["model\\.onnx"] = manifest["files"].pop("model.onnx")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DeploymentBundleError, match="not normalized"):
        verify_deployment_bundle(bundle)
