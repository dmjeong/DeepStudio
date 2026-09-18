"""기본 모델 카탈로그와 Docker worker 계약 회귀 검사."""

from io import BytesIO
import json
from pathlib import Path
import zipfile

import pytest

from core.container_worker import Frame, ContainerWorkerError, build_container_command
from core.model_registry import ModelRegistry, ModelRegistryError, builtin_model_specs


def test_requested_catalog_contains_every_product_family_and_redetr_sizes():
    registry = ModelRegistry.builtin()
    ids = {spec.model_id for spec in registry.list()}
    assert {"efficientnet_b0", "efficientnet_b1", "resnet18", "resnet50", "convnext_v1_tiny"} <= ids
    assert {"libreyolo_classify_mobilenetv4_small", "patchcore_wide_resnet50_2", "patchcore_resnet18"} <= ids
    assert {"re_detr_v4_small", "re_detr_v4_medium", "re_detr_v4_large"} <= ids
    assert {"libreyolo_detect_9t", "deeplabv3plus_resnet34", "unet_resnet18"} <= ids
    assert {f"sam2_hiera_{variant}" for variant in ("tiny", "small", "base_plus", "large")} <= ids


def test_catalog_keeps_unverified_models_out_of_release_ready_view():
    registry = ModelRegistry.builtin()
    assert registry.list("detect", release_ready=True) == ()
    assert registry.available("detect") == ()
    assert registry.get("re_detr_v4_medium").release_status == "requested"
    assert registry.get("sam2_hiera_large").capabilities >= {"prompt", "video"}


def test_registry_rejects_duplicate_or_unsafe_pack(tmp_path):
    registry = ModelRegistry.builtin()
    with pytest.raises(ModelRegistryError, match="duplicate"):
        registry.register(builtin_model_specs()[0])
    pack = tmp_path / "unsafe.dvmodel"
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("../manifest.json", json.dumps({}))
    with pytest.raises(ModelRegistryError, match="unsafe"):
        registry.load_pack(pack)


def test_registry_loads_a_valid_new_model_pack_without_importing_code(tmp_path):
    manifest = {
        "schema_version": 1,
        "model_id": "vendor.example-detector",
        "family": "Example Detector",
        "variant": "Small",
        "task": "detect",
        "runtimes": ["onnx"],
        "capabilities": ["infer", "export_onnx", "csharp", "cpp"],
        "input_size": [640, 640],
        "input_channels": [3],
        "release_status": "scoped",
        "runtime_requirements": {"opset": 17},
    }
    pack = tmp_path / "example.dvmodel"
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("README.ko.md", "offline pack")
    loaded = ModelRegistry.builtin().load_pack(pack)
    assert loaded.model_id == manifest["model_id"]
    assert loaded.release_status == "scoped"


def test_registry_rejects_unknown_manifest_schema(tmp_path):
    pack = tmp_path / "unknown-schema.dvmodel"
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"schema_version": 99}))
    with pytest.raises(ModelRegistryError, match="schema_version"):
        ModelRegistry.builtin().load_pack(pack)


def test_worker_frame_roundtrip_and_limits():
    frame = Frame({"type": "infer", "request_id": "r1"}, b"\x00\x01")
    assert Frame.read(BytesIO(frame.encode())) == frame
    with pytest.raises(ContainerWorkerError, match="payload"):
        Frame({"type": "infer"}, b"x" * (64 * 1024 * 1024 + 1)).encode()


def test_container_command_is_networkless_and_uses_read_only_mounts(tmp_path):
    model, data, work = (tmp_path / name for name in ("model", "data", "work"))
    for path in (model, data, work):
        path.mkdir()
    command = build_container_command("sha256:" + "a" * 64, model_dir=model, data_dir=data, work_dir=work,
                                     name="test-worker", cpus=2)
    assert command.argv[0:3] == ("docker", "run", "--rm")
    assert "--network" in command.argv and command.argv[command.argv.index("--network") + 1] == "none"
    assert "--log-driver=none" in command.argv
    assert "readonly" in command.argv[command.argv.index("--mount") + 1]
    with pytest.raises(ContainerWorkerError, match="absolute"):
        build_container_command("sha256:" + "a" * 64, model_dir="relative", data_dir=data, work_dir=work)
