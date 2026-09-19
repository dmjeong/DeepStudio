"""기본 모델 카탈로그와 Docker worker 계약 회귀 검사."""

import json
import stat
import zipfile
from io import BytesIO

import pytest
from core.container_worker import (
    ContainerCommand,
    ContainerWorker,
    ContainerWorkerError,
    Frame,
    build_container_command,
)
from core.model_registry import (
    ModelRegistry,
    ModelRegistryError,
    ModelSpec,
    builtin_model_specs,
    installed_model_path,
    registry_with_installed_packs,
)


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
    sam2 = registry.get("sam2_hiera_large")
    assert sam2.capabilities >= {"prompt", "automatic_mask"}
    assert "video" not in sam2.capabilities
    assert sam2.metadata["contracts"]["video_state"] is False
    assert registry.get("patchcore_resnet18").release_status == "export_verified"
    assert registry.get("patchcore_wide_resnet50_2").release_status == "export_verified"
    assert registry.get("resnet18").release_status == "export_verified"
    assert registry.get("deeplabv3plus_resnet34").release_status == "export_verified"


def test_registry_rejects_duplicate_or_unsafe_pack(tmp_path):
    registry = ModelRegistry.builtin()
    with pytest.raises(ModelRegistryError, match="duplicate"):
        registry.register(builtin_model_specs()[0])
    pack = tmp_path / "unsafe.dvmodel"
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("../manifest.json", json.dumps({}))
    with pytest.raises(ModelRegistryError, match="unsafe"):
        registry.load_pack(pack)


def test_registry_rejects_archive_symlinks_duplicate_paths_and_pack_symlink(tmp_path):
    registry = ModelRegistry.builtin()
    manifest = json.dumps({
        "schema_version": 1, "model_id": "vendor.archive-safe", "family": "Example",
        "variant": "Small", "task": "classify", "runtimes": ["onnx"],
        "capabilities": ["infer"], "input_size": [32, 32], "input_channels": [3],
        "release_status": "scoped",
    })

    symlink_pack = tmp_path / "symlink.dvmodel"
    with zipfile.ZipFile(symlink_pack, "w") as archive:
        archive.writestr("manifest.json", manifest)
        info = zipfile.ZipInfo("weights.onnx")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "outside")
    with pytest.raises(ModelRegistryError, match="symlink"):
        registry.load_pack(symlink_pack)

    duplicate_pack = tmp_path / "duplicate.dvmodel"
    with zipfile.ZipFile(duplicate_pack, "w") as archive:
        archive.writestr("manifest.json", manifest)
        archive.writestr("README.md", "one")
        archive.writestr("README.md", "two")
    with pytest.raises(ModelRegistryError, match="duplicate"):
        registry.load_pack(duplicate_pack)

    pack_link = tmp_path / "pack-link.dvmodel"
    try:
        pack_link.symlink_to(duplicate_pack)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")
    with pytest.raises(ModelRegistryError, match=".dvmodel"):
        registry.load_pack(pack_link)


def test_registry_rejects_windows_reserved_model_ids():
    with pytest.raises(ModelRegistryError, match="invalid model id"):
        ModelSpec("con", "Example", "Small", "classify", ("onnx",),
                  frozenset({"infer"}), (224, 224))


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


def test_registry_rejects_missing_declared_container_image_archive(tmp_path):
    manifest = {
        "schema_version": 1, "model_id": "vendor.container", "family": "Example",
        "variant": "Container", "task": "classify", "runtimes": ["container"],
        "capabilities": ["infer"], "input_size": [224, 224], "input_channels": [3],
        "container_image": "registry.invalid/example@sha256:" + "a" * 64,
        "container_image_archive": "docker/image.tar",
    }
    pack = tmp_path / "container.dvmodel"
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
    with pytest.raises(ModelRegistryError, match="archive"):
        ModelRegistry.builtin().load_pack(pack)


def test_registry_loads_activated_pack_manifests_after_restart(tmp_path):
    root = tmp_path / "installed" / "vendor.example" / "1.0.0"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "model_id": "vendor.example-classifier", "family": "Example",
        "variant": "Small", "task": "classify", "runtimes": ["container"],
        "capabilities": ["train", "infer"], "input_size": [224, 224],
        "input_channels": [3], "release_status": "scoped",
    }), encoding="utf-8")
    registry = ModelRegistry.builtin()
    loaded = registry.load_installed_root(tmp_path / "installed")
    assert loaded[0].model_id == "vendor.example-classifier"


def test_registry_rejects_symlinked_installed_manifest(tmp_path):
    root = tmp_path / "installed" / "vendor.example" / "1.0.0"
    root.mkdir(parents=True)
    external = tmp_path / "external-manifest.json"
    external.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    (root / "manifest.json").symlink_to(external)
    with pytest.raises(ModelRegistryError, match="manifest"):
        ModelRegistry.builtin().load_installed_pack(root)


def test_registry_rejects_non_object_installed_manifest(tmp_path):
    root = tmp_path / "installed" / "vendor.example" / "1.0.0"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ModelRegistryError, match="invalid installed model pack"):
        ModelRegistry.builtin().load_installed_pack(root)


def test_registry_discovers_installed_packs_without_hiding_builtins(tmp_path):
    root = tmp_path / "installed" / "vendor.extra" / "1.0.0"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "model_id": "vendor.extra", "family": "Extra",
        "variant": "Small", "task": "classify", "runtimes": ["container"],
        "capabilities": ["train", "infer"], "input_size": [224, 224],
        "input_channels": [3], "release_status": "scoped",
    }), encoding="utf-8")
    registry, errors = registry_with_installed_packs(tmp_path / "installed")
    assert not errors
    assert registry.get("efficientnet_b0").family == "EfficientNet"
    assert registry.get("vendor.extra").release_status == "scoped"


def test_registry_activates_current_pack_over_catalog_without_loading_old_versions(tmp_path):
    root = tmp_path / "installed" / "resnet18"
    first = root / "1.0.0"
    second = root / "2.0.0"
    for version, target in (("1.0.0", first), ("2.0.0", second)):
        target.mkdir(parents=True)
        (target / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "model_id": "resnet18", "family": "ResNet",
            "variant": "18", "task": "classify", "runtimes": ["onnx"],
            "capabilities": ["infer", "export_onnx", "csharp", "cpp"],
            "input_size": [224, 224], "input_channels": [1, 3],
            "release_status": "release_ready", "pack_version": version, "notes": version,
        }), encoding="utf-8")
    (root / "current.json").write_text(json.dumps({"path": str(first)}), encoding="utf-8")
    registry, errors = registry_with_installed_packs(tmp_path / "installed")
    assert not errors
    assert registry.get("resnet18").notes == "1.0.0"
    assert registry.get("resnet18").release_status == "release_ready"


def test_installed_model_path_follows_current_pointer_safely(tmp_path):
    root = tmp_path / "installed" / "vendor.extra"
    version = root / "1.0.0"
    version.mkdir(parents=True)
    (version / "manifest.json").write_text("{}", encoding="utf-8")
    (root / "current.json").write_text(json.dumps({"path": str(version)}), encoding="utf-8")
    assert installed_model_path("vendor.extra", tmp_path / "installed") == version.resolve()
    assert installed_model_path("../escape", tmp_path / "installed") is None


def test_installed_model_path_ignores_symlinked_current_version(tmp_path):
    model_root = tmp_path / "installed" / "vendor.extra"
    model_root.mkdir(parents=True)
    external = tmp_path / "outside-version"
    external.mkdir()
    (external / "manifest.json").write_text("{}", encoding="utf-8")
    link = model_root / "1.0.0"
    link.symlink_to(external, target_is_directory=True)
    (model_root / "current.json").write_text(json.dumps({"path": str(link)}), encoding="utf-8")
    assert installed_model_path("vendor.extra", tmp_path / "installed") is None


def test_registry_rejects_installed_pack_that_downgrades_catalog_status(tmp_path):
    root = tmp_path / "installed" / "resnet18" / "1.0.0"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "model_id": "resnet18", "family": "ResNet",
        "variant": "18", "task": "classify", "runtimes": ["onnx"],
        "capabilities": ["infer"], "input_size": [224, 224],
        "input_channels": [1, 3], "release_status": "requested",
    }), encoding="utf-8")
    registry, errors = registry_with_installed_packs(tmp_path / "installed")
    assert registry.get("resnet18").release_status == "export_verified"
    assert errors and "downgrade" in errors[0]


def test_registry_rejects_unknown_manifest_schema(tmp_path):
    pack = tmp_path / "unknown-schema.dvmodel"
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"schema_version": 99}))
    with pytest.raises(ModelRegistryError, match="schema_version"):
        ModelRegistry.builtin().load_pack(pack)


def test_registry_requires_special_graph_assets(tmp_path):
    manifest = {
        "schema_version": 1, "model_id": "vendor.sam2", "family": "SAM2",
        "variant": "Hiera Tiny", "task": "segment", "runtimes": ["onnx", "container"],
        "capabilities": ["infer"], "input_size": [1024, 1024], "input_channels": [3],
        "contracts": {"graphs": {
            "encoder": {"file": "encoder.onnx", "inputs": {"image": "input_image"},
                         "outputs": ["image_embeddings"]},
            "decoder": {"file": "decoder.onnx", "inputs": {
                "image_embeddings": "image_embeddings", "point_coords": "point_coords",
                "point_labels": "point_labels", "mask_input": "mask_input",
                "has_mask_input": "has_mask_input", "orig_im_size": "orig_im_size"},
                        "outputs": ["low_res_mask_logits"]},
        }, "prompt_types": ["point"], "video_state": False},
    }
    pack = tmp_path / "sam2.dvmodel"
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
    with pytest.raises(ModelRegistryError, match="missing"):
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
    work_file = tmp_path / "work-file"
    work_file.write_bytes(b"not a directory")
    with pytest.raises(ContainerWorkerError, match="work_dir must be a directory"):
        build_container_command("sha256:" + "a" * 64, model_dir=model, data_dir=data, work_dir=work_file)
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_data = tmp_path / "linked-data"
    linked_data.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ContainerWorkerError, match="data_dir cannot be a symlink"):
        build_container_command("sha256:" + "a" * 64, model_dir=model, data_dir=linked_data, work_dir=work)


def test_container_command_validates_offline_image_archive(tmp_path):
    model, data, work = (tmp_path / name for name in ("model", "data", "work"))
    for path in (model, data, work):
        path.mkdir()
    archive = tmp_path / "image.tar"
    archive.write_bytes(b"image")
    command = build_container_command("sha256:" + "a" * 64, model_dir=model, data_dir=data,
                                     work_dir=work, image_archive=archive)
    assert command.image_archive == str(archive.resolve())
    with pytest.raises(ContainerWorkerError, match="file"):
        build_container_command("sha256:" + "a" * 64, model_dir=model, data_dir=data,
                               work_dir=work, image_archive=tmp_path / "missing.tar")
    linked = tmp_path / "linked-archive.tar"
    linked.symlink_to(archive)
    with pytest.raises(ContainerWorkerError, match="archive"):
        build_container_command("sha256:" + "a" * 64, model_dir=model, data_dir=data,
                               work_dir=work, image_archive=linked)


def test_container_command_supports_owned_wsl_docker_prefix(tmp_path, monkeypatch):
    model, data, work = (tmp_path / name for name in ("model", "data", "work"))
    for path in (model, data, work):
        path.mkdir()
    prefix = ("wsl.exe", "-d", "DeepVisionStudio", "--", "docker")
    converted = iter(("/mnt/c/모델", "/mnt/c/data with spaces", "/mnt/c/work"))

    def fake_run(argv, **kwargs):
        assert tuple(argv[:4]) == prefix[:4]
        assert argv[4:7] == ("wslpath", "-a", "-u")
        return type("Completed", (), {
            "returncode": 0, "stdout": next(converted) + "\n", "stderr": "",
        })()

    monkeypatch.setattr("core.container_worker.subprocess.run", fake_run)
    command = build_container_command("sha256:" + "a" * 64, model_dir=model,
                                     data_dir=data, work_dir=work,
                                     docker_command=prefix)
    assert command.docker_command == prefix
    assert command.argv[: len(prefix) + 1] == (*prefix, "run")
    mounts = [command.argv[index + 1] for index, value in enumerate(command.argv)
              if value == "--mount"]
    assert mounts == [
        "type=bind,src=/mnt/c/모델,dst=/models,readonly",
        "type=bind,src=/mnt/c/data with spaces,dst=/data,readonly",
        "type=bind,src=/mnt/c/work,dst=/work",
    ]


def test_container_worker_loads_and_verifies_offline_image_digest(tmp_path, monkeypatch):
    archive = tmp_path / "image.tar"
    archive.write_bytes(b"image")
    digest = "sha256:" + "a" * 64
    command = ContainerCommand(digest, "worker-archive", (), "deepvision.owner=worker-archive",
                               str(archive))
    worker = ContainerWorker(command)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        if argv[1:3] == ("load", "--input"):
            return type("Completed", (), {"returncode": 0, "stdout": "Loaded", "stderr": ""})()
        if argv[-1] == digest and "{{json .RepoDigests}}" in argv:
            return type("Completed", (), {"returncode": 0, "stdout": "[]", "stderr": ""})()
        return type("Completed", (), {"returncode": 0, "stdout": digest + "\n", "stderr": ""})()

    monkeypatch.setattr("core.container_worker.subprocess.run", fake_run)
    worker._load_image_archive()
    assert calls[0] == ("docker", "load", "--input", str(archive.resolve()))
    assert calls[-1][-1] == digest


def test_container_cleanup_requires_matching_owner_label(monkeypatch):
    command = ContainerCommand("sha256:" + "a" * 64, "worker-1", (), "deepvision.owner=worker-1")
    worker = ContainerWorker(command)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        if argv[1] == "inspect":
            return type("Completed", (), {"returncode": 0, "stdout": "worker-1\n"})()
        return type("Completed", (), {"returncode": 0, "stdout": ""})()

    monkeypatch.setattr("core.container_worker.subprocess.run", fake_run)
    assert worker._remove_owned_container() is True
    assert calls == [
        ("docker", "inspect", "--format", '{{index .Config.Labels "deepvision.owner"}}', "worker-1"),
        ("docker", "rm", "--force", "worker-1"),
    ]

    calls.clear()
    def foreign_run(argv, **kwargs):
        calls.append(tuple(argv))
        return type("Completed", (), {"returncode": 0, "stdout": "someone-else\n"})()

    monkeypatch.setattr("core.container_worker.subprocess.run", foreign_run)
    assert worker._remove_owned_container() is False
    assert calls == [("docker", "inspect", "--format", '{{index .Config.Labels "deepvision.owner"}}', "worker-1")]
