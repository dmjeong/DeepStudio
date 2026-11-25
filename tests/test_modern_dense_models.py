"""Modern models must train, deploy FP32/INT8 and retain original coordinates."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "gui")]
import onnxruntime as ort
import torch

ort.disable_telemetry_events()
from modern_segmentation import MODEL_IDS


@pytest.fixture(scope="module", autouse=True)
def matching_native_runtime():
    """Compare language bindings of the same runtime, not different kernels."""
    if not os.environ.get("SEGMENT_PROBE"):
        return
    segment = Path(os.environ["SEGMENT_PROBE"])
    for probe in (segment, segment.with_name("anomaly_probe" + segment.suffix)):
        result = subprocess.run(
            [str(probe), "--runtime-version"], capture_output=True, text=True,
            check=True, timeout=10,
        )
        assert result.stdout.strip() == ort.__version__, (
            f"Python ORT {ort.__version__} != {probe.name} ORT {result.stdout.strip()}. "
            "Install the pinned runtime used by the native SDK before comparing outputs."
        )


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    yield
    torch.set_num_threads(previous)


def data(root):
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "masks" / split).mkdir(parents=True)
        for i in range(2):
            rng = np.random.default_rng(i + (10 if split == "val" else 0))
            x = rng.integers(30, 150, (75, 110, 3), dtype=np.uint8)
            y = np.zeros((75, 110), np.uint8)
            y[20:22, 15:70] = 1
            x[y == 1] = 240
            y[:2] = 255
            Image.fromarray(x).save(root / "images" / split / f"{i}.png")
            Image.fromarray(y).save(root / "masks" / split / f"{i}.png")
    return root


@pytest.mark.parametrize("model_id", MODEL_IDS)
@pytest.mark.parametrize("channels", [1, 3])
def test_modern_forward_restore_backward(model_id, channels):
    from builtin_models import (
        build_builtin_model,
        load_builtin_checkpoint,
        make_builtin_checkpoint,
    )
    from segmentation_loss import SegmentationLoss

    model = build_builtin_model(model_id, 3, channels)
    x = torch.randn(1, channels, 75, 110)
    y = torch.randint(0, 3, (1, 75, 110))
    y[:, :3] = 255
    criterion = SegmentationLoss(3)
    if hasattr(model, "compute_training_loss"):
        logits, loss = model.compute_training_loss(x, y, criterion)
    else:
        logits = model(x)
        loss = criterion(logits, y)
    loss.backward()
    assert logits.shape == (1, 3, 75, 110) and torch.isfinite(logits).all()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    model.eval()
    checkpoint = make_builtin_checkpoint(
        model_id,
        model,
        num_classes=3,
        input_size=64,
        in_channels=channels,
        class_names=["background", "crack", "fm"],
    )
    restored = load_builtin_checkpoint(checkpoint).eval()
    model.eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(x), restored(x))


@pytest.mark.skipif(
    os.environ.get("DVS_RUN_MODERN_INTEGRATION") != "1",
    reason="Requires prepared official pretrained assets and native SDK probe",
)
@pytest.mark.parametrize("model_id", MODEL_IDS)
@pytest.mark.parametrize("channels", [1, 3])
def test_modern_pretrained_train_fp32_int8_cpp_original(model_id, channels, tmp_path):
    import cv2
    from core.inference_loading import load_inference_engine
    from dense_deployment import predict_dense
    from export_onnx import export_checkpoint
    from model_crypto import create_key, encrypt_config, read_key
    from quantize_dense import quantize_dense
    from train_builtin import train_builtin

    root = data(tmp_path / "data")
    best = train_builtin(
        model_id,
        root,
        num_classes=2,
        class_names=["background", "crack"],
        input_size=int(os.environ.get("DVS_MODERN_INPUT_SIZE", "64")),
        in_channels=channels,
        epochs=1,
        batch_size=1,
        learning_rate=1e-4,
        pretrained=True,
        num_workers=0,
        patch_training=True,
        patch_scale=0.5 if channels == 1 else 1.0,
        patches_per_image=1,
        output_dir=tmp_path / "run",
        log=lambda _: None,
    )
    state = torch.load(best, map_location="cpu", weights_only=True)
    assert state["weight_provenance"]["source"] != "random"
    exported = export_checkpoint(
        best,
        tmp_path / "model.onnx",
        verify=True,
        dynamic_batch=True,
        log=lambda _: None,
    )
    assert exported["cpp_supported"] and exported["verification"] == "passed"
    config = Path(exported["config_path"])
    assert json.loads(config.read_text())["schema_version"] == 8
    quantized = quantize_dense(
        config, root / "images" / "val", count=1, log=lambda _: None
    )
    image = root / "images" / "val" / "0.png"
    frame = cv2.imread(str(image))
    probe = os.environ["SEGMENT_PROBE"]
    for manifest_path in (config, quantized):
        manifest = json.loads(manifest_path.read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # same setting as the native probe
        session = ort.InferenceSession(
            str(manifest_path.parent / manifest["model_path"]),
            options,
            providers=["CPUExecutionProvider"],
        )
        expected = predict_dense(
            manifest,
            frame,
            lambda t, s=session, n=manifest["input_name"]: s.run(None, {n: t}),
        ).argmax(0)
        expected = np.array(
            Image.fromarray(expected.astype(np.uint8)).resize(
                frame.shape[1::-1], Image.Resampling.NEAREST
            )
        )
        output = tmp_path / (manifest_path.stem + ".png")
        result = subprocess.run(
            [probe, str(manifest_path), str(image), str(output)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        np.testing.assert_array_equal(
            np.array(Image.open(output)), expected, err_msg=manifest_path.name,
        )
    key = create_key(tmp_path / "model.key")
    sealed = tmp_path / "model.dvsenc"
    encrypt_config(config, sealed, read_key(key))
    result = subprocess.run(
        [
            probe,
            str(sealed),
            str(image),
            str(tmp_path / "sealed.png"),
            "1",
            "1",
            str(key),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(
        np.array(Image.open(tmp_path / "sealed.png")),
        np.array(Image.open(tmp_path / "model.png")),
    )
    encrypt_config(quantized, sealed, read_key(key))
    result = subprocess.run(
        [
            probe,
            str(sealed),
            str(image),
            str(tmp_path / "sealed_int8.png"),
            "1",
            "1",
            str(key),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(
        np.array(Image.open(tmp_path / "sealed_int8.png")),
        np.array(Image.open(tmp_path / "model_int8.png")),
    )
    if model_id == "efficientvit_b0_seg" and channels == 3:
        # Exercise the GUI service's complete INT8 -> encryption route, not
        # just wrapping an already exported model ourselves.
        direct = tmp_path / "direct_int8.dvsenc"
        result = export_checkpoint(
            best,
            direct,
            precision="int8",
            calibration_dir=root / "images" / "val",
            calibration_count=1,
            encryption_key_path=key,
            dynamic_batch=True,
            log=lambda _: None,
        )
        assert result["verification"] == "int8_compared"
        process = subprocess.run(
            [
                probe,
                str(direct),
                str(image),
                str(tmp_path / "direct.png"),
                "1",
                "1",
                str(key),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert process.returncode == 0, process.stderr
        np.testing.assert_array_equal(
            np.array(Image.open(tmp_path / "direct.png")),
            np.array(Image.open(tmp_path / "model_int8.png")),
        )
    engine = load_inference_engine(str(best))
    engine.infer(str(image))
    np.testing.assert_array_equal(
        engine._original_preview_rgb, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )


@pytest.mark.parametrize("model_id", [*MODEL_IDS, "dinomaly_vits14"])
def test_modern_qt_controls_and_round_trip(model_id, tmp_path):
    from core.project import ProjectManager
    from core.training_modes import validate_training_options
    from PySide6.QtWidgets import QApplication
    from widgets.export_widget import ExportWidget
    from widgets.training_widget import TrainingWidget

    app = QApplication.instance() or QApplication([])
    task = "anomaly" if model_id == "dinomaly_vits14" else "segment"
    project = ProjectManager.create_new(
        "modern", task, str(tmp_path / "project"), ["background", "crack"]
    )
    widget = TrainingWidget()
    widget.set_project(project)
    widget.model_id_combo.setCurrentIndex(widget.model_id_combo.findData(model_id))
    assert widget.mode_combo.currentData() == "builtin_finetune"
    assert widget.mode_combo.isEnabled()
    assert widget.epochs_spin.isEnabled()
    assert widget.patch_check.isEnabled()
    widget.patch_check.setChecked(True)
    widget.collect_config()
    assert project.training.patch_training
    if task == "segment":
        widget.teacher_edit.setText("/tmp/teacher.pt")
        widget.kd_alpha.setValue(0.4)
        widget.collect_config()
        assert project.training.teacher_checkpoint == "/tmp/teacher.pt"
    else:
        assert project.training.anomaly_method == "dinomaly"
        assert project.training.training_mode == "builtin_finetune"
        assert not widget.pc_sampling_spin.isEnabled()
    capabilities = validate_training_options(project, require_runnable=False)
    assert capabilities["patch_training"] and "rotation" in capabilities["augmentation"]
    widget.mode_combo.setCurrentIndex(widget.mode_combo.findData("builtin_resume"))
    assert widget.epochs_spin.isEnabled()
    assert not widget.lr_spin.isEnabled()
    assert not widget.patch_check.isEnabled()
    export = ExportWidget()
    export.set_project(project)
    assert export.export_precision.findData("int8") >= 0
    widget.deleteLater()
    export.deleteLater()
    app.processEvents()


def test_distillation_same_view_ignore_labels_and_contract(tmp_path):
    from builtin_models import build_builtin_model, make_builtin_checkpoint
    from segmentation_distillation import SegmentationDistiller

    teacher = build_builtin_model("unet_lite", 2, 3).eval()
    checkpoint = make_builtin_checkpoint(
        "unet_lite",
        teacher,
        num_classes=2,
        input_size=64,
        in_channels=3,
        class_names=["background", "crack"],
    )
    path = tmp_path / "teacher.pt"
    torch.save(checkpoint, path)
    distiller = SegmentationDistiller(path, ["background", "crack"], 3, "cpu")
    seen = []
    distiller.teacher.register_forward_pre_hook(
        lambda _m, args: seen.append(args[0].detach().clone())
    )
    images = torch.randn(1, 3, 64, 64)
    logits = torch.randn(1, 2, 64, 64, requires_grad=True)
    masks = torch.full((1, 64, 64), 255)
    loss = distiller(images, logits, masks, logits.sum() * 0)
    loss.backward()
    torch.testing.assert_close(seen[0], images)
    assert logits.grad.count_nonzero() == 0
    assert all(p.grad is None for p in distiller.teacher.parameters())
    with pytest.raises(ValueError, match="클래스"):
        SegmentationDistiller(path, ["crack", "background"], 3, "cpu")


@pytest.mark.skipif(
    os.environ.get("DVS_RUN_MODERN_INTEGRATION") != "1",
    reason="Official DINOv2 asset and native SDK required",
)
@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("patch_training", [True, False], ids=["tiles", "whole"])
def test_dinomaly_train_export_int8_cpp_preview(channels, patch_training, tmp_path):
    import cv2
    from core.inference_loading import load_inference_engine
    from dense_deployment import predict_dense
    from dinomaly_runtime import DinomalyRuntime
    from export_onnx import export_checkpoint
    from model_crypto import create_key, encrypt_config, read_key
    from train_dinomaly import train_dinomaly

    root = tmp_path / "data"
    for split, labels in [
        ("train", ["good"]),
        ("val", ["good", "crack"]),
        ("test", ["unused"]),
    ]:
        for label in labels:
            folder = root / split / label
            folder.mkdir(parents=True)
            for i in range(2):
                image = np.random.default_rng(i + len(split)).integers(
                    60, 140, (73, 91, 3), dtype=np.uint8
                )
                if label == "crack":
                    image[22:25, 10:55] = 250
                Image.fromarray(image).save(folder / f"{i}.png")
    best = train_dinomaly(
        root,
        output_dir=tmp_path / "train",
        input_size=int(os.environ.get("DVS_DINOMALY_INPUT_SIZE", "64")),
        in_channels=channels,
        epochs=1,
        batch_size=2,
        patch_training=patch_training,
        patches_per_image=1,
        center_crop={"width": 80, "height": 64},
        log=lambda _: None,
    )
    state = torch.load(best, weights_only=True, map_location="cpu")
    assert state["calibration"]["test_used"] is False
    assert state["calibration"]["normal_count"] == 2
    runtime = DinomalyRuntime(state)
    assert all(not p.requires_grad for p in runtime.model.encoder.parameters())
    path = root / "val" / "crack" / "0.png"
    score, maps = runtime.predict_from_path(path, center_crop=runtime.center_crop)
    assert score[0] == maps.max() and np.isfinite(maps).all()
    exported = export_checkpoint(
        best, tmp_path / "model.onnx", verify=True, log=lambda _: None
    )
    quantized = export_checkpoint(
        best,
        tmp_path / "model_int8.onnx",
        precision="int8",
        calibration_dir=root / "train",
        calibration_count=1,
        log=lambda _: None,
    )
    frame = cv2.imread(str(path))
    for result in (exported, quantized):
        config = Path(result["config_path"])
        manifest = json.loads(config.read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # same setting as the native probe
        session = ort.InferenceSession(
            result["output_path"], options, providers=["CPUExecutionProvider"]
        )
        expected = predict_dense(
            manifest,
            frame,
            lambda t, s=session, n=manifest["input_name"]: s.run(None, {n: t}),
        )[0]
        target = tmp_path / (config.stem + ".f32")
        segment_probe = Path(os.environ["SEGMENT_PROBE"])
        probe = Path(
            os.environ.get(
                "ANOMALY_PROBE",
                segment_probe.with_name("anomaly_probe" + segment_probe.suffix),
            )
        )
        process = subprocess.run(
            [str(probe), str(config), str(path), str(target)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert process.returncode == 0, process.stderr
        actual = np.fromfile(target, np.float32).reshape(frame.shape[:2])
        truth = np.zeros(frame.shape[:2], np.float32)
        truth[4:68, 5:85] = cv2.resize(expected, (80, 64), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_allclose(actual, truth, atol=2e-5, rtol=2e-4, err_msg=config.name)
        info = json.loads(process.stdout.strip().splitlines()[-1])
        assert abs(info["score"] - float(expected.max())) < 2e-5
        key = create_key(tmp_path / (config.stem + ".key"))
        sealed = tmp_path / (config.stem + ".dvsenc")
        encrypt_config(config, sealed, read_key(key))
        encrypted_map = tmp_path / (config.stem + ".encrypted.f32")
        process = subprocess.run(
            [str(probe), str(sealed), str(path), str(encrypted_map), str(key)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert process.returncode == 0, process.stderr
        np.testing.assert_array_equal(
            np.fromfile(encrypted_map, np.float32), actual.ravel()
        )
    engine = load_inference_engine(best)
    prediction = engine.infer(str(path))
    assert prediction.status == "ok", prediction
    np.testing.assert_array_equal(
        engine._original_preview_rgb, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )


def test_distillation_training_resume_and_standalone_student(tmp_path):
    from builtin_models import build_builtin_model, make_builtin_checkpoint
    from export_onnx import export_checkpoint
    from train_builtin import train_builtin

    teacher = build_builtin_model("unet_lite", 2, 3).eval()
    path = tmp_path / "teacher.pt"
    torch.save(
        make_builtin_checkpoint(
            "unet_lite",
            teacher,
            num_classes=2,
            input_size=64,
            in_channels=3,
            class_names=["background", "crack"],
        ),
        path,
    )
    root = data(tmp_path / "data")
    options = {
        "num_classes": 2,
        "class_names": ["background", "crack"],
        "input_size": 64,
        "batch_size": 2,
        "patch_training": True,
        "patches_per_image": 1,
        "teacher_checkpoint": path,
        "log": lambda _: None,
    }
    train_builtin("unet_lite", root, output_dir=tmp_path / "first", epochs=1, **options)
    best = train_builtin(
        "unet_lite",
        root,
        output_dir=tmp_path / "resume",
        epochs=2,
        resume=tmp_path / "first" / "last.pt",
        **options,
    )
    state = torch.load(best, map_location="cpu", weights_only=True)
    assert state["training_config"]["distillation"]["views"] == "same_augmented_input"
    assert not any("teacher" in name for name in state["model_state_dict"])
    with pytest.raises(ValueError, match="증류"):
        train_builtin(
            "unet_lite",
            root,
            output_dir=tmp_path / "bad",
            epochs=3,
            resume=tmp_path / "resume" / "last.pt",
            **{**options, "distillation_alpha": 0.7},
        )
    path.unlink()  # Student export must not reopen or include its teacher.
    export_checkpoint(best, tmp_path / "student.onnx", log=lambda _: None)


def test_dense_tiles_preserve_pixels_and_training_rounding():
    from checkpoint import tiling_contract
    from dataset import scaled_size
    from dense_deployment import (
        PreparedWindows,
        predict_dense,
        region_pixels,
        tile_contract,
    )

    config = {
        "task": "segment",
        "input_height": 64,
        "input_width": 64,
        "input_channels": 3,
        "num_classes": 3,
        "normalize_mean": [0, 0, 0],
        "normalize_std": [1, 1, 1],
        "preprocessing": {"tiling": tiling_contract(64, 0.5)},
    }
    bgr = np.random.default_rng(43).integers(0, 256, (181, 277, 3), np.uint8)
    pixels = region_pixels(config, bgr)
    assert pixels.shape[1:] == scaled_size(277, 181, 0.5)[::-1] == (90, 138)
    assert len(PreparedWindows(config, pixels)) > 1
    expected = np.exp(pixels - pixels.max(0, keepdims=True))
    expected /= expected.sum(0, keepdims=True)
    np.testing.assert_allclose(
        predict_dense(config, bgr, lambda t: [t]), expected, atol=1e-7
    )
    config["preprocessing"]["tiling"]["overlap"] = 64
    with pytest.raises(ValueError, match="contract"):
        tile_contract(config)


def test_export_worker_carries_precision_calibration_and_encryption(monkeypatch):
    from widgets import export_widget

    seen = []
    monkeypatch.setattr(
        export_widget, "export_checkpoint", lambda *a, **kw: seen.append(kw) or {}
    )
    worker = export_widget.ExportWorker(
        "best.pt",
        "model.dvsenc",
        17,
        False,
        False,
        True,
        precision="int8",
        calibration_dir="train",
        encryption_key_path="secret.key",
    )
    worker.run()
    assert seen[0]["precision"] == "int8" and seen[0]["calibration_dir"] == "train"
    assert seen[0]["encryption_key_path"] == "secret.key" and seen[0]["verify"]


@pytest.mark.skipif(
    os.environ.get("DVS_RUN_MODERN_INTEGRATION") != "1",
    reason="Official DINOv2 asset required",
)
def test_dinomaly_desktop_worker_metrics_csv_and_resume(tmp_path):
    from core.project import ProjectManager
    from widgets.training_results import _metric_info_for

    from webapp.worker import _train_builtin_project

    root = tmp_path / "data"
    for split in ("train", "val"):
        (root / split / "good").mkdir(parents=True)
        for index in range(2):
            Image.fromarray(np.full((64, 64, 3), 90 + index * 10, np.uint8)).save(
                root / split / "good" / f"{index}.png"
            )
    project = ProjectManager.create_new(
        "dino", "anomaly", str(tmp_path / "project"), ["good", "crack"]
    )
    project.model.model_id = "dinomaly_vits14"
    project.data.root = str(root)
    cfg = project.training
    cfg.anomaly_method = "dinomaly"
    cfg.training_mode = "builtin_finetune"
    cfg.input_size = 64
    cfg.epochs = 1
    cfg.batch_size = 2
    cfg.scheduler = "none"
    events = []

    class Context:
        def emit(self, event, args):
            events.append((event, args))

        def cancelled(self):
            return False

    record = _train_builtin_project(Context(), project, "cpu")
    assert record.best_metric_name == "val_loss" and record.best_epoch == 1
    assert "val_loss" in Path(record.checkpoint_path).name
    assert any(name == "epoch_finished" for name, _ in events)
    assert any(name == "lr_updated" for name, _ in events)
    assert _metric_info_for(project)["primary"] == "val_loss"
    run = Path(record.checkpoint_path).parent
    csvs = list(run.glob("*.csv"))
    assert csvs and any(
        "learning_rate" in p.read_text(encoding="utf-8-sig") for p in csvs
    )
    project.model.pretrained_weights = str(run / "last.pt")
    cfg.training_mode = "builtin_resume"
    cfg.epochs = 2
    resumed = _train_builtin_project(Context(), project, "cpu")
    assert resumed.epochs_done == 2
    cfg.augmentation.rotation += 1
    with pytest.raises(ValueError, match="재개 학습 설정"):
        _train_builtin_project(Context(), project, "cpu")


@pytest.mark.skipif(
    not os.environ.get("SEGMENT_PROBE"), reason="Native C++ probes required"
)
@pytest.mark.parametrize("task", ["segment", "anomaly"])
def test_dense_cpp_multitile_crop_scale_and_no_arguments(task, tmp_path):
    import copy
    import shutil

    import cv2
    import onnx
    from checkpoint import tiling_contract
    from dense_deployment import predict_dense
    from dinomaly_runtime import deployment_config
    from onnx import TensorProto, helper

    tile = tiling_contract(32, 0.5)
    tile["blend"] = "center_weighted_map"
    config = deployment_config(
        {
            "input_size": 32,
            "in_channels": 1,
            "tiling": tile,
            "center_crop": {"width": 275, "height": 179},
            "anomaly_threshold": 1.0,
        }
    )
    config["model_path"] = "model.onnx"
    def value(name, shape):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
    if task == "segment":
        config.update(
            task="segment",
            backend="builtin",
            num_classes=2,
            class_names=["background", "crack"],
            output_name="logits",
            output_names=["logits"],
            postprocessing={},
        )
        config["preprocessing"]["tiling"]["blend"] = "center_weighted_softmax"
        nodes = [
            helper.make_node("Neg", ["input_image"], ["negative"]),
            helper.make_node("Concat", ["input_image", "negative"], ["logits"], axis=1),
        ]
        outputs = [value("logits", [1, 2, 32, 32])]
    else:
        nodes = [
            helper.make_node("Abs", ["input_image"], ["anomaly_map"]),
            helper.make_node(
                "ReduceMax",
                ["anomaly_map"],
                ["anomaly_score"],
                axes=[1, 2, 3],
                keepdims=0,
            ),
        ]
        outputs = [value("anomaly_score", [1]), value("anomaly_map", [1, 1, 32, 32])]
    graph = helper.make_model(
        helper.make_graph(
            nodes, "dense-contract", [value("input_image", [1, 1, 32, 32])], outputs
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=8,
    )
    onnx.save(graph, tmp_path / "model.onnx")
    manifest = tmp_path / "model.json"
    manifest.write_text(json.dumps(config), encoding="utf-8")
    pixels = np.random.default_rng(28).integers(0, 256, (181, 277, 3), np.uint8)
    path = tmp_path / "image.png"
    Image.fromarray(pixels).save(path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(tmp_path / "model.onnx"), options)
    dense = predict_dense(
        config, pixels[..., ::-1], lambda x: session.run(None, {"input_image": x})
    )
    segment = Path(os.environ["SEGMENT_PROBE"])
    probe = (
        segment
        if task == "segment"
        else segment.with_name("anomaly_probe" + segment.suffix)
    )
    output = tmp_path / ("result.png" if task == "segment" else "result.f32")
    process = subprocess.run(
        [str(probe), str(manifest), str(path), str(output)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    if task == "segment":
        expected = np.zeros(pixels.shape[:2], np.uint8)
        expected[1:180, 1:276] = np.array(
            Image.fromarray(dense.argmax(0).astype(np.uint8)).resize(
                (275, 179), Image.Resampling.NEAREST
            )
        )
        np.testing.assert_array_equal(np.array(Image.open(output)), expected)
    else:
        expected = np.zeros(pixels.shape[:2], np.float32)
        expected[1:180, 1:276] = cv2.resize(
            dense[0], (275, 179), interpolation=cv2.INTER_LINEAR
        )
        # Only the restored display map uses OpenCV's float bilinear resize:
        # 4.10 x86 vs 5.0 Python differ by <= 7.7e-6 in coordinate rounding.
        # The model/stitched score below retains its separate stricter check.
        np.testing.assert_allclose(
            np.fromfile(output, np.float32).reshape(expected.shape), expected, atol=1e-5
        )
        assert json.loads(process.stdout)["score"] == pytest.approx(
            float(dense.max()), abs=2e-6
        )
    invalid = copy.deepcopy(config)
    invalid["preprocessing"]["tiling"]["overlap"] = 32
    broken = tmp_path / "invalid.json"
    broken.write_text(json.dumps(invalid), encoding="utf-8")
    rejected = subprocess.run(
        [str(probe), str(broken), str(path), str(tmp_path / "invalid.png")],
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert rejected.returncode != 0 and not (tmp_path / "invalid.png").exists()
    example = Path(
        os.environ.get(
            "DENSE_EXAMPLE", segment.with_name("dense_model_example" + segment.suffix)
        )
    )
    assert example.is_file(), (
        "Build the public dense_model_example target as well as the probes"
    )
    folder = tmp_path / "한글 실행 폴더"
    folder.mkdir()
    for artifact in (manifest, path, tmp_path / "model.onnx"):
        shutil.copy2(artifact, folder / artifact.name)
    binary = folder / example.name
    shutil.copy2(example, binary)
    # Deploy the runtime as well as the EXE. System32 precedes PATH and may
    # contain an incompatible Microsoft onnxruntime.dll (access violation).
    if os.name == "nt":
        assert (example.parent / "onnxruntime.dll").is_file()
        for library in example.parent.glob("*.dll"):
            shutil.copy2(library, folder / library.name)
    environment = {
        **os.environ,
        "PATH": str(example.parent) + os.pathsep + os.environ.get("PATH", ""),
    }
    result = subprocess.run(
        [str(binary)],
        check=False,
        cwd=tmp_path.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(np.array(Image.open(folder / "original.png")), pixels)
    if task == "segment":
        np.testing.assert_array_equal(
            np.array(Image.open(folder / "classes.png")), expected
        )
    else:
        np.testing.assert_allclose(
            np.fromfile(folder / "anomaly_scores.f32", np.float32).reshape(
                expected.shape
            ),
            expected,
            atol=1e-5,  # float display-map resize; see native comparison above
        )


@pytest.mark.parametrize("shape", [(31, 45), (75, 110), (640, 640)])
def test_eomt_query_targets_keep_original_coordinates_when_padded(shape):
    from eomt import query_loss
    from torch.nn import functional as F

    height, width = shape
    target = torch.zeros(1, height, width, dtype=torch.long)
    target[:, height // 3 : height // 3 + 2, width // 3 : width - 2] = 1
    target[:, :2] = 255
    masks = torch.randn(1, 3, 8, 12, requires_grad=True)
    classes = torch.randn(1, 3, 3, requires_grad=True)
    # Adding only ignored right/bottom padding must not move the real labels.
    padded = F.pad(target, (0, (-width) % 14, 0, (-height) % 14), value=255)
    cropped_loss = query_loss(masks, classes, target)
    padded_loss = query_loss(masks, classes, padded)
    torch.testing.assert_close(cropped_loss, padded_loss, rtol=0, atol=1e-6)
    cropped_gradient = torch.autograd.grad(cropped_loss, masks, retain_graph=True)[0]
    padded_gradient = torch.autograd.grad(padded_loss, masks)[0]
    torch.testing.assert_close(cropped_gradient, padded_gradient, rtol=0, atol=1e-6)


def test_eomt_query_loss_keeps_dense_reductions_finite_under_fp16_autocast():
    from eomt import query_loss

    # Match an almost full 448×448 background mask. Its dot product with a
    # positive query is larger than the largest finite FP16 number.
    masks = torch.tensor([1., -1.]).reshape(1, 2, 1, 1).expand(-1, -1, 4, 4).clone()
    classes = torch.tensor([[[2., -1., 0.], [-1., 2., 0.]]])
    target = torch.zeros(1, 448, 448, dtype=torch.long)
    target[:, 32:48, 32:48] = 1
    target[:, :2] = 255
    masks.requires_grad_()
    classes.requires_grad_()
    reference = query_loss(masks, classes, target)
    reference_gradients = torch.autograd.grad(reference, (masks, classes))
    half_inputs = tuple(t.detach().half().requires_grad_() for t in (masks, classes))
    probe = torch.ones(1, 1)
    try:
        with torch.autocast("cpu", dtype=torch.float16):
            active = (probe @ probe).dtype == torch.float16
    except (RuntimeError, TypeError):
        active = False
    if not active:
        pytest.skip("PyTorch CPU FP16 autocast is unavailable")
    with torch.autocast("cpu", dtype=torch.float16):
        actual = query_loss(*half_inputs, target)
    actual_gradients = torch.autograd.grad(actual, half_inputs)
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=0)
    for gradient, expected in zip(actual_gradients, reference_gradients):
        torch.testing.assert_close(gradient.float(), expected, atol=1e-4, rtol=1e-3)
