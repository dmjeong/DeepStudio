"""Native CrackSegNet Studio contracts; synthetic masks are NOT accuracy benchmarks."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "gui"), str(ROOT)]
torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
ort.disable_telemetry_events()
from builtin_models import build_builtin_model, make_builtin_checkpoint, load_builtin_checkpoint
from cracksegnet import DeformConv2d, LocalGraphAttention
from onnx_session import cpu_session_options


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(4)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("offset_scale", [0., .08])
def test_deformable_sampling_values_and_gradients(offset_scale):
    from torchvision.ops import deform_conv2d
    model = DeformConv2d(3, 5)
    with torch.no_grad():
        model.offset.weight.normal_(0, offset_scale)
        model.offset.bias.normal_(0, offset_scale)
    x = torch.randn(2, 3, 13, 17, requires_grad=True)
    actual = model(x)
    expected = deform_conv2d(x, model.offset(x), model.weight, model.bias, padding=(1, 1))
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=2e-5)
    variables = (x, model.weight, model.bias, model.offset.weight, model.offset.bias)
    a = torch.autograd.grad(actual.square().sum(), variables, retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), variables)
    for left, right in zip(a, b):
        assert torch.isfinite(left).all()
        # At exactly integer sampling coordinates, floating-point roundoff can
        # choose the other one-sided derivative. Compare gradients off-grid.
        if offset_scale:
            torch.testing.assert_close(left, right, atol=2e-3, rtol=1e-3)
    assert a[3].abs().sum() > 0  # learned positions really train


def test_graph_attention_has_no_wraparound_or_padded_neighbors():
    layer = LocalGraphAttention(4, 4).eval()
    with torch.no_grad():
        layer.source_score.weight.zero_()
        layer.target_score.weight.zero_()
        layer.edge_score.weight.zero_()
        x = torch.randn(1, 4, 2, 3)
        projected = layer.project(x)
        expected = torch.empty_like(projected)
        for y in range(2):
            for z in range(3):
                values = [projected[:, :, y + dy, z + dx] for dy, dx in layer.OFFSETS
                          if 0 <= y + dy < 2 and 0 <= z + dx < 3]
                expected[:, :, y, z] = torch.relu(layer.norm(torch.stack(values).mean(0)))
        torch.testing.assert_close(layer(x), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("channels,size", [(1, (32, 32)), (3, (75, 110))])
def test_odd_shapes_amp_backward_checkpoint_and_pretrained_rejection(channels, size):
    model = build_builtin_model("cracksegnet", 3, channels)
    x = torch.randn(1, channels, *size)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.float(), torch.randint(0, 3, (1, *size)))
    loss.backward()
    assert logits.shape == (1, 3, *size)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    checkpoint = make_builtin_checkpoint("cracksegnet", model, num_classes=3,
                                         input_size=size, in_channels=channels)
    assert checkpoint["weight_provenance"]["official_implementation"] is False
    with torch.no_grad():
        torch.testing.assert_close(load_builtin_checkpoint(checkpoint)(x), model(x), atol=0, rtol=0)
    with pytest.raises(ValueError, match="사전학습"):
        build_builtin_model("cracksegnet", 3, channels, pretrained=True)


def test_ui_training_modes_catalog_and_packaging(tmp_path):
    from PySide6.QtWidgets import QApplication
    from core.project import ProjectManager
    from core.training_modes import training_capabilities
    from widgets.training_widget import TrainingWidget
    from tools.export_model_catalog import build_catalog
    app = QApplication.instance() or QApplication([])
    widget = TrainingWidget()
    try:
        widget.set_project(ProjectManager.create_new("seg", "segment", str(tmp_path / "p"),
                                                     ["background", "crack", "fm"]))
        index = widget.model_id_combo.findData("cracksegnet")
        assert index >= 0
        widget.model_id_combo.setCurrentIndex(index)
        modes = {widget.mode_combo.itemData(i) for i in range(widget.mode_combo.count())}
        assert modes == {"builtin_scratch", "builtin_transfer", "builtin_resume"}
        assert training_capabilities("segment", "builtin_scratch", model_id="cracksegnet")["patch_training"]
        entry = next(m for m in build_catalog()["models"] if m["model_id"] == "cracksegnet")
        assert not entry["pretrained_asset"]
        assert "cracksegnet" in (ROOT / "gui/build_exe.py").read_text()
    finally:
        widget.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("teacher_id,student_id", [("unet_lite", "cracksegnet"), ("cracksegnet", "unet_lite")])
def test_can_be_teacher_or_student(tmp_path, teacher_id, student_id):
    from segmentation_distillation import SegmentationDistiller
    teacher = build_builtin_model(teacher_id, 3, 3).eval()
    path = tmp_path / "teacher.pt"
    torch.save(make_builtin_checkpoint(teacher_id, teacher, num_classes=3, input_size=64,
                                       class_names=["background", "crack", "fm"]), path)
    distiller = SegmentationDistiller(path, ["background", "crack", "fm"], 3, "cpu")
    student = build_builtin_model(student_id, 3, 3)
    x = torch.randn(2, 3, 64, 64)
    mask = torch.randint(0, 3, (2, 64, 64))
    mask[:, :3] = 255
    y = student(x)
    ce = torch.nn.functional.cross_entropy(y, mask, ignore_index=255)
    loss = distiller(x, y, mask, ce)
    assert torch.isfinite(loss)
    loss.backward()
    assert student.head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in distiller.teacher.parameters())


@pytest.mark.parametrize("channels,size", [(1, (256, 256)), (3, (257, 385))])
def test_deployment_at_full_and_odd_window_sizes(tmp_path, channels, size):
    from export_onnx import export_checkpoint
    model = build_builtin_model("cracksegnet", 3, channels).eval()
    source = tmp_path / "model.pt"
    torch.save(make_builtin_checkpoint("cracksegnet", model, num_classes=3, in_channels=channels,
                                       input_size=size, class_names=["background", "crack", "fm"]), source)
    export_checkpoint(source, tmp_path / "model.onnx", log=lambda _: None)
    session = ort.InferenceSession(str(tmp_path / "model.onnx"), cpu_session_options(num_threads=2))
    x = torch.randn(1, channels, *size)
    with torch.no_grad():
        expected = model(x).numpy()
    actual = session.run(None, {session.get_inputs()[0].name: x.numpy()})[0]
    assert actual.shape == (1, 3, *size)
    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not present")
def test_cuda_amp_backward():
    model = build_builtin_model("cracksegnet", 3, 3).cuda().train()
    x = torch.randn(2, 3, 64, 96, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(x).float().square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def dataset(root):
    rng = np.random.default_rng(44)
    for split in ("train", "val"):
        for folder in ("images", "masks"):
            (root / folder / split).mkdir(parents=True)
        for i in range(4):
            image = rng.integers(70, 95, (64, 96, 3), dtype=np.uint8)
            mask = Image.new("L", (96, 64))
            draw = ImageDraw.Draw(mask)
            draw.line([(9 + i * 3, 9), (34, 28 + i), (17 + i, 56)], fill=1, width=3)
            draw.rectangle((55 + i, 20, 67 + i, 36), fill=2)
            labels = np.asarray(mask).copy()
            image[labels == 1] = (240, 225, 210)
            image[labels == 2] = (210, 45, 30)
            labels[:2] = 255
            Image.fromarray(image).save(root / "images" / split / f"{i}.png")
            Image.fromarray(labels).save(root / "masks" / split / f"{i}.png")
    return root


def test_training_deployment_precisions_resume_transfer_and_cpp(tmp_path):
    from train_builtin import train_builtin
    from export_onnx import export_checkpoint, deployment_preview
    from quantize_onnx import quantize_segment, preprocess
    from core.segment_defects import save_model_rules
    from model_crypto import create_key, read_key, encrypt_config
    import cv2
    import onnx

    data = dataset(tmp_path / "data")
    common = dict(num_classes=3, input_size=64, in_channels=3, batch_size=2,
                  learning_rate=2e-3, num_workers=0, class_weights="sqrt", scheduler_name="none",
                  horizontal_flip=0, rotation=0, color_jitter=0,
                  class_names=["background", "crack", "fm"], log=lambda _: None)
    best = train_builtin("cracksegnet", data, epochs=40, output_dir=tmp_path / "train",
                         original_size_training=True, **common)
    checkpoint = torch.load(best, weights_only=True, map_location="cpu")
    model = load_builtin_checkpoint(checkpoint).eval()
    history = json.loads((tmp_path / "train/training.json").read_text())
    assert history["completed_epochs"] == 40
    assert history["metrics_history"][-1]["train_loss"] < history["metrics_history"][0]["train_loss"]
    save_model_rules(best, {"min_area": 1, "min_score": 0.0})
    exported = tmp_path / "model.onnx"
    export_checkpoint(best, exported, dynamic_batch=True, log=lambda _: None)
    cfg = json.loads(exported.with_suffix(".json").read_text())
    assert cfg["cpp_supported"] and cfg["preprocessing"]["whole_image"]["region"] == [96, 64]
    assert cfg["preprocessing"]["whole_image"]["calibration"] == "full_input"
    graph = onnx.load(exported)
    assert any(n.op_type == "GridSample" for n in graph.graph.node)
    assert all(n.domain in ("", "ai.onnx") for n in graph.graph.node)
    session = ort.InferenceSession(str(exported), cpu_session_options(num_threads=2))
    frame = cv2.imread(str(data / "images/val/0.png"))
    tensor = preprocess(cfg, frame)
    with torch.no_grad():
        expected = model(torch.from_numpy(tensor)).numpy()
    actual = session.run(None, {cfg["input_name"]: tensor})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)
    labels = np.asarray(Image.open(data / "masks/val/0.png"))
    prediction = actual.argmax(1)[0]
    for class_id in (1, 2):
        assert np.mean(prediction[labels == class_id] == class_id) > .5, "foreground must actually be learned"
    double = np.concatenate((tensor, tensor * .7), 0)
    assert session.run(None, {cfg["input_name"]: double})[0].shape == (2, 3, 64, 96)
    with pytest.raises(ValueError, match="opset 16"):
        export_checkpoint(best, tmp_path / "unsupported.onnx", opset_version=15, log=lambda _: None)

    fp16_path = tmp_path / "fp16.onnx"
    fp16 = export_checkpoint(best, fp16_path, precision="float16",
                             validation_dir=data / "images/val", log=lambda _: None)
    assert fp16["verification"] == "fp16_compared"
    half_cfg = json.loads(fp16_path.with_suffix(".json").read_text())
    assert half_cfg["export"]["fp32_preserved"] == "cracksegnet_deformable_sampling"
    int8_config = quantize_segment(exported.with_suffix(".json"), data / "images/val",
                                    count=4, log=lambda _: None)
    quantized = json.loads(int8_config.read_text())
    assert quantized["verification"] == "int8_compared"
    assert any(n.op_type == "QuantizeLinear" for n in onnx.load(int8_config.with_suffix(".onnx")).graph.node)
    for config_path in (fp16_path.with_suffix(".json"), int8_config):
        converted_cfg = json.loads(config_path.read_text())
        converted = ort.InferenceSession(str(config_path.parent / converted_cfg["model_path"]),
                                        cpu_session_options(num_threads=2))
        converted_mask = converted.run(None, {converted_cfg["input_name"]: tensor})[0].argmax(1)[0]
        for class_id in (1, 2):
            # A successful file conversion must not hide an all-background or
            # missing-class prediction on the synthetic training smoke data.
            assert np.mean(converted_mask[labels == class_id] == class_id) > .5

    # Resume optimizer/scheduler and change the head for a different class set.
    train_builtin("cracksegnet", data, epochs=41, output_dir=tmp_path / "resume",
                  resume=tmp_path / "train/last.pt", original_size_training=True, **common)
    resumed = json.loads((tmp_path / "resume/training.json").read_text())
    assert resumed["completed_epochs"] == 41 and len(resumed["metrics_history"]) == 41
    transfer = dict(common, num_classes=4, class_names=["background", "scratch", "fm", "tray"])
    changed = train_builtin("cracksegnet", data, epochs=1, output_dir=tmp_path / "transfer",
                            initial_weights=best, patch_training=True, patches_per_image=2, **transfer)
    assert torch.load(changed, weights_only=True)["num_classes"] == 4
    assert deployment_preview(changed)["vit"]  # export must retain training-window context
    tiled = tmp_path / "tiled.onnx"
    export_checkpoint(changed, tiled, log=lambda _: None)
    tiled_cfg = json.loads(tiled.with_suffix(".json").read_text())
    assert tiled_cfg["preprocessing"].get("tiling") and not tiled_cfg["preprocessing"].get("whole_image")
    from core.inference_loading import load_inference_engine
    for checkpoint_path in (best, changed):
        engine = load_inference_engine(checkpoint_path, device="cpu", gradcam=False)
        for filename in ("0.png", "1.png"):
            image_path = data / "images/val" / filename
            result = engine.infer(str(image_path))
            assert result.status == "ok", result.error
            assert result.task == "segment"
            np.testing.assert_array_equal(engine._original_preview_rgb, np.asarray(Image.open(image_path)))
            assert engine._current_preview_rgb.shape == (64, 96, 3)

    # A native ORT runtime distinct from the Python wheel exercises ALL three
    # precisions and their encrypted packages without OpenCV/eVision dependency.
    probe = os.environ.get("DVS_ONNX_TENSOR_PROBE")
    if probe:
        key = create_key(tmp_path / "model.key")
        tensor.astype(np.float32).tofile(tmp_path / "input.f32")
        for config in (exported.with_suffix(".json"), fp16_path.with_suffix(".json"), int8_config):
            config_data = json.loads(config.read_text())
            expected_session = ort.InferenceSession(str(config.parent / config_data["model_path"]),
                                                    cpu_session_options(num_threads=1))
            reference = expected_session.run(None, {config_data["input_name"]: tensor})[0]
            encrypted = config.with_suffix(".dvsenc")
            encrypt_config(config, encrypted, read_key(key))
            outputs = []
            for source in (config, encrypted):
                result = subprocess.run([probe, str(source), str(tmp_path / "input.f32"),
                                         str(tmp_path / "output.f32"), str(key)],
                                        capture_output=True, text=True, timeout=90)
                assert result.returncode == 0, result.stderr
                output = np.fromfile(tmp_path / "output.f32", np.float32).reshape(reference.shape)
                np.testing.assert_allclose(output, reference, atol=2e-3, rtol=2e-3)
                outputs.append(output)
            np.testing.assert_array_equal(*outputs)

    # Full image SDK path, including normalization, mask decoding and rules.
    # Enabled in Linux native CI, which supplies OpenCV and the built SDK.
    image_probe = os.environ.get("SEGMENT_PROBE")
    if image_probe:
        for config in (exported.with_suffix(".json"), fp16_path.with_suffix(".json"), int8_config,
                       tiled.with_suffix(".json")):
            from dense_deployment import predict_dense_maps
            config_data = json.loads(config.read_text())
            expected_session = ort.InferenceSession(str(config.parent / config_data["model_path"]),
                                                    cpu_session_options(num_threads=1))
            def run(t):
                return [expected_session.run(None, {config_data["input_name"]: t})]
            if config == tiled.with_suffix(".json"):
                expected_mask = predict_dense_maps(config_data, frame, run)[0].argmax(0)
            else:
                expected_mask = expected_session.run(None, {config_data["input_name"]: tensor})[0][0].argmax(0)
            # Use unfiltered masks for parity (the native SDK also applies rules).
            config_data["postprocessing"]["defect_rules"] = {"min_area": 1, "min_score": 0.0}
            raw_config = config.with_name(config.stem + "-raw.json")
            raw_config.write_text(json.dumps(config_data))
            result = subprocess.run([image_probe, str(raw_config), str(data / "images/val/0.png"),
                                     str(tmp_path / "native-mask.png")], capture_output=True, text=True, timeout=90)
            assert result.returncode == 0, result.stderr
            np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / "native-mask.png")), expected_mask)
