"""Exercise native UI -> train -> checkpoint -> ONNX -> deployed segmentation."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "gui")]
torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
# Disable at collection time, before any test/import can create an ORT session.
# ORT 1.29 macOS's telemetry HTTP worker can otherwise race process shutdown.
ort.disable_telemetry_events()
pytest.importorskip("timm")
from segmentation_models import MODEL_IDS, SCRATCH_IDS, boundary_loss


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(4)
    yield
    torch.set_num_threads(old)


def dataset(root):
    rng = np.random.default_rng(4)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "masks" / split).mkdir(parents=True)
        for i in range(3):
            image = rng.integers(50, 120, (64, 96, 3), dtype=np.uint8)
            mask = np.zeros((64, 96), np.uint8)
            image[20:30, 25:55] = 230
            mask[20:30, 25:55] = 1
            mask[:2] = 255
            Image.fromarray(image).save(root / "images" / split / f"{i}.png")
            Image.fromarray(mask).save(root / "masks" / split / f"{i}.png")
    return root


@pytest.mark.parametrize("model_id", MODEL_IDS)
@pytest.mark.parametrize("channels", (1, 3))
def test_odd_size_single_batch_backward_and_checkpoint(model_id, channels):
    from builtin_models import (
        build_builtin_model,
        make_builtin_checkpoint,
        load_builtin_checkpoint,
    )
    from segmentation_loss import SegmentationLoss
    from train_builtin import _segmentation_epoch

    model = build_builtin_model(model_id, 3, channels)
    images = torch.randn(1, channels, 75, 110)
    masks = torch.randint(0, 3, (1, 75, 110))
    masks[:, :3] = 255
    before = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if "head" in n or "boundary" in n
    }
    loss, metric = _segmentation_epoch(
        model,
        [(images, masks)],
        SegmentationLoss(3, loss="ce_dice"),
        torch.optim.Adam(model.parameters(), lr=1e-4),
    )
    assert np.isfinite([loss, metric]).all()
    assert any(
        not torch.equal(before[n], p)
        for n, p in model.named_parameters()
        if n in before
    )
    if model_id in SCRATCH_IDS:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for n, p in model.named_parameters()
            if "aux" in n
        )
    if model_id == "stdc1":
        assert model.boundary.conv_out.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        expected = model(images)
        checkpoint = make_builtin_checkpoint(
            model_id, model, num_classes=3, in_channels=channels
        )
        restored = load_builtin_checkpoint(checkpoint).eval()
        assert expected.shape == (1, 3, 75, 110)
        torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)
    if model_id in SCRATCH_IDS:
        # Evaluation/best selection must not run auxiliary or boundary heads.
        def unexpected_auxiliary(*_):
            raise AssertionError("Training-only head executed during evaluation")

        handles = [
            child.register_forward_pre_hook(unexpected_auxiliary)
            for name, child in model.named_modules()
            if name.startswith(("aux", "net.aux", "boundary"))
        ]
        try:
            _segmentation_epoch(model, [(images, masks)], SegmentationLoss(3))
        finally:
            for handle in handles:
                handle.remove()


def test_boundary_supervision_ignores_class_numbers_and_excluded_pixels():
    masks = torch.zeros(1, 20, 24, dtype=torch.long)
    masks[:, 5:14, 6:16] = 1
    masks[:, :3] = 255
    logits = torch.randn(1, 1, 20, 24, requires_grad=True)
    torch.testing.assert_close(
        boundary_loss(logits, masks),
        boundary_loss(logits, masks.masked_fill(masks == 1, 2)),
    )
    ignored = boundary_loss(logits, torch.full_like(masks, 255))
    assert ignored == 0
    ignored.backward()
    assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_qt_model_modes_and_packaged_catalog(model_id, tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from core.project import ProjectManager
    from widgets.training_widget import TrainingWidget
    from core.training_modes import training_capabilities
    from tools.export_model_catalog import build_catalog

    app = QApplication.instance() or QApplication([])
    project = ProjectManager.create_new(
        "seg", "segment", str(tmp_path / "p"), ["background", "crack"]
    )
    widget = TrainingWidget()
    try:
        widget.set_project(project)
        index = widget.model_id_combo.findData(model_id)
        assert index >= 0
        widget.model_id_combo.setCurrentIndex(index)
        modes = [
            widget.mode_combo.itemData(i) for i in range(widget.mode_combo.count())
        ]
        assert "builtin_scratch" in modes and "builtin_transfer" in modes
        assert ("builtin_finetune" in modes) == (model_id not in SCRATCH_IDS)
        assert training_capabilities("segment", "builtin_scratch", model_id=model_id)[
            "patch_training"
        ]
        assert any(s["model_id"] == model_id for s in build_catalog()["models"])
    finally:
        widget.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_training_export_quantization_and_cpp(model_id, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    ort.disable_telemetry_events()
    from train_builtin import train_builtin
    from builtin_models import load_builtin_checkpoint
    from export_onnx import export_checkpoint
    from quantize_onnx import quantize_segment
    from onnx_session import cpu_session_options

    data = dataset(tmp_path / "data")
    best = train_builtin(
        model_id,
        data,
        num_classes=2,
        input_size=64,
        in_channels=3,
        epochs=1,
        batch_size=1,
        learning_rate=1e-4,
        num_workers=0,
        patch_training=True,
        patches_per_image=2,
        class_weights="sqrt",
        class_names=["background", "crack"],
        output_dir=tmp_path / "train",
        log=lambda _: None,
    )
    checkpoint = torch.load(best, weights_only=True, map_location="cpu")
    model = load_builtin_checkpoint(checkpoint).eval()
    assert checkpoint["model_id"] == model_id
    assert (tmp_path / "train" / "training.json").is_file()
    exported = tmp_path / "model.onnx"
    from core.segment_defects import save_model_rules

    save_model_rules(best, {"min_area": 1, "min_score": 0.0})
    export_checkpoint(best, exported, log=lambda _: None)
    import onnx

    graph = onnx.load(exported)
    assert len(graph.graph.output) == 1
    assert not any(
        "aux" in node.name or "boundary" in node.name for node in graph.graph.node
    )
    config_path = exported.with_suffix(".json")
    config = json.loads(config_path.read_text())
    assert config["cpp_supported"]
    assert config["preprocessing"]["whole_image"]["region"] == [96, 64]
    session = ort.InferenceSession(
        str(exported),
        cpu_session_options(num_threads=2),
        providers=["CPUExecutionProvider"],
    )
    x = np.random.default_rng(4).standard_normal((1, 3, 64, 96), dtype=np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    actual = session.run(None, {config["input_name"]: x})[0]
    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=5e-4)
    quantized = quantize_segment(
        config_path, data / "images" / "val", count=1, tile=64, log=lambda _: None
    )
    qconfig = json.loads(quantized.read_text())
    qsession = ort.InferenceSession(
        str(quantized.parent / qconfig["model_path"]),
        cpu_session_options(num_threads=2),
        providers=["CPUExecutionProvider"],
    )
    assert np.isfinite(qsession.run(None, {qconfig["input_name"]: x})[0]).all()
    # Multiple output heads must all be reset when class names/count change.
    transferred = train_builtin(
        model_id,
        data,
        num_classes=3,
        input_size=64,
        epochs=1,
        batch_size=1,
        learning_rate=1e-4,
        num_workers=0,
        initial_weights=best,
        class_names=["background", "scratch", "fm"],
        output_dir=tmp_path / "transfer",
        log=lambda _: None,
    )
    assert torch.load(transferred, weights_only=True)["num_classes"] == 3
    probe = os.environ.get("SEGMENT_PROBE")
    if probe:
        result = subprocess.run(
            [
                probe,
                str(config_path),
                str(data / "images" / "val" / "0.png"),
                str(tmp_path / "mask.png"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        from quantize_onnx import preprocess
        import cv2

        frame = cv2.imread(str(data / "images" / "val" / "0.png"))
        raw = session.run(None, {config["input_name"]: preprocess(config, frame)})[0]
        np.testing.assert_array_equal(
            np.asarray(Image.open(tmp_path / "mask.png")), raw.argmax(1)[0]
        )
        from model_crypto import create_key, encrypt_config, read_key

        key = create_key(tmp_path / "model.key")
        encrypted = tmp_path / "model.dvsenc"
        encrypt_config(config_path, encrypted, read_key(key))
        sealed = subprocess.run(
            [
                probe,
                str(encrypted),
                str(data / "images" / "val" / "0.png"),
                str(tmp_path / "encrypted-mask.png"),
                "1",
                "1",
                str(key),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert sealed.returncode == 0, sealed.stderr
        assert sealed.stdout == result.stdout
        np.testing.assert_array_equal(
            np.asarray(Image.open(tmp_path / "mask.png")),
            np.asarray(Image.open(tmp_path / "encrypted-mask.png")),
        )


@pytest.mark.parametrize(
    "model_id,asset",
    (
        ("unet_mobilenetv4_small", "mobilenetv4_small_encoder"),
        ("unet_efficientnet_b1", "efficientnet_b1_encoder"),
        ("unet_shufflenetv2", "shufflenet_v2_x1_0"),
    ),
)
def test_official_pretrained_encoder_tensor_is_used(model_id, asset):
    from builtin_assets import builtin_asset_path
    from builtin_models import build_builtin_model

    try:
        path = builtin_asset_path(asset)
    except RuntimeError:
        pytest.skip(
            "Official backbone file not installed; build-time download smoke covers this separately"
        )
    model = build_builtin_model(model_id, 3, 3, pretrained=True).eval()
    source = torch.load(path, map_location="cpu", weights_only=True)
    if model_id == "unet_shufflenetv2":
        actual, expected = model.encoder.layers[0][0].weight, source["conv1.0.weight"]
    else:
        actual, expected = (
            model.encoder.base.conv_stem.weight,
            source["conv_stem.weight"],
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert model.weight_provenance["source"] == "imagenet"
