"""Real published encoders: all tensors, offline restore, training and deployment."""

import json
import os
import subprocess

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
ort.disable_telemetry_events()
pytest.importorskip("timm")

from builtin_assets import BUILTIN_BACKBONE_ASSETS, builtin_asset_path
from builtin_models import (
    build_builtin_model,
    load_builtin_checkpoint,
    make_builtin_checkpoint,
)
from segmentation_models import MODEL_IDS, AUXILIARY_MODEL_IDS
from tests.test_segmentation_candidates import dataset


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(8)
    yield
    torch.set_num_threads(old)


def official(model_id):
    try:
        return builtin_asset_path(BUILTIN_BACKBONE_ASSETS[model_id])
    except RuntimeError:
        if os.environ.get("DVS_REQUIRE_SEGMENT_PRETRAINED") == "1":
            raise
        pytest.skip(
            "Install the five official encoders to run the weight-backed integration checks"
        )


@pytest.mark.parametrize("model_id", sorted(AUXILIARY_MODEL_IDS))
@pytest.mark.parametrize("channels", (1, 3))
def test_official_backbone_complete_and_gray_stems_and_offline_restore(
    model_id, channels, monkeypatch
):
    path = official(model_id)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = build_builtin_model(model_id, 3, channels, pretrained=True).eval()
    if model_id == "stdc1":
        source = payload["state_dict"]
        target = model.cp.backbone.state_dict()
        for key in target:
            canonical = (
                "features." + key.split(".", 2)[2]
                if key.startswith(("x2.", "x4.", "x8.", "x16.", "x32."))
                else key
            )
            expected = source[canonical]
            if channels == 1 and canonical == "features.0.conv.weight":
                expected = expected.sum(1, keepdim=True)
            torch.testing.assert_close(target[key], expected, atol=0, rtol=0)
    else:
        for branch, source in payload.items():
            target = getattr(model.net, branch).state_dict()
            assert set(source) == set(target)
            for key, expected in source.items():
                if channels == 1 and (branch, key) in {
                    ("detail", "S1.0.conv.weight"),
                    ("segment", "S1S2.conv.conv.weight"),
                }:
                    expected = expected.sum(1, keepdim=True)
                torch.testing.assert_close(target[key], expected, atol=0, rtol=0)
    checkpoint = make_builtin_checkpoint(
        model_id, model, num_classes=3, in_channels=channels
    )
    monkeypatch.setattr(
        "builtin_assets.builtin_asset_path",
        lambda _: pytest.fail("restore fetched pretrained weights"),
    )
    restored = load_builtin_checkpoint(checkpoint).eval()
    assert restored.weight_provenance["source"] == "imagenet"
    x = torch.randn(1, channels, 64, 96)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)


@pytest.mark.parametrize("model_id", sorted(AUXILIARY_MODEL_IDS))
def test_missing_or_partial_pretrain_never_falls_back_to_random(
    model_id, monkeypatch, tmp_path
):
    import builtin_assets

    def missing(_):
        raise RuntimeError("missing official weight")

    monkeypatch.setattr(builtin_assets, "builtin_asset_path", missing)
    with pytest.raises(RuntimeError, match="missing official"):
        build_builtin_model(model_id, 3, pretrained=True)
    model = build_builtin_model(model_id, 3)
    if model_id == "stdc1":
        canonical = {
            k: v
            for k, v in model.cp.backbone.state_dict().items()
            if not k.startswith(("x2.", "x4.", "x8.", "x16.", "x32."))
        }
        del canonical["features.1.conv.weight"]
        payload = {"state_dict": canonical}
    else:
        payload = {
            branch: getattr(model.net, branch).state_dict()
            for branch in ("detail", "segment", "bga")
        }
        del payload["bga"]["left1.0.weight"]
    file = tmp_path / "incomplete.pth"
    torch.save(payload, file)
    with pytest.raises(ValueError, match="전체 구조"):
        build_builtin_model(model_id, 3, backbone_weights=file)


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_pretrained_patch_train_export_int8_cpp_and_encryption(model_id, tmp_path):
    official(model_id)
    from train_builtin import train_builtin
    from export_onnx import export_checkpoint
    from onnx_session import cpu_session_options
    from quantize_onnx import quantize_segment, preprocess
    from core.segment_defects import save_model_rules
    from model_crypto import create_key, read_key, encrypt_config
    import cv2

    data = dataset(tmp_path / "data")
    best = train_builtin(
        model_id,
        data,
        num_classes=2,
        class_names=["background", "crack"],
        input_size=64,
        epochs=1,
        batch_size=1,
        learning_rate=1e-4,
        pretrained=True,
        num_workers=0,
        patch_training=True,
        patches_per_image=2,
        class_weights="sqrt",
        output_dir=tmp_path / "train",
        log=lambda _: None,
    )
    checkpoint = torch.load(best, map_location="cpu", weights_only=True)
    assert checkpoint["weight_provenance"]["source"] == "imagenet"
    assert checkpoint["training_config"]["pretrained"] is True
    save_model_rules(best, {"min_area": 1, "min_score": 0.0})
    output = tmp_path / "pretrained.onnx"
    exported = export_checkpoint(best, output, verify=True, log=lambda _: None)
    assert exported["verification"] == "passed"
    config = output.with_suffix(".json")
    manifest = json.loads(config.read_text())
    image = data / "images" / "val" / "0.png"
    x = preprocess(manifest, cv2.imread(str(image)))
    session = ort.InferenceSession(
        str(output),
        cpu_session_options(num_threads=2),
        providers=["CPUExecutionProvider"],
    )
    raw = session.run(None, {manifest["input_name"]: x})[0]
    model = load_builtin_checkpoint(checkpoint).eval()
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(raw, expected, atol=1e-3, rtol=5e-4)
    quantized = quantize_segment(
        config, data / "images" / "val", count=2, tile=64, log=lambda _: None
    )
    qconfig = json.loads(quantized.read_text())
    qsession = ort.InferenceSession(
        str(quantized.parent / qconfig["model_path"]),
        cpu_session_options(num_threads=2),
        providers=["CPUExecutionProvider"],
    )
    assert np.isfinite(qsession.run(None, {qconfig["input_name"]: x})[0]).all()
    probe = os.environ.get("SEGMENT_PROBE")
    if probe:
        plain = subprocess.run(
            [probe, str(config), str(image), str(tmp_path / "plain.png")],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert plain.returncode == 0, plain.stderr
        np.testing.assert_array_equal(
            np.asarray(Image.open(tmp_path / "plain.png")), raw.argmax(1)[0]
        )
        key = create_key(tmp_path / "key")
        package = tmp_path / "pretrained.dvsenc"
        encrypt_config(config, package, read_key(key))
        sealed = subprocess.run(
            [
                probe,
                str(package),
                str(image),
                str(tmp_path / "sealed.png"),
                "1",
                "1",
                str(key),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert sealed.returncode == 0, sealed.stderr
        assert plain.stdout == sealed.stdout
        np.testing.assert_array_equal(
            np.asarray(Image.open(tmp_path / "plain.png")),
            np.asarray(Image.open(tmp_path / "sealed.png")),
        )


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_frozen_pretrained_backbone_preserves_parameters_and_bn(model_id):
    official(model_id)
    from builtin_models import HEAD_PREFIXES
    from train_builtin import _segmentation_epoch
    from segmentation_loss import SegmentationLoss

    model = build_builtin_model(model_id, 3, pretrained=True)
    prefixes = HEAD_PREFIXES[model_id]
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    before = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    stats = {
        n: b.clone() for n, b in model.named_buffers() if not n.startswith(prefixes)
    }
    head_before = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    _segmentation_epoch(
        model,
        [(torch.randn(1, 3, 64, 64), torch.randint(0, 3, (1, 64, 64)))],
        SegmentationLoss(3),
        optimizer,
    )
    assert before and head_before
    for name, parameter in model.named_parameters():
        if name in before:
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
    for name, value in model.named_buffers():
        if name in stats:
            torch.testing.assert_close(value, stats[name], atol=0, rtol=0)
    assert any(
        not torch.equal(p, head_before[n])
        for n, p in model.named_parameters()
        if n in head_before
    )


@pytest.mark.parametrize("model_id", sorted(AUXILIARY_MODEL_IDS))
def test_13_84_single_optimizer_group_resume(model_id, tmp_path, monkeypatch):
    from builtin_models import HEAD_PREFIXES
    from train_builtin import train_builtin

    data = dataset(tmp_path / "data")
    options = dict(
        num_classes=2,
        class_names=["background", "crack"],
        input_size=64,
        batch_size=1,
        num_workers=0,
        output_dir=tmp_path / "train",
        log=lambda _: None,
    )
    with monkeypatch.context() as legacy:
        legacy.setitem(HEAD_PREFIXES, model_id, ("",))
        train_builtin(model_id, data, epochs=1, **options)
    last = tmp_path / "train" / "last.pt"
    assert (
        len(torch.load(last, weights_only=True)["optimizer_state_dict"]["param_groups"])
        == 1
    )
    train_builtin(model_id, data, epochs=2, resume=last, **options)
    assert torch.load(last, weights_only=True)["epoch"] == 1
