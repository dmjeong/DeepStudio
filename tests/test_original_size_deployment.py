"""Original-size checkpoints retain the training region in app and ONNX inference."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gui"), str(ROOT / "python")]
torch = pytest.importorskip("torch")
REGION = (73, 45)
PADDED = (64, 96)


@pytest.fixture
def original_checkpoint(tmp_path):
    from builtin_models import build_builtin_model, make_builtin_checkpoint
    from checkpoint import record_original_size_training

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        model = build_builtin_model("unet_lite", 2, 3).eval()
    payload = make_builtin_checkpoint("unet_lite", model, num_classes=2, input_size=PADDED,
                                      in_channels=3, class_names=["background", "spot"])
    crop = dict(zip(("width", "height"), REGION))
    payload["center_crop"] = crop
    payload["preprocessing"]["center_crop"] = crop
    record_original_size_training(payload, REGION)
    path = tmp_path / "best.pt"
    torch.save(payload, path)
    return path


@pytest.fixture
def image_path(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    pixels = np.random.default_rng(8).integers(0, 256, (59, 103, 3), dtype=np.uint8)
    path = images / "part.png"
    Image.fromarray(pixels).save(path)
    return path


def test_app_uses_original_pixels_and_rejects_a_different_region(original_checkpoint, image_path):
    from core.inference_loading import load_inference_engine

    engine = load_inference_engine(original_checkpoint)
    samples = []
    hook = engine.model.register_forward_pre_hook(lambda _model, args: samples.append(args[0].clone()))
    try:
        result = engine.infer(str(image_path))
    finally:
        hook.remove()
    assert result.status == "ok", result.error
    assert result.details["crop_box"] == [15, 7, 88, 52]
    assert sum(result.details["pixel_counts"].values()) == REGION[0] * REGION[1]
    assert result.details["original_size_region"] == list(REGION)
    assert "tiling" not in result.details
    tensor, = samples
    assert tensor.shape == (1, 3, *PADDED)
    assert torch.count_nonzero(tensor[:, :, 45:]) == 0
    assert torch.count_nonzero(tensor[:, :, :, 73:]) == 0
    with Image.open(image_path) as image:
        rgb = np.asarray(image)[7:52, 15:88]
    mean, std = np.asarray([.485, .456, .406], np.float32), np.asarray([.229, .224, .225], np.float32)
    expected = (rgb.astype(np.float32) / 255 - mean) / std
    np.testing.assert_allclose(tensor[0, :, :45, :73].numpy(), expected.transpose(2, 0, 1), atol=1e-6)

    engine.configure_input_region({"mode": "full"})
    mismatch = engine.infer(str(image_path))
    assert mismatch.status == "error" and "73×45" in mismatch.error and "103×59" in mismatch.error
    cropped_path = image_path.with_name("cropped.png")
    Image.fromarray(rgb).save(cropped_path)
    region_input = engine.infer(str(cropped_path))
    assert region_input.status == "ok" and region_input.details["pixel_counts"] == result.details["pixel_counts"]


def test_export_fixes_original_region_and_runs_fp32_and_int8(original_checkpoint, image_path, tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxruntime.quantization")
    from export_onnx import deployment_preview, export_checkpoint
    from quantize_onnx import judgement_region, preprocess

    preview = deployment_preview(original_checkpoint)
    assert preview["original_size_region"] == REGION and not preview["patch"]
    with pytest.raises(ValueError, match="73×45"):
        export_checkpoint(original_checkpoint, tmp_path / "invalid.onnx", overrides={"region_size": [75, 45]},
                          log=lambda _message: None)
    assert not (tmp_path / "invalid.onnx").exists()
    for precision in ("float32", "int8"):
        output = tmp_path / f"{precision}.onnx"
        result = export_checkpoint(original_checkpoint, output, precision=precision, calibration_dir=image_path.parent,
                                   log=lambda _message: None)
        config = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
        assert config["cpp_supported"] and config["export"]["precision"] == precision
        whole = config["preprocessing"]["whole_image"]
        assert whole["region"] == whole["scaled"] == list(REGION)
        assert whole["input"] == list(PADDED) and whole["pad"] == "zero_after_normalize"
        assert "tiling" not in config["preprocessing"]
        graph = onnx.load(str(output))
        dims = [d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]
        assert dims == [1, 3, *PADDED]

        with Image.open(image_path) as image:
            rgb = np.asarray(image)
        region = judgement_region(config, rgb[:, :, ::-1])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
        result_tensor, = session.run(None, {config["input_name"]: preprocess(config, region)})
        assert result_tensor.shape == (1, 2, *PADDED) and np.isfinite(result_tensor).all()

        if probe := os.environ.get("SEGMENT_PROBE"):
            mask_path = tmp_path / f"{precision}.png"
            subprocess.run([probe, str(output.with_suffix(".json")), str(image_path), str(mask_path)],
                           check=True, capture_output=True, timeout=60)
            with Image.open(mask_path) as image:
                mask = np.asarray(image)
            assert mask.shape == rgb.shape[:2]
            np.testing.assert_array_equal(mask[7:52, 15:88], result_tensor[0, :, :45, :73].argmax(0))
            assert not mask[:7].any() and not mask[52:].any()
            assert not mask[:, :15].any() and not mask[:, 88:].any()
