"""Real OpenVINO CPU inference, shared preprocessing and reusable output safety."""
import json

import numpy as np
import pytest

ov = pytest.importorskip("openvino")
torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from checkpoint import make_checkpoint_metadata
from efficientnet import EfficientNet
from export_onnx import export_checkpoint
from onnx_classifier import OnnxClassifier
from openvino_classifier import OpenVINOClassifier


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("efficientnet-openvino")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    result = {}
    for channels in (1, 3):
        torch.manual_seed(17)
        model = EfficientNet("efficientnet_b0", 3, channels).eval()
        # Nontrivial BN state exercises fusion; random initialization alone is near zero.
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean.uniform_(-.1, .1)
                module.running_var.uniform_(.8, 1.2)
        checkpoint = make_checkpoint_metadata("classify", 3, ["a", "b", "c"], (32, 48), channels,
                                              center_crop={"height": 40, "width": 50})
        checkpoint.update(engine="efficientnet", model_config=model.checkpoint_config(), model_state_dict=model.state_dict())
        source = root / f"source-{channels}.pt"
        torch.save(checkpoint, source)
        config = export_checkpoint(source, root / f"model-{channels}.onnx", dynamic_batch=True, log=lambda _: None)["config_path"]
        result[channels] = (config, model)
    yield result
    torch.set_num_threads(previous)


@pytest.mark.parametrize("channels", [1, 3])
def test_real_parity_preprocessing_and_output_ownership(exported, channels):
    config, model = exported[channels]
    ort = OnnxClassifier(config, num_threads=1)
    runtime = OpenVINOClassifier(config, num_threads=1)
    rgb = np.random.default_rng(7).integers(0, 256, (45, 67, 3), dtype=np.uint8)
    tensor = runtime.preprocess(rgb)
    np.testing.assert_array_equal(tensor, ort.preprocess(rgb))
    with torch.inference_mode():
        expected = model(torch.from_numpy(tensor)).numpy()
    actual = runtime.logits(tensor)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    saved = actual.copy()
    runtime.logits(tensor * .25)
    np.testing.assert_array_equal(actual, saved)
    prediction = runtime.predict_rgb(rgb)
    assert prediction["class_id"] == int(expected.argmax(1)[0])
    assert sum(prediction["probabilities"]) == pytest.approx(1)
    assert "f32" in runtime.runtime_settings["INFERENCE_PRECISION_HINT"] or "float32" in runtime.runtime_settings["INFERENCE_PRECISION_HINT"]
    assert int(runtime.runtime_settings["NUM_STREAMS"]) == 1
    if channels == 1:
        gray = np.zeros((45, 67), dtype=np.uint8)
        assert runtime.predict_gray(gray)["class_id"] == ort.predict_gray(gray)["class_id"]
    with pytest.raises(ValueError, match="float32 입력"):
        runtime.logits(tensor.astype(np.float64))
    with pytest.raises(ValueError, match="float32 입력"):
        runtime.logits(np.repeat(tensor, 2, axis=0))


@pytest.mark.parametrize("change", [{"input_name": "wrong"}, {"output_name": "wrong"},
                                   {"num_classes": 4, "class_names": ["a", "b", "c", "d"]},
                                   {"input_height": 64}])
def test_manifest_mismatch_rejected(exported, change):
    from pathlib import Path
    config, _ = exported[1]
    source = Path(config)
    path = source.parent / "mismatch.json"
    path.write_text(json.dumps({**json.loads(source.read_text()), **change}))
    with pytest.raises(ValueError):
        OpenVINOClassifier(path, num_threads=1)


def test_invalid_thread_count_rejected(exported):
    for threads in (-1, True, 1.5):
        with pytest.raises(ValueError, match="스레드"):
            OpenVINOClassifier(exported[1][0], num_threads=threads)


def test_precision_benchmark_runs_same_efficientnet_graphs_on_both_cpu_runtimes(exported, tmp_path):
    import cv2
    from precision_benchmark import benchmark_precisions

    config, _ = exported[1]
    images = tmp_path / "시험"
    images.mkdir()
    rng = np.random.default_rng(42)
    for index in range(3):
        pixels = rng.integers(0, 256, (52, 67), dtype=np.uint8)
        (images / f"영상-{index}.png").write_bytes(cv2.imencode(".png", pixels)[1].tobytes())
    rows = benchmark_precisions(config, images, thread_counts=(1,), calibration_dir=images,
                                backends=("openvino", "onnxruntime"), warmup=1, repeats=1, log=lambda _: None)
    assert [(row["precision"], row["backend"]) for row in rows] == [
        (p, runtime) for p in ("FP32", "FP16", "INT8") for runtime in ("onnxruntime", "openvino")
    ]
    assert all(row["status"] == "완료" for row in rows), rows
    assert all(row["p95_ms"] >= row["model_p95_ms"] > 0 and row["samples"] == 3 for row in rows)
    assert rows[0]["comparison"] == "기준"
    assert "Top-1 변경 0장" in rows[1]["comparison"]
    assert "실제 스레드 1" in rows[1]["detail"]
    assert all("확률차 최대" in row["comparison"] for row in rows[2:])


def test_precision_benchmark_preserves_ort_results_without_openvino(exported, tmp_path, monkeypatch):
    import sys
    import cv2
    from precision_benchmark import benchmark_precisions

    config, _ = exported[1]
    pixels = np.zeros((52, 67), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "zero.png"), pixels)
    monkeypatch.setitem(sys.modules, "openvino", None)
    result = benchmark_precisions(config, tmp_path, thread_counts=(1,), backends=("onnxruntime", "openvino"),
                                  run_fp16=False, run_int8=False, warmup=0, repeats=1, log=lambda _: None)
    assert result[0]["backend"] == "onnxruntime" and result[0]["status"] == "완료"
    assert result[1]["backend"] == "openvino" and result[1]["status"] == "실행 실패"
    assert "OpenVINO를 불러올 수 없습니다" in result[1]["detail"]
