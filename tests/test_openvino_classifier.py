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
