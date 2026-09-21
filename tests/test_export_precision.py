"""Real ORT execution for high precision grouped Conv and B1/100px deployment."""
import copy
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch
import onnxruntime as ort

import export_onnx
from checkpoint import make_checkpoint_metadata
from efficientnet import EfficientNet
from efficientnet_precision import _DoubleConv, prepare_precision_export


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("groups,kernel,stride,dilation", [(1, 1, 1, 1), (2, 3, 2, 1), (4, 5, 1, 2)])
def test_double_conv_matches_torch_and_ort(tmp_path, groups, kernel, stride, dilation):
    torch.manual_seed(10)
    conv = torch.nn.Conv2d(4, 8, kernel, stride=stride, padding=kernel // 2,
                           dilation=dilation, groups=groups).double().eval()
    x = torch.randn(2, 4, 13, 15).double() * 1000
    lowered = _DoubleConv(copy.deepcopy(conv))
    with torch.no_grad():
        expected = conv(x).numpy()
        np.testing.assert_allclose(lowered(x).numpy(), expected, rtol=1e-11, atol=1e-10)
    path = tmp_path / "conv.onnx"
    export_onnx.export_to_onnx(lowered, x, path, dynamic_batch=True, constant_folding=False)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input_image": x.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-10)


@pytest.mark.parametrize("corrupt_candidate", [False, True])
def test_b1_100px_precision_fallback_uses_original_reference_and_preserves_failed_output(tmp_path, corrupt_candidate):
    from onnx_classifier import OnnxClassifier
    torch.manual_seed(42)
    model = EfficientNet("efficientnet_b1", 3, 1).eval()
    # Trained-like, nonidentity normalization, including the reported variance.
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.bias.uniform_(-.1, .1)
        model.features[4][0].block[1][1].running_var[0] = 4.328019258537097e-6
    checkpoint = make_checkpoint_metadata("classify", 3, ["a", "b", "c"], (100, 100), 1)
    checkpoint.update(engine="efficientnet", model_config=model.checkpoint_config(), model_state_dict=model.state_dict())
    source, output = tmp_path / "model.pt", tmp_path / "model.onnx"
    torch.save(checkpoint, source)
    output.write_bytes(b"previous")
    real_verify = export_onnx.verify_onnx
    tested = []

    def verify(path, tensor, candidate, **kwargs):
        if candidate.inference_optimization.get("fallback") != "portable_fp64":
            raise ValueError("force final precision candidate")
        tested.append((tensor.shape[0], bool(torch.count_nonzero(tensor) == 0)))
        return real_verify(path, tensor, candidate, **kwargs)

    def factory(reference):
        candidate = prepare_precision_export(reference)
        if corrupt_candidate:
            with torch.no_grad():
                candidate.model.classifier[1].bias.add_(1.)
        return candidate

    with patch.object(export_onnx, "verify_onnx", side_effect=verify), \
            patch("efficientnet_precision.prepare_precision_export", side_effect=factory):
        if corrupt_candidate:
            with pytest.raises(ValueError, match="모두 ONNX 검증 실패"):
                export_onnx.export_checkpoint(source, output, dynamic_batch=True, log=lambda _: None, allow_precision_fallback=True)
            assert output.read_bytes() == b"previous"
            return
        result = export_onnx.export_checkpoint(source, output, dynamic_batch=True, log=lambda _: None, allow_precision_fallback=True)
    assert tested[:3] == [(1, False), (1, True), (2, False)]
    assert tested[-3:] == [(1, False), (1, True), (2, False)]  # Restored winner is reloaded.
    manifest = json.loads(output.with_suffix(".json").read_text())
    assert manifest["export"]["optimization"]["compute_precision"] == "float64"
    assert manifest["onnxruntime"]["graph_optimization_level"] == "disabled"
    classifier = OnnxClassifier(result["config_path"])
    x = torch.zeros(2, 1, 100, 100)
    with torch.no_grad():
        expected = model(x).numpy()
    export_onnx.validate_classification_outputs(expected, classifier.logits(x.numpy()))


def test_app_precision_candidate_is_checked_against_unmodified_model():
    from core.efficientnet_onnx import EfficientNetOnnx
    model = EfficientNet("efficientnet_b1", 3, 1).eval()
    before = copy.deepcopy(model.state_dict())
    real_export = export_onnx.export_to_onnx

    def export(candidate, dummy, path, **kwargs):
        if getattr(candidate, "inference_optimization", {}).get("fallback") != "portable_fp64":
            with torch.no_grad():
                candidate.classifier[1].bias.add_(1.)
        return real_export(candidate, dummy, path, **kwargs)

    with patch.object(export_onnx, "export_to_onnx", side_effect=export):
        runtime = EfficientNetOnnx(model, (100, 100), threads=2, warmup=0)
    assert runtime.export_graph == "portable_fp64"
    assert runtime.optimization_level == "disabled"
    x = torch.zeros(1, 1, 100, 100)
    with torch.no_grad():
        export_onnx.validate_classification_outputs(model(x).numpy(), runtime.logits(x.numpy()))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
