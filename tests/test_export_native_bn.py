"""Exercise trained BN cancellation with real ONNX kernels, not only init weights."""
import copy
import json
from unittest.mock import patch

import numpy as np
import onnx
import pytest
import torch

import export_onnx
from checkpoint import make_checkpoint_metadata
from efficientnet import EfficientNet, _NativeAffineExportBatchNorm, prepare_native_bn_export


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_trained_bn_coefficients_survive_actual_onnx_execution(tmp_path):
    import onnxruntime as ort
    torch.manual_seed(4)
    bn = torch.nn.BatchNorm2d(3).eval()
    bn.running_mean.fill_(1000.)
    bn.running_var.fill_(0.001)
    with torch.no_grad():
        bn.weight.uniform_(0.5, 2.)
        bn.bias.uniform_(-1., 1.)
    sample = torch.randn(1, 3, 8, 8) * .001 + 1000.
    before = copy.deepcopy(bn.state_dict())
    candidate = _NativeAffineExportBatchNorm(bn).eval()
    path = tmp_path / "bn.onnx"
    export_onnx.export_to_onnx(candidate, sample, path, constant_folding=False)
    graph = onnx.load(path)
    assert not any(node.op_type == "BatchNormalization" for node in graph.graph.node)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input_image": sample.numpy()})[0]
    with torch.no_grad():
        expected = bn(sample).numpy()
    export_onnx.validate_outputs(expected, actual, **export_onnx.verification_tolerances("classify"))
    for key, value in bn.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_fusion_cannot_replace_original_reference(tmp_path):
    model = EfficientNet(num_classes=2).eval()
    candidate = prepare_native_bn_export(model)
    with torch.no_grad():
        candidate.classifier[1].bias.add_(1.)
    sample = torch.randn(1, 1, 32, 32)
    path = tmp_path / "changed.onnx"
    export_onnx.export_to_onnx(candidate, sample, path, constant_folding=False)
    # A self-comparison passes, but cannot certify the original checkpoint.
    assert export_onnx.verify_onnx(path, sample, candidate)
    with pytest.raises(ValueError, match="검증 실패"):
        export_onnx.verify_onnx(path, sample, candidate, reference_model=model)


def test_bn_fallback_is_verified_against_original_and_reloaded(tmp_path):
    from onnx_classifier import OnnxClassifier
    model = EfficientNet(num_classes=2).eval()
    checkpoint = make_checkpoint_metadata("classify", 2, ["OK", "NG"], (32, 32), 1)
    checkpoint.update(engine="efficientnet", model_config=model.checkpoint_config(),
                      model_state_dict=model.state_dict())
    source, output = tmp_path / "model.pt", tmp_path / "model.onnx"
    torch.save(checkpoint, source)
    real_verify = export_onnx.verify_onnx
    references, batches = [], []

    def verify(path, probe, candidate, **kwargs):
        references.append(id(kwargs["reference_model"]))
        if candidate.inference_optimization.get("fallback") != "native_batch_norm_affine":
            raise ValueError("force BN fallback")
        batches.append((probe.shape[0], bool(torch.count_nonzero(probe) == 0)))
        return real_verify(path, probe, candidate, **kwargs)

    with patch.object(export_onnx, "verify_onnx", side_effect=verify):
        result = export_onnx.export_checkpoint(source, output, dynamic_batch=True, log=lambda _: None)
    assert len(set(references)) == 1
    assert batches == [(1, False), (1, True), (2, False)]
    config = json.loads(output.with_suffix(".json").read_text())
    assert config["export"]["verification_reference"] == "original_checkpoint_pytorch"
    assert config["export"]["optimization"]["fallback"] == "native_batch_norm_affine"
    assert config["onnxruntime"]["graph_optimization_level"] == "disabled"
    classifier = OnnxClassifier(result["config_path"])
    sample = torch.randn(2, 1, 32, 32)
    with torch.no_grad():
        export_onnx.validate_classification_outputs(model(sample).numpy(), classifier.logits(sample.numpy()))


def test_gui_uses_same_bn_candidate_and_original_reference(tmp_path):
    from core.efficientnet_onnx import EfficientNetOnnx
    real_export = export_onnx.export_to_onnx
    model = EfficientNet(num_classes=2).eval()
    before = copy.deepcopy(model.state_dict())

    def export(candidate, dummy, path, **kwargs):
        if candidate.inference_optimization.get("fallback") != "native_batch_norm_affine":
            with torch.no_grad():
                candidate.classifier[1].bias.add_(1.)
        return real_export(candidate, dummy, path, **kwargs)

    # The unfused copy has no metadata before export.
    model.inference_optimization = {"conv_bn_fused": 0}
    with patch.object(export_onnx, "export_to_onnx", side_effect=export):
        runtime = EfficientNetOnnx(model, (32, 32), threads=1, warmup=0)
    assert runtime.export_graph == "native_batch_norm_affine"
    assert runtime.optimization_level == "disabled"
    sample = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        export_onnx.validate_classification_outputs(model(sample).numpy(), runtime.logits(sample.numpy()))
    for name, value in model.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), before[name].numpy())
