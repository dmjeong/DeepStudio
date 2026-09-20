import copy
import json

import numpy as np
import pytest
import torch

from efficientnet import EfficientNet
from export_onnx import export_to_onnx
from onnx_diagnostics import comparison, conclusion, diagnose_efficientnet


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_numerical_summary_distinguishes_rank_and_scale():
    result = comparison(np.array([[2800., -1.]]), np.array([[2828., -1.]]), classification=True)
    assert not result["passed"] and result["top1_equal"]
    assert result["max_abs_error"] == 28
    assert result["failed_elements"] == 1
    assert not comparison(np.array([[0., 1e-5]]), np.array([[1e-5, 0.]]), classification=True)["passed"]
    assert not comparison(np.array([np.inf]), np.array([np.inf]))["passed"]


def test_conclusions_require_measured_evidence():
    assert conclusion({}) == "unresolved_graph_or_runtime_difference"
    assert conclusion({"onnx_sessions": {"all": {"passed": False}, "disabled": {"passed": True}}}) == "ort_optimization_difference"
    assert conclusion({"onnx_sessions": {"all_default": {"passed": False}, "all": {"passed": True}}}) == "ort_thread_difference"
    assert conclusion({"onnx_sessions": {"all": {"error": "failed to load"}, "disabled": {"passed": True}}}) == "unresolved_graph_or_runtime_difference"
    assert conclusion({"pytorch_fp64_vs_fp32": {"passed": False}}) == "pytorch_precision_sensitive"
    assert conclusion({"state": {"nonfinite_tensors": ["bn.weight"]}}) == "invalid_model_state"
    assert conclusion({"pytorch_repeat": {"passed": False, "finite": False}}) == "pytorch_nonfinite_output"


def test_real_stage_diagnostics_do_not_modify_model_or_publish_graph(tmp_path):
    torch.manual_seed(42)
    model = EfficientNet(num_classes=2, in_channels=1).eval()
    before = copy.deepcopy(model.state_dict())
    sample = torch.randn(1, 1, 224, 224)
    path = tmp_path / "model.onnx"
    export_to_onnx(model, sample, path, constant_folding=False)
    original_bytes = path.read_bytes()
    report = diagnose_efficientnet(model, path, sample, probe_name="seeded")
    assert "diagnostic_error" not in report and "stage_diagnostic_error" not in report
    assert report["finding"] == "not_reproduced_in_diagnostic"
    assert all(row["passed"] for row in report["onnx_sessions"].values())
    assert len(report["stages"]) == 9 and report["first_divergent_stage"] is None
    assert report["instrumentation_vs_original_onnx"]["passed"]
    assert report["pytorch_fp64_vs_fp32"]["passed"]
    assert report["requires_retraining"] == "not_determined"
    json.dumps(report, allow_nan=False)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    assert not any(module._forward_hooks for module in model.modules())
    assert path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [path]


def test_diagnostic_distinguishes_modified_artifact_from_instrumented_graph(tmp_path):
    import onnx

    class SmallModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1), torch.nn.ReLU())
            self.classifier = torch.nn.Linear(2, 2)

        def forward(self, images):
            return self.classifier(self.features(images).mean(dim=(2, 3)))

    model = SmallModel().eval()
    sample = torch.ones(1, 1, 16, 16)
    path = tmp_path / "modified.onnx"
    export_to_onnx(model, sample, path)
    graph = onnx.load(path)
    gemm = next(node for node in graph.graph.node if node.op_type == "Gemm")
    bias = next(value for value in graph.graph.initializer if value.name == gemm.input[2])
    bias.CopyFrom(onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(bias) + 28, name=bias.name))
    onnx.save(graph, path)
    report = diagnose_efficientnet(model, path, sample, probe_name="zero")
    assert all(not value["passed"] for value in report["onnx_sessions"].values())
    assert report["instrumented_final"]["passed"]
    assert not report["instrumentation_vs_original_onnx"]["passed"]
    assert report["first_divergent_stage"] is None
    assert report["finding"] == "unresolved_graph_or_runtime_difference"
