"""Numerical regression for specialized FP64 Conv with dynamic ONNX batches."""
import copy

import numpy as np
import onnxruntime as ort
import pytest
import torch

from efficientnet_precision import _DoubleConv
from export_onnx import export_to_onnx


@pytest.mark.parametrize("kernel,stride,padding,dilation,groups,out_channels,bias", [
    (1, 1, 0, 1, 1, 8, True),
    (1, 2, 0, 1, 2, 8, False),
    (3, 2, 1, 1, 4, 4, True),
    (5, 1, 2, 2, 4, 4, False),
    ((3, 5), (1, 2), (2, 2), (2, 1), 4, 4, True),
])
def test_specialized_fp64_conv_preserves_values_and_dynamic_batch(
        tmp_path, kernel, stride, padding, dilation, groups, out_channels, bias):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(315)
        conv = torch.nn.Conv2d(4, out_channels, kernel, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias).double().eval()
        lowered = _DoubleConv(copy.deepcopy(conv)).eval()
        path = tmp_path / "specialized.onnx"
        export_to_onnx(lowered, torch.zeros(1, 4, 19, 17, dtype=torch.float64),
            path, dynamic_batch=True, constant_folding=False)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 2
        runtime = ort.InferenceSession(str(path), options,
            providers=["CPUExecutionProvider"])
        # Trace batch=1, execute several nonidentical images to expose accidental
        # fixed-batch reshape/indexing and signed cancellation errors.
        for batch in (1, 3):
            values = torch.randn(batch, 4, 19, 17, dtype=torch.float64) * 1000
            with torch.no_grad():
                expected = conv(values).numpy()
                pytorch_actual = lowered(values).numpy()
            actual = runtime.run(None, {"input_image": values.numpy()})[0]
            np.testing.assert_allclose(pytorch_actual, expected, rtol=1e-11, atol=1e-10)
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-10)
    finally:
        torch.set_num_threads(previous_threads)
