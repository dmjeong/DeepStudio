"""SCTNet must not need timm solely for stochastic depth."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from segmentation_backbones.sctnet import DropPath


@pytest.mark.parametrize("probability", [0., .1, .5, 1.])
@pytest.mark.parametrize("training", [True, False])
def test_drop_path_matches_previous_implementation(probability, training):
    PreviousDropPath = pytest.importorskip("timm.layers").DropPath

    x = torch.randn(8, 4, 7, 5, requires_grad=True)
    old, new = PreviousDropPath(probability).train(training), DropPath(probability).train(training)
    torch.manual_seed(53)
    expected = old(x)
    expected.sum().backward()
    gradient = x.grad.clone()
    x.grad = None
    torch.manual_seed(53)
    actual = new(x)
    actual.sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, gradient, rtol=0, atol=0)
    assert list(new.state_dict()) == list(old.state_dict()) == []


def test_sctnet_train_restore_and_fp32_int8_without_timm(tmp_path):
    script = r'''
import importlib.abc
import sys

class BlockTimm(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "timm" or fullname.startswith("timm."):
            raise ModuleNotFoundError("timm intentionally unavailable", name="timm")

sys.meta_path.insert(0, BlockTimm())
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from builtin_models import build_builtin_model, make_builtin_checkpoint, load_builtin_checkpoint
from segmentation_loss import SegmentationLoss

torch.set_num_threads(2)
torch.manual_seed(13)
root = Path(sys.argv[1])
for channels in (1, 3):
    model = build_builtin_model("sctnet_s", 3, channels)
    x = torch.randn(2, channels, 64, 64)
    target = torch.zeros(2, 64, 64, dtype=torch.long)
    target[:, 10:20, 15:35] = 1
    target[:, 32:45, 40:50] = 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = SegmentationLoss(3)
    for _ in range(2):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        optimizer.step()
    model.eval()
    checkpoint = make_builtin_checkpoint("sctnet_s", model, num_classes=3,
        input_size=64, in_channels=channels, class_names=["background", "crack", "fm"])
    checkpoint_path = root / f"sctnet_{channels}.pt"
    torch.save(checkpoint, checkpoint_path)
    restored = load_builtin_checkpoint(torch.load(checkpoint_path, weights_only=False)).eval()
    x = x[:1]
    with torch.inference_mode():
        expected = model(x)
        torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
    path = root / f"sctnet_{channels}.onnx"
    torch.onnx.export(restored, x, str(path), input_names=["image"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"image": x.numpy()})[0]
    np.testing.assert_allclose(actual, expected.numpy(), rtol=5e-4, atol=1e-4)
    class Reader(CalibrationDataReader):
        def __init__(self):
            self.rows = iter([{"image": x.numpy()}, {"image": np.zeros_like(x.numpy())}])
        def get_next(self):
            return next(self.rows, None)
    quantized = root / f"sctnet_{channels}_int8.onnx"
    quantize_static(str(path), str(quantized), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    int8 = ort.InferenceSession(str(quantized), options, providers=["CPUExecutionProvider"])
    output = int8.run(None, {"image": x.numpy()})[0]
    assert output.shape == tuple(expected.shape) and np.isfinite(output).all()
assert not any(name == "timm" or name.startswith("timm.") for name in sys.modules)
print("NO_TIMM_TRAIN_RESTORE_FP32_INT8_OK")
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "python"),
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=150, check=False)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-5000:]
    assert "NO_TIMM_TRAIN_RESTORE_FP32_INT8_OK" in result.stdout
