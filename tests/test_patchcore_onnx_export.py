"""PatchCore export includes the frozen bank and score/map postprocessing."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


@pytest.mark.skipif(not all(importlib.util.find_spec(name) for name in ("torch", "onnx", "onnxruntime")),
                    reason="ONNX export dependencies required")
def test_patchcore_wrapper_exports_and_verifies_fixed_bank(tmp_path):
    import torch
    import torch.nn as nn
    from export_patchcore_onnx import export_patchcore_model

    class TinyBackbone(nn.Module):
        def forward(self, images):
            return images[:, :2, ::4, ::4]

    class TinyPatchCore:
        backbone = TinyBackbone()
        _avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
        memory_bank = torch.randn(4, 2, generator=torch.Generator().manual_seed(7))
        n_neighbors = 2
        input_size = 8
        anomaly_threshold = 0.1

    result = export_patchcore_model(TinyPatchCore(), tmp_path / "patchcore.onnx")
    assert result["verification"] == "passed"
    assert result["cpp_supported"] is True
    assert (tmp_path / "patchcore.onnx").is_file()
    config = (tmp_path / "patchcore.json")
    assert config.is_file()
    manifest = json.loads(config.read_text())
    assert manifest["schema_version"] == 5
    assert manifest["export"]["opset"] == 17
    assert manifest["export"]["verification_tolerance"] == {"atol": 1e-3, "rtol": 1e-3}
    assert result["verification_tolerance"] == {"atol": 1e-3, "rtol": 1e-3}
