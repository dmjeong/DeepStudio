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
@pytest.mark.parametrize("input_size", [8, (18, 20)])
def test_patchcore_wrapper_exports_and_verifies_fixed_bank(tmp_path, input_size):
    import torch
    import torch.nn as nn
    from export_patchcore_onnx import export_patchcore_model
    from patchcore import STANDARD_SCORE_DEFINITION

    class TinyBackbone(nn.Module):
        def forward(self, images):
            return images[:, :2, ::4, ::4]

    class TinyPatchCore:
        backbone = TinyBackbone()
        _avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
        memory_bank = torch.randn(4, 2, generator=torch.Generator().manual_seed(7))
        n_neighbors = 2
        anomaly_threshold = 0.1
        score_definition = STANDARD_SCORE_DEFINITION
        center_crop = {"width": 20, "height": 18}
        preprocessing = "opencv_full_range_v2"

    model = TinyPatchCore()
    model.input_size = input_size
    result = export_patchcore_model(model, tmp_path / "patchcore.onnx")
    assert result["verification"] == "passed"
    assert result["cpp_supported"] is True
    assert (tmp_path / "patchcore.onnx").is_file()
    config = (tmp_path / "patchcore.json")
    assert config.is_file()
    manifest = json.loads(config.read_text())
    import onnxruntime as ort
    session = ort.InferenceSession(result["output_path"], providers=["CPUExecutionProvider"])
    assert manifest["input_name"] == session.get_inputs()[0].name
    assert manifest["output_names"] == [value.name for value in session.get_outputs()]
    assert manifest["output_name"] == manifest["output_names"][0]
    assert session.get_inputs()[0].shape[2:] == ([8, 8] if input_size == 8 else [18, 20])
    assert manifest["preprocessing"].get("original_size_region") == (None if input_size == 8 else [20, 18])
    assert manifest["schema_version"] == 5
    assert manifest["export"]["opset"] == 17
    assert manifest["export"]["verification_tolerance"] == {"atol": 1e-3, "rtol": 1e-3}
    assert manifest["postprocessing"]["score"] == STANDARD_SCORE_DEFINITION
    assert manifest["postprocessing"]["n_neighbors"] == 2
    assert manifest["preprocessing"]["center_crop"] == {"width": 20, "height": 18}
    assert manifest["preprocessing"]["value_range"] == "uint8_0_255_or_uint16_0_65535"
    assert result["verification_tolerance"] == {"atol": 1e-3, "rtol": 1e-3}


@pytest.mark.skipif(not all(importlib.util.find_spec(name) for name in ("torch", "onnx", "onnxruntime")),
                    reason="ONNX export dependencies required")
def test_patchcore_export_rejects_uncalibrated_model_and_clamps_neighbors(tmp_path):
    import torch
    import torch.nn as nn
    from export_patchcore_onnx import export_patchcore_model
    from patchcore import STANDARD_SCORE_DEFINITION

    class TinyBackbone(nn.Module):
        def forward(self, images):
            return images[:, :2, ::4, ::4]

    class UncalibratedPatchCore:
        backbone = TinyBackbone()
        _avg_pool = nn.AvgPool2d(3, stride=1, padding=1)
        memory_bank = torch.randn(4, 2, generator=torch.Generator().manual_seed(19))
        n_neighbors = 9
        input_size = 8
        anomaly_threshold = None
        score_definition = STANDARD_SCORE_DEFINITION
        center_crop = None
        preprocessing = "opencv_full_range_v2"

    target = tmp_path / "uncalibrated.onnx"
    with pytest.raises(ValueError, match="미보정"):
        export_patchcore_model(UncalibratedPatchCore(), target)
    assert not target.exists()
    UncalibratedPatchCore.anomaly_threshold = .2
    result = export_patchcore_model(UncalibratedPatchCore(), target)
    manifest = json.loads(Path(result["config_path"]).read_text())
    assert manifest["postprocessing"]["n_neighbors"] == 4
