"""EfficientNet export retries must verify the actual replacement artifact."""
import json
from unittest.mock import patch

import pytest
import torch

import export_onnx
from checkpoint import make_checkpoint_metadata
from efficientnet import EfficientNet


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("reject_original", [False, True])
def test_retry_verifies_original_graph_or_preserves_existing_files(tmp_path, reject_original):
    model = EfficientNet("efficientnet_b0", 2, 1).eval()
    checkpoint = make_checkpoint_metadata("classify", 2, ["OK", "NG"], (224, 224), 1)
    checkpoint.update(engine="efficientnet", model_config=model.checkpoint_config(),
                      model_state_dict=model.state_dict())
    source = tmp_path / "model.pt"
    torch.save(checkpoint, source)
    output = tmp_path / "model.onnx"
    output.write_bytes(b"previous onnx")
    output.with_suffix(".json").write_bytes(b"previous config")
    real_verify = export_onnx.verify_onnx
    attempts = []

    def verify(path, probe, candidate, **kwargs):
        fused = candidate.inference_optimization["conv_bn_fused"]
        attempts.append(fused)
        if fused or reject_original:
            raise ValueError("injected output mismatch")
        return real_verify(path, probe, candidate, **kwargs)

    with patch.object(export_onnx, "verify_onnx", side_effect=verify):
        if reject_original:
            with pytest.raises(ValueError, match="모두 ONNX 검증 실패"):
                export_onnx.export_checkpoint(source, output, dynamic_batch=True)
            assert output.read_bytes() == b"previous onnx"
            assert output.with_suffix(".json").read_bytes() == b"previous config"
            diagnostic = json.loads(output.with_suffix(".export-error.json").read_text(encoding="utf-8"))
            assert diagnostic["numerical_diagnostic"]["probe"] == "seeded"
            # The failure was injected in verify_onnx, so independent actual
            # ORT comparisons correctly report that it cannot be reproduced.
            assert diagnostic["numerical_diagnostic"]["finding"] == "not_reproduced_in_diagnostic"
        else:
            result = export_onnx.export_checkpoint(source, output, dynamic_batch=True)
            assert result["verification"] == "passed"
            metadata = json.loads(output.with_suffix(".json").read_text())
            assert metadata["export"]["optimization"]["fallback"] == "unfused_fp32"
            assert attempts[1:] == [0, 0, 0]
    assert attempts[0] > 0 and attempts[1] == 0
