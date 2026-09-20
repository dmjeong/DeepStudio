import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import numpy as np
import pytest

from onnx_classifier import OnnxClassifier
from onnx_session import deployment_session_settings


@pytest.mark.parametrize("config", [
    {"schema_version": 6},
    {"schema_version": 6, "num_threads": 1, "onnxruntime": {}},
    {"schema_version": 6, "num_threads": 1, "onnxruntime": {"graph_optimization_level": "off"}},
    {"schema_version": 6, "num_threads": True, "onnxruntime": {"graph_optimization_level": "disabled"}},
    {"schema_version": 6, "num_threads": -1, "onnxruntime": {"graph_optimization_level": "disabled"}},
    {"schema_version": 6, "num_threads": 2**32, "onnxruntime": {"graph_optimization_level": "disabled"}},
])
def test_rejects_missing_or_invalid_verified_settings(config):
    with pytest.raises(ValueError):
        deployment_session_settings(config)


def test_deployment_uses_disabled_optimization_with_real_fp32_cancellation(tmp_path):
    script = Path(__file__).resolve().parents[1] / "cpp" / "tests" / "make_models.py"
    with patch.object(sys, "argv", [str(script), str(tmp_path)]):
        runpy.run_path(str(script), run_name="__main__")
    path = tmp_path / "runtime_optimization.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(backend="efficientnet", postprocessing={"output": "logits"})
    path.write_text(json.dumps(config), encoding="utf-8")
    sample = np.ones((1, 1, 2, 3), dtype=np.float32)
    disabled = OnnxClassifier(path).logits(sample)
    np.testing.assert_allclose(disabled, [[1., -1.]], atol=1e-6, rtol=0)
    config["onnxruntime"]["graph_optimization_level"] = "all"
    path.write_text(json.dumps(config), encoding="utf-8")
    optimized = OnnxClassifier(path).logits(sample)
    assert np.max(np.abs(disabled - optimized)) > 0.1
