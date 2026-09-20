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
            metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            assert metadata["export"]["optimization"]["fallback"] == "unfused_fp32"
            assert attempts[1:] == [0, 0, 0]
    assert attempts[0] > 0 and attempts[1] == 0


@pytest.mark.parametrize("passing_level", ["basic", "disabled"])
def test_runtime_fallback_is_verified_saved_and_used_by_deployment(tmp_path, passing_level):
    import onnxruntime as ort
    import numpy as np
    from onnx_classifier import OnnxClassifier
    from model_runtime.deployment_bundle import verify_deployment_bundle

    model = EfficientNet("efficientnet_b0", 2, 1).eval()
    checkpoint = make_checkpoint_metadata("classify", 2, ["OK", "NG"], (224, 224), 1)
    checkpoint.update(engine="efficientnet", model_config=model.checkpoint_config(),
                      model_state_dict=model.state_dict())
    source, output = tmp_path / "model.pt", tmp_path / "model.onnx"
    torch.save(checkpoint, source)
    create_session = ort.InferenceSession
    levels = {"basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
              "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL}
    passing = levels[passing_level]
    probes = []

    class FaultyOptimization:
        def __init__(self, session, level):
            self.session, self.level = session, level

        def __getattr__(self, name):
            return getattr(self.session, name)

        def run(self, names, feed, *args, **kwargs):
            outputs = self.session.run(names, feed, *args, **kwargs)
            tensor = next(iter(feed.values()))
            probes.append((self.level, tensor.shape[0], bool(np.all(tensor == 0))))
            if self.level != passing:
                outputs[0] = outputs[0] + 1.0
            return outputs

    def factory(path, options=None, **kwargs):
        options = options or ort.SessionOptions()
        return FaultyOptimization(create_session(path, options, **kwargs), options.graph_optimization_level)

    with patch.object(ort, "InferenceSession", side_effect=factory):
        result = export_onnx.export_checkpoint(source, output, dynamic_batch=True,
                                               bundle_output=tmp_path / "model.dvdeploy", log=lambda _: None)
        assert result["verification"] == "passed"
        config = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
        assert config["schema_version"] == 6
        assert config["onnxruntime"]["graph_optimization_level"] == passing_level
        assert config["num_threads"] == 1
        assert (passing, 1, False) in probes and (passing, 1, True) in probes and (passing, 2, False) in probes
        assert verify_deployment_bundle(result["bundle_path"])["verification"] == "passed"
        classifier = OnnxClassifier(result["config_path"])
        sample = torch.randn(1, 1, 224, 224)
        with torch.no_grad():
            expected = model(sample).numpy()
        export_onnx.validate_classification_outputs(expected, classifier.logits(sample.numpy()))
        assert classifier.session.get_session_options().graph_optimization_level == passing
        assert classifier.session.get_session_options().intra_op_num_threads == 1


def test_one_passing_probe_does_not_publish_fallback(tmp_path):
    model = EfficientNet("efficientnet_b0", 2, 1).eval()
    checkpoint = make_checkpoint_metadata("classify", 2, ["OK", "NG"], (224, 224), 1)
    checkpoint.update(engine="efficientnet", model_config=model.checkpoint_config(),
                      model_state_dict=model.state_dict())
    source, output = tmp_path / "model.pt", tmp_path / "model.onnx"
    torch.save(checkpoint, source)
    output.write_bytes(b"previous onnx")
    output.with_suffix(".json").write_bytes(b"previous config")
    verify = export_onnx.verify_onnx

    def reject_zero(path, probe, candidate, runtime_settings=None, **kwargs):
        if runtime_settings is None or runtime_settings["graph_optimization_level"] != "disabled":
            raise ValueError("injected optimized output mismatch")
        if torch.count_nonzero(probe) == 0:
            raise ValueError("injected zero input mismatch")
        return verify(path, probe, candidate, runtime_settings=runtime_settings, **kwargs)

    with patch.object(export_onnx, "verify_onnx", side_effect=reject_zero):
        with pytest.raises(ValueError, match="zero input mismatch"):
            export_onnx.export_checkpoint(source, output, log=lambda _: None)
    assert output.read_bytes() == b"previous onnx"
    assert output.with_suffix(".json").read_bytes() == b"previous config"
    report = json.loads(output.with_suffix(".export-error.json").read_text(encoding="utf-8"))
    assert report["numerical_diagnostic"]["probe"] == "zero"
