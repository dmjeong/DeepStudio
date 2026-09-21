import copy
import json
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

import export_onnx
from classification_export_validation import compare_classification, collect_images
from checkpoint import make_checkpoint_metadata
from efficientnet import EfficientNet


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_classification_policy_is_distinct_from_strict_logits():
    ref = np.array([[2., -2.]])
    actual = ref + .02
    with pytest.raises(ValueError):
        export_onnx.validate_classification_outputs(ref, actual)
    report = compare_classification(ref, actual)
    assert report["max_logit_error"] > .019
    assert report["max_probability_error"] < 1e-12


@pytest.mark.parametrize("ref,out", [([[1., 0.]], [[0., 1.]]),
                                      ([[1., 0.]], [[3., 0.]]),
                                      ([[.0001, 0.]], [[.0001, -.0002]]),
                                      ([[1., 0.]], [[float('nan'), 0.]])])
def test_wrong_rank_confidence_boundary_or_nonfinite_never_pass(ref, out):
    with pytest.raises(ValueError):
        compare_classification(ref, out)


@pytest.mark.parametrize("corrupt", [False, True])
def test_real_images_choose_fp32_and_reload_or_preserve_existing_file(tmp_path, corrupt):
    from onnx_classifier import OnnxClassifier
    names = ["a", "b"]
    data = tmp_path / "images"
    for index, name in enumerate(names):
        (data / name).mkdir(parents=True)
        cv2.imwrite(str(data / name / "sample.png"), np.full((32, 32), index * 255, np.uint8))
    model = EfficientNet(num_classes=2).eval()
    with torch.no_grad():
        model.classifier[1].bias.copy_(torch.tensor([2., -2.]))
    checkpoint = make_checkpoint_metadata("classify", 2, names, (32, 32), 1)
    checkpoint.update(engine="efficientnet", model_config=model.checkpoint_config(), model_state_dict=model.state_dict())
    source, output = tmp_path / "model.pt", tmp_path / "model.onnx"
    torch.save(checkpoint, source)
    output.write_bytes(b"previous")
    real_export = export_onnx.export_to_onnx

    def altered_export(candidate, dummy, path, *args, **kwargs):
        candidate = copy.deepcopy(candidate)
        with torch.no_grad():
            candidate.classifier[1].bias.add_(torch.tensor([-5., 5.]) if corrupt else .02)
        return real_export(candidate, dummy, path, *args, **kwargs)

    with patch.object(export_onnx, "export_to_onnx", side_effect=altered_export), \
            patch("efficientnet_precision.prepare_precision_export", side_effect=AssertionError("FP64 must not run")):
        if corrupt:
            with pytest.raises(ValueError, match="통과한 FP32 후보가 없습니다"):
                export_onnx.export_checkpoint(source, output, validation_dir=data, log=lambda _: None)
            assert output.read_bytes() == b"previous"
            return
        result = export_onnx.export_checkpoint(source, output, validation_dir=data, dynamic_batch=True, log=lambda _: None)
    report = result["classification_validation"]
    assert report["image_count"] == 2 and report["selected"]["passed"]
    assert report["selected"]["max_logit_error"] > .019
    assert report["selected"]["max_probability_error"] < .001
    config = json.loads(output.with_suffix(".json").read_text())
    assert config["export"]["verification_policy"] == "classification_dataset_v1"
    assert config["export"]["compute_precision"] == "float32"
    assert "verification_tolerance" not in config["export"]
    classifier = OnnxClassifier(result["config_path"])
    assert classifier.predict_gray(np.zeros((32, 32), np.uint8))["class_id"] == 0
    assert str(data) not in json.dumps(report)


def test_missing_class_cannot_silently_certify_partial_corpus(tmp_path):
    (tmp_path / "a").mkdir()
    cv2.imwrite(str(tmp_path / "a" / "sample.png"), np.zeros((8, 8), np.uint8))
    with pytest.raises(ValueError, match="클래스 하위"):
        collect_images(tmp_path, ["a", "b"])


def test_no_automatic_whole_fp64_export(tmp_path):
    model = EfficientNet(num_classes=2).eval()
    checkpoint = make_checkpoint_metadata("classify", 2, ["a", "b"], (32, 32), 1)
    checkpoint.update(engine="efficientnet", model_config=model.checkpoint_config(), model_state_dict=model.state_dict())
    source = tmp_path / "model.pt"
    torch.save(checkpoint, source)
    with patch.object(export_onnx, "verify_onnx", side_effect=ValueError("mismatch")), \
            patch("efficientnet_precision.prepare_precision_export", side_effect=AssertionError("must not run")) as precision:
        with pytest.raises(ValueError, match="모두 ONNX 검증 실패"):
            export_onnx.export_checkpoint(source, tmp_path / "model.onnx", log=lambda _: None)
        precision.assert_not_called()
